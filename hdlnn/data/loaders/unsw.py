"""UNSW-NB15 raw dataset loader implementing CanonicalFlow parsing."""

from datetime import datetime, timezone, timedelta
import math
from typing import Any
import pandas as pd
from pydantic import ValidationError

from hdlnn.common.logging import get_logger
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.data.interfaces import ILoader
from hdlnn.data.quality import DataQualityReport, log_quality_report

# Primary UNSW-NB15 raw column to CanonicalFlow field mapping table
UNSW_COLUMN_MAP: dict[str, str] = {
    "srcip": "src_ip",
    "dstip": "dst_ip",
    "sport": "src_port",
    "dsport": "dst_port",
    "proto": "protocol",
    "dur": "duration",
    "state": "state",
    "label": "label",
    "attack_cat": "attack_category",
}

# Categorical columns included in CanonicalFlow.categorical_features
UNSW_CATEGORICAL_COLS: set[str] = {"service", "proto", "state"}

# Columns excluded from CanonicalFlow.numeric_features dictionary
UNSW_EXCLUDE_NUMERIC: set[str] = {
    "srcip",
    "dstip",
    "sport",
    "dsport",
    "proto",
    "state",
    "dur",
    "service",
    "label",
    "attack_cat",
    "id",
    "flow_id",
    "stime",
    "ltime",
    "sttime",
    "lttime",
    "occurred_at",
}


class UNSWLoader(ILoader):
    """Dataset loader for UNSW-NB15 CSV files converting rows into CanonicalFlow objects."""

    def __init__(self, run_id: str | None = None) -> None:
        self.logger = get_logger("hdlnn.data.loaders.unsw", run_id=run_id)
        self.last_quality_report: DataQualityReport | None = None

    def load(self, raw_path: str) -> list[CanonicalFlow]:
        """Load raw UNSW-NB15 CSV dataset into a list of CanonicalFlow models.

        Args:
            raw_path: File path to the UNSW-NB15 CSV file.

        Returns:
            List of validated CanonicalFlow instances.
        """
        self.logger.info("Loading raw UNSW-NB15 dataset from %s", raw_path)

        df = pd.read_csv(raw_path, comment="#")
        # Normalize column headers to lowercase for robust matching
        df.columns = [col.strip().lower() for col in df.columns]

        flows: list[CanonicalFlow] = []
        seen_flow_ids: set[str] = set()

        missing_counts: dict[str, int] = {}
        type_coercions = 0
        duplicate_ids = 0
        unparseable_rows = 0
        total_rows = len(df)

        base_synthetic_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        for idx, row in df.iterrows():
            try:
                # 1. Flow ID determination
                raw_id = row.get("id")
                if pd.notna(raw_id) and str(raw_id).strip():
                    flow_id = f"unsw_{str(raw_id).strip()}"
                else:
                    flow_id = f"unsw_{idx}"

                if flow_id in seen_flow_ids:
                    duplicate_ids += 1
                    flow_id = f"{flow_id}_{idx}"
                seen_flow_ids.add(flow_id)

                # 2. IP Addresses
                src_ip = str(row.get("srcip", "")).strip()
                if not src_ip or src_ip.lower() in ("nan", "none"):
                    src_ip = "0.0.0.0"
                    missing_counts["src_ip"] = missing_counts.get("src_ip", 0) + 1

                dst_ip = str(row.get("dstip", "")).strip()
                if not dst_ip or dst_ip.lower() in ("nan", "none"):
                    dst_ip = "0.0.0.0"
                    missing_counts["dst_ip"] = missing_counts.get("dst_ip", 0) + 1

                # 3. Ports
                src_port = self._parse_int(row.get("sport"), "src_port", missing_counts, type_coercions)
                dst_port = self._parse_int(row.get("dsport"), "dst_port", missing_counts, type_coercions)

                # 4. Protocol & State
                proto = str(row.get("proto", "unknown")).strip()
                if not proto or proto.lower() in ("nan", "none"):
                    proto = "unknown"
                    missing_counts["protocol"] = missing_counts.get("protocol", 0) + 1

                state = str(row.get("state", "CON")).strip()
                if not state or state.lower() in ("nan", "none"):
                    state = "CON"
                    missing_counts["state"] = missing_counts.get("state", 0) + 1

                # 5. Duration
                dur_raw = row.get("dur")
                try:
                    dur = float(dur_raw) if pd.notna(dur_raw) else 0.0
                    if isinstance(dur_raw, str):
                        type_coercions += 1
                except (ValueError, TypeError):
                    dur = 0.0
                    type_coercions += 1
                    missing_counts["duration"] = missing_counts.get("duration", 0) + 1

                # 6. Label & Attack Category
                raw_label = row.get("label")
                if pd.notna(raw_label) and str(raw_label).strip() in ("1", "1.0", "attack"):
                    label = "attack"
                else:
                    label = "normal"

                raw_attack_cat = row.get("attack_cat")
                attack_category = None
                if label == "attack" and pd.notna(raw_attack_cat):
                    cat_str = str(raw_attack_cat).strip()
                    if cat_str and cat_str.lower() not in ("normal", "-", "nan", "none", "0"):
                        attack_category = cat_str

                # 7. Occurred At (Timestamp)
                # UNSW-NB15 raw CSVs in public releases lack explicit timestamps in some splits.
                # Fallback: If Stime/Ltime columns are present, parse them. Otherwise, synthesize a
                # monotonically increasing timestamp based on row index starting from base_synthetic_time.
                occurred_at = self._parse_timestamp(row, idx, base_synthetic_time)

                # 8. Numeric Features
                numeric_features: dict[str, float] = {}
                for col_name, val in row.items():
                    if col_name in UNSW_EXCLUDE_NUMERIC:
                        continue
                    if pd.notna(val):
                        try:
                            fval = float(val)
                            if not math.isnan(fval):
                                numeric_features[col_name] = fval
                        except (ValueError, TypeError):
                            pass

                # 9. Categorical Features
                categorical_features: dict[str, str] = {
                    "service": str(row.get("service", "-")).strip(),
                    "proto": proto,
                    "state": state,
                }

                flow = CanonicalFlow(
                    flow_id=flow_id,
                    dataset_source="UNSW-NB15",
                    occurred_at=occurred_at,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port[0],
                    dst_port=dst_port[0],
                    protocol=proto,
                    state=state,
                    duration=dur,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    label=label,
                    attack_category=attack_category,
                    split=None,  # Set exclusively by Splitter later
                )
                flows.append(flow)

                # Accumulate coercion counts from port parsing tuple
                type_coercions += src_port[1] + dst_port[1]

            except (ValidationError, Exception) as err:
                self.logger.debug("Failed to parse row %d: %s", idx, err)
                unparseable_rows += 1

        self.last_quality_report = DataQualityReport(
            missing_field_counts=missing_counts,
            type_coercion_count=type_coercions,
            duplicate_flow_id_count=duplicate_ids,
            unparseable_row_count=unparseable_rows,
            total_rows=total_rows,
        )

        log_quality_report(self.logger, self.last_quality_report)
        return flows

    def _parse_int(
        self, val: Any, field_name: str, missing_counts: dict[str, int], coercions: int
    ) -> tuple[int, int]:
        """Helper to parse integer values (e.g. ports) with fallback and coercion tracking."""
        coerced = 0
        if pd.isna(val) or val is None or str(val).strip() == "":
            missing_counts[field_name] = missing_counts.get(field_name, 0) + 1
            return 0, coerced

        try:
            if isinstance(val, str):
                coerced = 1
            return int(float(val)), coerced
        except (ValueError, TypeError):
            missing_counts[field_name] = missing_counts.get(field_name, 0) + 1
            coerced = 1
            return 0, coerced

    def _parse_timestamp(self, row: pd.Series, idx: int, base_time: datetime) -> datetime:
        """Parse native timestamp if present (Stime/Ltime), or fallback to monotonic row sequence."""
        for ts_col in ("stime", "ltime", "sttime", "lttime"):
            raw_ts = row.get(ts_col)
            if pd.notna(raw_ts):
                try:
                    ts_float = float(raw_ts)
                    return datetime.fromtimestamp(ts_float, tz=timezone.utc)
                except (ValueError, TypeError, OverflowError):
                    pass

        # Fallback: Monotonic row-index sequence (1 second steps)
        return base_time + timedelta(seconds=idx)
