"""NSL-KDD raw dataset loader implementing CanonicalFlow parsing."""

from datetime import datetime, timezone, timedelta
import math
from typing import Any
import pandas as pd
from pydantic import ValidationError

from hdlnn.common.logging import get_logger
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.data.interfaces import ILoader
from hdlnn.data.quality import DataQualityReport, log_quality_report

# Standard 43 column names for unheadered NSL-KDD files (KDDTrain+.txt, KDDTest+.txt)
NSLKDD_COLUMN_NAMES: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

# Column mapping table
NSLKDD_COLUMN_MAP: dict[str, str] = {
    "protocol_type": "protocol",
    "service": "service",
    "flag": "state",
    "duration": "duration",
    "label": "label",
}

# Categorical columns
NSLKDD_CATEGORICAL_COLS: set[str] = {"service", "protocol_type", "flag", "protocol", "state"}

# Excluded from numeric_features
NSLKDD_EXCLUDE_NUMERIC: set[str] = {
    "label",
    "attack_category",
    "difficulty",
    "id",
    "flow_id",
    "occurred_at",
    "srcip",
    "dstip",
    "sport",
    "dsport",
    "protocol_type",
    "service",
    "flag",
    "protocol",
    "state",
}


class NSLKDDLoader(ILoader):
    """Dataset loader for NSL-KDD files converting rows into CanonicalFlow objects.

    NSL-KDD records are aggregated per-connection and do not include host IP addresses.
    Per ADR-004 specification, entity grouping for NSL-KDD defaults to per-record
    single-record sequences (flow_id = 'nslkdd_{idx}').
    """

    def __init__(self, run_id: str | None = None) -> None:
        self.logger = get_logger("hdlnn.data.loaders.nslkdd", run_id=run_id)
        self.last_quality_report: DataQualityReport | None = None

    def load(self, raw_path: str) -> list[CanonicalFlow]:
        """Load raw NSL-KDD dataset into a list of CanonicalFlow models.

        Args:
            raw_path: Absolute or relative file path to NSL-KDD raw file (KDDTrain+.txt, KDDTest+.txt, or CSV).

        Returns:
            List of validated CanonicalFlow instances.
        """
        self.logger.info("Loading raw NSL-KDD dataset from %s", raw_path)

        # Inspect first line to determine if header exists
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip().lower()

        has_header = "duration" in first_line or "protocol" in first_line

        if has_header:
            df = pd.read_csv(raw_path, comment="#")
            df.columns = [col.strip().lower() for col in df.columns]
        else:
            df = pd.read_csv(raw_path, header=None, comment="#")
            if len(df.columns) == len(NSLKDD_COLUMN_NAMES):
                df.columns = NSLKDD_COLUMN_NAMES
            elif len(df.columns) == 42:
                df.columns = NSLKDD_COLUMN_NAMES[:42]
            else:
                df.columns = [f"col_{i}" for i in range(len(df.columns))]
                df.rename(columns={0: "duration", 1: "protocol_type", 2: "service", 3: "flag", len(df.columns)-2: "label"}, inplace=True)

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
                flow_id = f"nslkdd_{idx}"
                if flow_id in seen_flow_ids:
                    duplicate_ids += 1
                    flow_id = f"{flow_id}_{idx}"
                seen_flow_ids.add(flow_id)

                # 2. Categoricals & Protocol
                proto = str(row.get("protocol_type", "tcp")).strip().upper()
                service = str(row.get("service", "http")).strip()
                flag = str(row.get("flag", "SF")).strip()

                # 3. Duration
                dur_raw = row.get("duration", 0.0)
                try:
                    dur = float(dur_raw) if pd.notna(dur_raw) else 0.0
                except (ValueError, TypeError):
                    dur = 0.0
                    type_coercions += 1
                    missing_counts["duration"] = missing_counts.get("duration", 0) + 1

                # 4. Label & Attack Category
                raw_label = row.get("label", "normal")
                label_clean = str(raw_label).strip().rstrip(".").lower()
                if label_clean in ("normal", "benign", "0", "0.0"):
                    label = "normal"
                    attack_category = None
                else:
                    label = "attack"
                    attack_category = str(raw_label).strip().rstrip(".")

                # 5. Timestamp
                occurred_at = base_synthetic_time + timedelta(seconds=idx)

                # 6. Numeric Features
                numeric_features: dict[str, float] = {}
                for col_name, val in row.items():
                    if col_name in NSLKDD_EXCLUDE_NUMERIC:
                        continue
                    if pd.notna(val):
                        try:
                            fval = float(val)
                            if not math.isnan(fval):
                                numeric_features[col_name] = fval
                        except (ValueError, TypeError):
                            pass

                # 7. Categorical Features
                categorical_features: dict[str, str] = {
                    "protocol": proto,
                    "service": service,
                    "state": flag,
                }

                flow = CanonicalFlow(
                    flow_id=flow_id,
                    dataset_source="NSL-KDD",
                    occurred_at=occurred_at,
                    src_ip=f"nslkdd_entity_{idx}",
                    dst_ip="0.0.0.0",
                    src_port=0,
                    dst_port=0,
                    protocol=proto,
                    state=flag,
                    duration=dur,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    label=label,
                    attack_category=attack_category,
                    split=None,
                )
                flows.append(flow)

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

