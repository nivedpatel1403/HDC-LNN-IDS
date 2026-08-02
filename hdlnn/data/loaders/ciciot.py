"""CICIoT2023 raw dataset loader implementing CanonicalFlow parsing."""

from datetime import datetime, timezone, timedelta
import math
from typing import Any
import pandas as pd
from pydantic import ValidationError

from hdlnn.common.logging import get_logger
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.data.interfaces import ILoader
from hdlnn.data.quality import DataQualityReport, log_quality_report

# Standard CICIoT2023 column mapping table
CICIOT_COLUMN_MAP: dict[str, str] = {
    "header_length": "header_length",
    "protocol type": "protocol",
    "time_to_live": "ttl",
    "rate": "rate",
    "label": "label",
}

# Categorical columns included in CanonicalFlow.categorical_features
CICIOT_CATEGORICAL_COLS: set[str] = {"protocol", "protocol_type"}

# Columns excluded from CanonicalFlow.numeric_features dictionary
CICIOT_EXCLUDE_NUMERIC: set[str] = {
    "label",
    "attack_category",
    "id",
    "flow_id",
    "occurred_at",
    "srcip",
    "dstip",
    "sport",
    "dsport",
}

PROTO_MAP: dict[int, str] = {
    6: "TCP",
    17: "UDP",
    1: "ICMP",
    0: "HOPOPT",
    2: "IGMP",
    47: "GRE",
    50: "ESP",
    51: "AH",
}


class CICIoTLoader(ILoader):
    """Dataset loader for CICIoT2023 CSV files converting rows into CanonicalFlow objects.

    CICIoT2023 features consist of NetFlow-style statistical flow metrics rather than per-host IP headers.
    Per ADR-004 specification, entity grouping for CICIoT2023 degrades to per-row single-record sequences
    (flow_id = 'ciciot_{idx}').
    """

    def __init__(self, run_id: str | None = None) -> None:
        self.logger = get_logger("hdlnn.data.loaders.ciciot", run_id=run_id)
        self.last_quality_report: DataQualityReport | None = None

    def load(self, raw_path: str) -> list[CanonicalFlow]:
        """Load raw CICIoT2023 CSV dataset into a list of CanonicalFlow models.

        Args:
            raw_path: Absolute or relative file path to the CICIoT2023 CSV file.

        Returns:
            List of validated CanonicalFlow instances.
        """
        self.logger.info("Loading raw CICIoT2023 dataset from %s", raw_path)

        df = pd.read_csv(raw_path, comment="#")

        # Strip whitespace from column headers but retain original casing mapping reference
        col_raw_to_norm = {col: col.strip() for col in df.columns}
        col_norm_to_lower = {col.strip(): col.strip().lower() for col in df.columns}
        df.columns = [col.strip() for col in df.columns]

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
                flow_id = f"ciciot_{idx}"
                if flow_id in seen_flow_ids:
                    duplicate_ids += 1
                    flow_id = f"{flow_id}_{idx}"
                seen_flow_ids.add(flow_id)

                # 2. Protocol parsing
                raw_proto = row.get("Protocol Type", row.get("protocol_type"))
                proto_str = "unknown"
                if pd.notna(raw_proto):
                    try:
                        proto_val = int(float(raw_proto))
                        proto_str = PROTO_MAP.get(proto_val, str(proto_val))
                    except (ValueError, TypeError):
                        proto_str = str(raw_proto).strip()
                        type_coercions += 1

                # 3. Label & Attack Category determination
                raw_label = row.get("label", row.get("Label"))
                if pd.isna(raw_label):
                    # Check filename or fallback
                    raw_label = "normal"
                    missing_counts["label"] = missing_counts.get("label", 0) + 1

                label_str = str(raw_label).strip()
                if label_str.lower() in ("benigntraffic", "benign", "normal", "0", "0.0"):
                    label = "normal"
                    attack_category = None
                else:
                    label = "attack"
                    attack_category = label_str

                # 4. Monotonic synthetic timestamp
                occurred_at = base_synthetic_time + timedelta(seconds=idx)

                # 5. Numeric Features extraction
                numeric_features: dict[str, float] = {}
                for col_name, val in row.items():
                    if col_name in CICIOT_EXCLUDE_NUMERIC or col_norm_to_lower.get(col_name) in CICIOT_EXCLUDE_NUMERIC:
                        continue
                    if pd.notna(val):
                        try:
                            fval = float(val)
                            if not math.isnan(fval):
                                numeric_features[col_name] = fval
                        except (ValueError, TypeError):
                            pass

                # 6. Categorical Features
                categorical_features: dict[str, str] = {
                    "protocol": proto_str,
                }

                # 7. Duration extraction if present, else 0.0
                dur_raw = row.get("Duration", row.get("duration", 0.0))
                try:
                    dur = float(dur_raw) if pd.notna(dur_raw) else 0.0
                except (ValueError, TypeError):
                    dur = 0.0

                flow = CanonicalFlow(
                    flow_id=flow_id,
                    dataset_source="CICIOT2023",
                    occurred_at=occurred_at,
                    src_ip=f"ciciot_entity_{idx}",
                    dst_ip="0.0.0.0",
                    src_port=0,
                    dst_port=0,
                    protocol=proto_str,
                    state="CON",
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

