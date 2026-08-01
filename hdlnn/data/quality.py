"""Data quality tracking, verification, and reporting utilities."""

from collections import Counter
from typing import Any
from pydantic import BaseModel, Field

from hdlnn.contracts.flow import CanonicalFlow


class DataQualityReport(BaseModel):
    """Container for data quality audit results and metrics."""

    missing_field_counts: dict[str, int] = Field(default_factory=dict)
    type_coercion_count: int = 0
    duplicate_flow_id_count: int = 0
    unparseable_row_count: int = 0
    total_rows: int = 0


def check_quality(flows: list[CanonicalFlow]) -> DataQualityReport:
    """Scan already-loaded CanonicalFlow list for duplicate flow_ids and basic quality metrics.

    Args:
        flows: List of loaded CanonicalFlow objects.

    Returns:
        DataQualityReport containing scanned metrics.
    """
    total = len(flows)
    flow_ids = [f.flow_id for f in flows]
    id_counts = Counter(flow_ids)
    duplicates = sum(count - 1 for count in id_counts.values() if count > 1)

    return DataQualityReport(
        total_rows=total,
        duplicate_flow_id_count=duplicates,
        missing_field_counts={},
        type_coercion_count=0,
        unparseable_row_count=0,
    )


def log_quality_report(logger: Any, report: DataQualityReport) -> None:
    """Log a single summary line for a DataQualityReport.

    Args:
        logger: Logger instance (or LoggerAdapter).
        report: DataQualityReport instance to summarize.
    """
    missing_str = (
        ", ".join(f"{k}:{v}" for k, v in report.missing_field_counts.items())
        if report.missing_field_counts
        else "none"
    )
    logger.info(
        "Data Quality Report - Total: %d, Unparseable: %d, Duplicates: %d, Coercions: %d, Missing: [%s]",
        report.total_rows,
        report.unparseable_row_count,
        report.duplicate_flow_id_count,
        report.type_coercion_count,
        missing_str,
    )
