"""Anomaly decision contract for final drift/anomaly classification."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class AnomalyDecision(BaseModel):
    """Contract for anomaly detection decision output."""

    model_config = ConfigDict(frozen=False)

    entity_id: str
    flow_id: str
    t: datetime
    drift_score: float
    threshold: float
    is_anomalous: bool
    manifold_version: str
    ground_truth_label: str
    attack_category: str | None = None
