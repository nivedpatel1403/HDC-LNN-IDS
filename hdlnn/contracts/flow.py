"""Canonical flow contract for network flow representation."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict


class CanonicalFlow(BaseModel):
    """Canonical network flow contract unifying dataset-specific formats."""

    model_config = ConfigDict(frozen=False)

    flow_id: str
    dataset_source: Literal["UNSW-NB15", "CICIOT2023", "NSL-KDD"]
    occurred_at: datetime
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    state: str
    duration: float
    numeric_features: dict[str, float]
    categorical_features: dict[str, str]
    label: str
    attack_category: str | None = None
    split: Literal["train", "val", "test"] | None = None
