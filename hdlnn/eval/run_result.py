"""RunResult container model for experiment results and artifacts."""

from typing import Any
from pydantic import BaseModel, Field

from hdlnn.contracts.decision import AnomalyDecision


class RunResult(BaseModel):
    """Container model for single-run evaluation results and metadata."""

    run_id: str
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    detection_metrics: dict[str, Any] = Field(default_factory=dict)
    latency_stats: dict[str, Any] = Field(default_factory=dict)
    memory_stats: dict[str, Any] = Field(default_factory=dict)
    decisions: list[AnomalyDecision] = Field(default_factory=list)
    git_commit_hash: str = "unknown"
