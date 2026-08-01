"""Trajectory state contract for LNN sequence hidden states."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class TrajectoryState(BaseModel):
    """Contract for continuous trajectory states computed by LNN modules."""

    model_config = ConfigDict(frozen=False)

    entity_id: str
    flow_id: str
    t: datetime
    hidden_state_ref: str
    model_version: str
