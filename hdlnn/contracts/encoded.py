"""Encoded hypervector contract for HDC representation."""

from typing import Literal
from pydantic import BaseModel, ConfigDict


class EncodedHypervector(BaseModel):
    """Contract for hypervector representations produced by HDC encoders."""

    model_config = ConfigDict(frozen=False)

    flow_id: str
    encoder_version: str
    dim: int
    representation: Literal["binary", "bipolar", "MAP"]
    vector_ref: str
    oov_fields: list[str]
