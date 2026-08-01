"""Configuration settings loader using Pydantic Settings."""

import os
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from hdlnn.contracts.errors import ConfigError


class Settings(BaseSettings):
    """Application configuration settings for HD-LNN pipeline."""

    run_id: str = "base_run_001"
    seed: int = 42
    log_level: str = "INFO"

    # Dataset Source & Pipeline Settings
    dataset_source: str = "unsw"
    dataset_raw_path: str | None = None
    split_version: str = "v1"

    # Ablation Toggles (Phase 7b)
    hdc_encoder: Literal["on", "off"] = "on"
    lnn_stage: Literal["on", "off"] = "on"
    sequence_model_type: Literal["cfc", "ltc", "lstm", "cnn"] = "cfc"

    # Training & Evaluation Hyperparameters
    epochs: int = 5
    lr: float = 1e-3
    target_fpr: float = 0.05
    hidden_size: int = 64
    dimensionality: int = 10000
    num_levels: int = 100
    window_size: int = 10
    max_sequence_length: int = 100
    device: str = "cpu"

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file_encoding="utf-8",
    )


def load_settings(config_path: str) -> Settings:
    """Load configuration settings from a YAML file path with environment variable overrides.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Instantiated Settings object.

    Raises:
        ConfigError: If the configuration file cannot be found, parsed, or validated.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML configuration file at '{config_path}': {e}") from e

    if not isinstance(yaml_data, dict):
        raise ConfigError(f"YAML configuration at '{config_path}' must evaluate to a dictionary.")

    merged_kwargs: dict[str, Any] = dict(yaml_data)

    # Environment variables override YAML file settings
    for field in Settings.model_fields.keys():
        env_val = os.environ.get(field.upper()) or os.environ.get(field)
        if env_val is not None:
            merged_kwargs[field] = env_val

    try:
        return Settings(**merged_kwargs)
    except Exception as e:
        raise ConfigError(f"Failed to validate settings loaded from '{config_path}': {e}") from e
