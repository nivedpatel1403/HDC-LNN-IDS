import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Literal

class Config:
    """Configuration class that loads settings from YAML files."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        if item in self._data:
            val = self._data[item]
            if isinstance(val, dict):
                return Config(val)
            return val
        raise AttributeError(f"No such configuration key: {item}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

def load_config(base_path: Path, override_path: Optional[Path] = None) -> Config:
    """Loads a base configuration YAML and merges it with an optional override configuration."""
    with open(base_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f) or {}

    if override_path and override_path.exists():
        with open(override_path, 'r', encoding='utf-8') as f:
            overrides = yaml.safe_load(f) or {}
        _deep_update(config_data, overrides)

    return Config(config_data)

def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Recursively updates base dictionary with overrides in-place."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# Backward compatibility aliases for legacy scripts and test harnesses
try:
    import os
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        run_id: str = "base_run_001"
        seed: int = 42
        log_level: str = "INFO"
        dataset_source: str = "unsw"
        dataset_raw_path: Optional[str] = None
        split_version: str = "v1"
        hdc_encoder: Literal["on", "off"] = "on"
        lnn_stage: Literal["on", "off"] = "on"
        sequence_model_type: Literal["cfc", "ltc", "lstm", "cnn"] = "cfc"
        epochs: int = 5
        lr: float = 1e-3
        target_fpr: float = 0.05
        hidden_size: int = 64
        dimensionality: int = 10000
        num_levels: int = 100
        window_size: int = 10
        max_sequence_length: int = 100
        device: str = "cpu"
        model_config = SettingsConfigDict(extra="ignore", env_file_encoding="utf-8")

    def load_settings(config_path: str) -> Settings:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(p, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        return Settings(**yaml_data)

except ImportError:
    pass
