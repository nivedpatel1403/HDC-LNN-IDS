import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

DATA_KEYS = ("dataset_name", "entity_id_column", "entity_id_modulo", "cache_dir", "split_version")


class Config:
    """Configuration class that loads settings from YAML files with bidirectional data-block resolution."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        if item in self._data:
            val = self._data[item]
            if isinstance(val, dict):
                return Config(val)
            return val
        # Transparent fallback to nested data block
        if "data" in self._data and isinstance(self._data["data"], dict) and item in self._data["data"]:
            val = self._data["data"][item]
            if isinstance(val, dict):
                return Config(val)
            return val
        # If accessing .data when not explicitly present, return empty Config view
        if item == "data":
            return Config({})
        raise AttributeError(f"No such configuration key: {item}")

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            return self._data[key]
        if "data" in self._data and isinstance(self._data["data"], dict) and key in self._data["data"]:
            return self._data["data"][key]
        return default

    def to_dict(self) -> Dict[str, Any]:
        return self._data


def _normalize_data_config(config_data: Dict[str, Any]) -> None:
    """Synchronizes dataset configuration keys bidirectionally between document root and 'data' block."""
    if "data" not in config_data or not isinstance(config_data["data"], dict):
        config_data["data"] = {}
    data_section = config_data["data"]

    for key in DATA_KEYS:
        # Override values defined at root take precedence over base data-block values
        if key in config_data:
            data_section[key] = config_data[key]
        elif key in data_section:
            config_data[key] = data_section[key]


def load_config(base_path: Union[str, Path], override_path: Optional[Union[str, Path]] = None) -> Config:
    """Loads a base configuration YAML and merges it with an optional override configuration."""
    base_p = Path(base_path)
    with open(base_p, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f) or {}

    if override_path:
        override_p = Path(override_path)
        if override_p.exists():
            with open(override_p, 'r', encoding='utf-8') as f:
                overrides = yaml.safe_load(f) or {}
            _deep_update(config_data, overrides)

    _normalize_data_config(config_data)
    return Config(config_data)


def resolve_experiment_config(
    cfg_path: Optional[Union[str, Path]] = None,
    dataset_hint: Optional[str] = None,
    dataset_file: Optional[Union[str, Path]] = None,
    configs_dir: Union[str, Path] = Path("configs")
) -> Config:
    """Resiliently resolves and loads configuration.
    
    1. If cfg_path exists, loads it directly.
    2. Otherwise, matches dataset config in configs_dir and merges with configs/base.yaml.
    """
    if cfg_path:
        cp = Path(cfg_path)
        if cp.exists():
            return load_config(cp)

    configs_dir = Path(configs_dir)
    base_cfg = configs_dir / "base.yaml"
    ds_cfg: Optional[Path] = None

    if dataset_file and Path(dataset_file).exists():
        from hdlnn.common.dataset_utils import match_config_for_dataset
        ds_cfg = match_config_for_dataset(Path(dataset_file), configs_dir=configs_dir)
    elif dataset_hint:
        name_lower = dataset_hint.lower()
        if "kdd" in name_lower or "nsl" in name_lower:
            ds_cfg = configs_dir / "dataset_kdd.yaml"
        elif "ton" in name_lower:
            ds_cfg = configs_dir / "dataset_ton_iot.yaml"
        elif "botnet" in name_lower:
            ds_cfg = configs_dir / "dataset_unsw_botnet.yaml"
        elif "iot" in name_lower or "dataset" in name_lower:
            ds_cfg = configs_dir / "dataset_iot23.yaml"
        else:
            ds_cfg = configs_dir / "dataset_unsw_nb15.yaml"
    else:
        ds_cfg = configs_dir / "dataset_unsw_nb15.yaml"

    if base_cfg.exists() and ds_cfg and ds_cfg.exists():
        return load_config(base_cfg, ds_cfg)
    elif ds_cfg and ds_cfg.exists():
        return load_config(ds_cfg)
    elif base_cfg.exists():
        return load_config(base_cfg)
    else:
        raise FileNotFoundError(f"Neither candidate '{cfg_path}' nor fallback configs found in {configs_dir}.")


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Recursively updates base dictionary with overrides in-place."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
