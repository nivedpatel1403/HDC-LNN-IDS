from pathlib import Path
import pytest

from hdlnn.common.config import load_config, resolve_experiment_config, Config
from hdlnn.data.loaders import load_dataset_flows, load_raw_unsw_nb15


def test_config_bidirectional_normalization():
    """Verify bidirectional synchronization between document root and data block."""
    base_yaml = Path("configs/base.yaml")
    assert base_yaml.exists()

    # 1. TON-IoT override: defines entity_id_column='src_ip' at root
    ton_cfg = load_config(base_yaml, Path("configs/dataset_ton_iot.yaml"))
    assert ton_cfg.dataset_name == "ton_iot"
    assert ton_cfg.data.dataset_name == "ton_iot"
    assert ton_cfg.entity_id_column == "src_ip"
    assert ton_cfg.data.entity_id_column == "src_ip"
    assert ton_cfg.entity_id_modulo == 254
    assert ton_cfg.data.entity_id_modulo == 254

    # 2. IoT-23 override: defines entity_id_column='id.orig_h' at root
    iot_cfg = load_config(base_yaml, Path("configs/dataset_iot23.yaml"))
    assert iot_cfg.dataset_name == "iot23"
    assert iot_cfg.data.dataset_name == "iot23"
    assert iot_cfg.entity_id_column == "id.orig_h"
    assert iot_cfg.data.entity_id_column == "id.orig_h"

    # 3. KDD override: defines dataset_name='kdd' at root
    kdd_cfg = load_config(base_yaml, Path("configs/dataset_kdd.yaml"))
    assert kdd_cfg.dataset_name == "kdd"
    assert kdd_cfg.data.dataset_name == "kdd"
    assert kdd_cfg.data.get("entity_id_modulo") == 254


def test_config_delegation_and_missing_data_block():
    """Verify Config transparently delegates to data block and handles missing data block gracefully."""
    raw = {
        "seed": 42,
        "data": {
            "dataset_name": "custom_data",
            "nested_param": 100
        }
    }
    cfg = Config(raw)
    # Direct access to root
    assert cfg.seed == 42
    # Transparent delegation to data block
    assert cfg.dataset_name == "custom_data"
    assert cfg.nested_param == 100
    assert cfg.get("dataset_name") == "custom_data"
    assert cfg.get("nested_param") == 100
    assert cfg.get("nonexistent", "default_val") == "default_val"

    # Config lacking data block
    no_data_cfg = Config({"seed": 99})
    assert no_data_cfg.seed == 99
    assert isinstance(no_data_cfg.data, Config)
    assert no_data_cfg.data.get("anything") is None


def test_resolve_experiment_config():
    """Verify resolve_experiment_config resolves existing files or falls back to standard configs."""
    base_yaml = Path("configs/base.yaml")
    # Case 1: candidate exists
    cfg1 = resolve_experiment_config(base_yaml)
    assert cfg1.seed == 42

    # Case 2: candidate missing, fallback by dataset hint
    cfg_unsw = resolve_experiment_config("nonexistent/run_config.yaml", dataset_hint="UNSW")
    assert cfg_unsw.dataset_name == "unsw_nb15"

    cfg_kdd = resolve_experiment_config("nonexistent/run_config.yaml", dataset_hint="KDD")
    assert cfg_kdd.dataset_name == "kdd"

    cfg_ton = resolve_experiment_config("nonexistent/run_config.yaml", dataset_hint="ton_iot")
    assert cfg_ton.dataset_name == "ton_iot"
    assert cfg_ton.entity_id_column == "src_ip"


def test_entity_id_resolution_in_loaders():
    """Verify that loaders resolve real endpoint IPs for TON-IoT instead of falling back to synthetic 10.0.0.X."""
    ton_csv = Path("datasets/ton-iot.csv")
    if not ton_csv.exists():
        pytest.skip("datasets/ton-iot.csv not available")

    cfg = load_config(Path("configs/base.yaml"), Path("configs/dataset_ton_iot.yaml"))
    flows = load_dataset_flows(cfg, Path("datasets"), "ton-iot.csv", limit=10)
    assert len(flows) == 10

    # Ensure real endpoint IP addresses were extracted
    entity_ids = [f.entity_id for f in flows]
    assert any("." in eid and not eid.startswith("10.0.0.") for eid in entity_ids), (
        f"Expected real IP addresses in entity_ids, got {entity_ids}"
    )


def test_load_raw_unsw_nb15_compatibility():
    """Verify load_raw_unsw_nb15 compatibility wrapper function works as expected."""
    cfg = load_config(Path("configs/base.yaml"), Path("configs/dataset_unsw_nb15.yaml"))
    unsw_csv = Path("datasets/UNSW_NB15_testing-set.csv")
    if not unsw_csv.exists():
        pytest.skip("UNSW test set not available")

    flows = load_raw_unsw_nb15(cfg, data_dir=Path("datasets"), limit=5)
    assert len(flows) == 5

