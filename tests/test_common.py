"""Unit tests for L1 common utilities (config, logging, seeding)."""

import os
import random
import numpy as np
import pytest
import torch

from hdlnn.common.config import Settings, load_settings
from hdlnn.common.logging import get_logger
from hdlnn.common.seed import set_global_seed
from hdlnn.contracts.errors import ConfigError


@pytest.mark.unit
def test_settings_defaults():
    settings = Settings()
    assert settings.run_id == "base_run_001"
    assert settings.seed == 42
    assert settings.log_level == "INFO"


@pytest.mark.unit
def test_load_settings_valid_yaml(tmp_path):
    yaml_content = """
    run_id: "test_run_123"
    seed: 99
    log_level: "DEBUG"
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    settings = load_settings(str(config_file))
    assert settings.run_id == "test_run_123"
    assert settings.seed == 99
    assert settings.log_level == "DEBUG"


@pytest.mark.unit
def test_load_settings_nonexistent_file_raises():
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_settings("nonexistent_config_file_xyz.yaml")


@pytest.mark.unit
def test_set_global_seed_reproducibility():
    set_global_seed(42)
    py_rand1 = random.randint(0, 1000000)
    np_rand1 = np.random.randint(0, 1000000)
    torch_rand1 = torch.randint(0, 1000000, (1,)).item()

    set_global_seed(42)
    py_rand2 = random.randint(0, 1000000)
    np_rand2 = np.random.randint(0, 1000000)
    torch_rand2 = torch.randint(0, 1000000, (1,)).item()

    assert py_rand1 == py_rand2
    assert np_rand1 == np_rand2
    assert torch_rand1 == torch_rand2


@pytest.mark.unit
def test_get_logger_formatting():
    logger = get_logger("hdlnn.test", run_id="run_abc")
    assert logger.name == "hdlnn.test"
    assert hasattr(logger, "info")
