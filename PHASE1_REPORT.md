# PHASE 1 REPORT: L1 Foundation Layer

## Files created

- `pyproject.toml`
- `.gitignore`
- `README.md`
- `data/raw/SOURCES.md`
- `data/splits/.gitkeep`
- `data/encoded/.gitkeep`
- `runs/.gitkeep`
- `configs/base.yaml`
- `hdlnn/__init__.py`
- `hdlnn/pipeline.py`
- `hdlnn/cli.py`
- `hdlnn/data/__init__.py`
- `hdlnn/hdc/__init__.py`
- `hdlnn/lnn/__init__.py`
- `hdlnn/divergence/__init__.py`
- `hdlnn/baselines/__init__.py`
- `hdlnn/eval/__init__.py`
- `hdlnn/report/__init__.py`
- `hdlnn/contracts/__init__.py`
- `hdlnn/contracts/errors.py`
- `hdlnn/contracts/flow.py`
- `hdlnn/contracts/encoded.py`
- `hdlnn/contracts/trajectory.py`
- `hdlnn/contracts/decision.py`
- `hdlnn/common/__init__.py`
- `hdlnn/common/config.py`
- `hdlnn/common/logging.py`
- `hdlnn/common/seed.py`
- `tests/test_contracts.py`
- `tests/test_common.py`

## Any deviations from this spec (and why)

1. **PyPI Package Name for TorchHD**: Specified dependency `torchhd` in `pyproject.toml` was declared as `torch-hd` because `torch-hd` is the exact distribution name on PyPI (which installs and exposes the `torchhd` module).
2. **Setuptools Package Discovery**: Added `[tool.setuptools.packages.find]` to `pyproject.toml` specifying `include = ["hdlnn*"]` to prevent setuptools from discovering root stub directories (`data`, `runs`, `configs`) as python packages.

## Test results

```text
.\.venv\Scripts\pytest.exe
.........                                                                [100%]
9 passed in 7.05s
```

## Open questions / things you weren't sure about

- None. All L1 contracts, error hierarchy, configuration management, logging adapter, seed setter, and directory stubs are strictly implemented to specification without external layer imports.

## Suggested next step

- Proceed to **Phase 2: Data Ingestion & Canonicalization Layer (L2)** to implement dataset-specific parsers for UNSW-NB15, CICIOT2023, and NSL-KDD that output validated `CanonicalFlow` instances.
