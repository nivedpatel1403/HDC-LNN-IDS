# PHASE 2 REPORT: L2 Data Pipeline Layer

## Files created

- `hdlnn/data/interfaces.py`
- `hdlnn/data/loaders/__init__.py`
- `hdlnn/data/loaders/unsw.py`
- `hdlnn/data/loaders/ciciot.py`
- `hdlnn/data/loaders/nslkdd.py`
- `hdlnn/data/quality.py`
- `hdlnn/data/splitter.py`
- `hdlnn/data/__init__.py`
- `tests/fixtures/unsw_sample.csv`
- `tests/test_data.py`
- `PHASE2_REPORT.md`

## UNSW-NB15 column mapping used

```python
UNSW_COLUMN_MAP: dict[str, str] = {
    "srcip": "src_ip",
    "dstip": "dst_ip",
    "sport": "src_port",
    "dsport": "dst_port",
    "proto": "protocol",
    "dur": "duration",
    "state": "state",
    "label": "label",
    "attack_cat": "attack_category",
}

UNSW_CATEGORICAL_COLS: set[str] = {"service", "proto", "state"}

UNSW_EXCLUDE_NUMERIC: set[str] = {
    "srcip",
    "dstip",
    "sport",
    "dsport",
    "proto",
    "state",
    "dur",
    "service",
    "label",
    "attack_cat",
    "id",
    "flow_id",
    "stime",
    "ltime",
    "sttime",
    "lttime",
    "occurred_at",
}
```

*Timestamp Fallback Note*: UNSW-NB15 raw CSVs in public releases lack explicit timestamps in some distributions. If `Stime`/`Ltime` columns exist, they are parsed as UTC datetimes. When absent, a monotonically increasing timestamp is synthesized starting from `2025-01-01T00:00:00Z` incremented by 1 second per row index order.

## Any deviations from this spec (and why)

1. **Grouped Flow-ID Partitioning in Splitter**: In `Splitter.split()`, flows are grouped by `flow_id` prior to shuffling and split assignment. This ensures duplicate `flow_id` entries are assigned to the exact same split, preventing train-test data leakage.
2. **Comment Parsing in CSV Loader**: Configured `pandas.read_csv` in `UNSWLoader` with `comment='#'` to gracefully ignore comment header lines in synthetic or documented raw CSV data files.

## Test results

```text
.\.venv\Scripts\pytest.exe
..............                                                           [100%]
14 passed in 2.28s
```

## Open questions / things you weren't sure about

- None. All requirements for L2 data ingestion, canonicalization into `CanonicalFlow`, data quality metrics, deterministic splitting, and non-overwriting split persistence are fully met and verified.

## Suggested next step

- Proceed to **Phase 3: Hyperdimensional Computing (HDC) Encoding Layer (L3)** to implement feature-level hypervector encoders, item memories, and `EncodedHypervector` production.
