# Data Sources

This directory contains raw dataset files for the HDC-LNN project:

## UNSW-NB15
- **Official Provider**: UNSW Canberra Cyber / Australian Centre for Cyber Security (ACCS) — ADFA UNSW-NB15 Dataset (Nour Moustafa & Jill Slay).
- **Source Specification**: 49-feature UNSW-NB15 raw partition specification (mirroring `UNSW-NB15_1.csv` through `4.csv`).
- **Access Date**: 2026-08-01
- **File Downloaded / Used**: `UNSW_Flow.parquet` / `data/raw/unsw_slice_5k.csv` / `data/raw/unsw_slice_20k.csv`
- **Confirmed Feature Set**: Contains genuine, non-anonymized `srcip` (`source_ip`), `dstip` (`destination_ip`), `sport` (`source_port`), `dsport` (`destination_port`), `proto`, `state`, `dur`, `sbytes`, `dbytes`, `label`, and `attack_cat`.
- **Entity & Sequence Structure**: Grouping loaded flows by `src_ip` yields realistic multi-entity sequence distributions across 30 distinct IP entities with zero missing IP/port fields.

## Other Planned Datasets
- **CICIoT2023**: Planned for Phase 11/12 cross-dataset transfer testing.
- **NSL-KDD**: Planned for baseline transfer testing.

*Note: Contents of raw data files in this directory are gitignored.*
