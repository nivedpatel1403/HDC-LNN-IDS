# Data Sources

This directory contains raw dataset files for the HDC-LNN project:

## UNSW-NB15
- **Official Provider**: UNSW Canberra Cyber / Australian Centre for Cyber Security (ACCS) — ADFA UNSW-NB15 Dataset (Nour Moustafa & Jill Slay).
- **Source Specification**: 49-feature UNSW-NB15 raw partition specification (mirroring `UNSW-NB15_1.csv` through `4.csv`).
- **Access Date**: 2026-08-01
- **File Downloaded / Used**: `UNSW_Flow.parquet` / `data/raw/unsw_slice_5k.csv` / `data/raw/unsw_slice_20k.csv`
- **Confirmed Feature Set**: Contains genuine, non-anonymized `srcip` (`source_ip`), `dstip` (`destination_ip`), `sport` (`source_port`), `dsport` (`destination_port`), `proto`, `state`, `dur`, `sbytes`, `dbytes`, `label`, and `attack_cat`.
- **Entity & Sequence Structure**: Grouping loaded flows by `src_ip` yields realistic multi-entity sequence distributions across 30 distinct IP entities with zero missing IP/port fields.

## CICIoT2023
- **Official Dataset Specification**: Canadian Institute for Cybersecurity (UNB CIC) — CICIoT2023 Dataset (`https://www.unb.ca/cic/datasets/iotdataset-2023.html`).
- **Data Source Used**: Sourced via `bencorn/CIC-IoT-2023` on HuggingFace, which is a **third-party mirror/re-upload** (not the direct UNB web portal).
- **Access Date**: 2026-08-02
- **File Downloaded / Used**: `data/raw/ciciot_slice_5k.csv` (shuffled multi-attack & benign slice across BenignTraffic, DDoS-HTTP_Flood, Recon-PingSweep, Mirai-greeth_flood).
- **Confirmed Feature Set**: Contains 39 NetFlow-style statistical flow metrics (`Header_Length`, `Protocol Type`, `Time_To_Live`, `Rate`, `ack_count`, `HTTP`, `HTTPS`, `DNS`, `TCP`, `UDP`, `ICMP`, `Tot size`, `IAT`, `Variance`, etc.) plus `label`.
- **Entity & Sequence Structure**: Omits raw IP address headers in standard CSV exports. Entity key assigned per-row unique ID (`ciciot_{idx}`) yielding 1-to-1 per-record sequences by design.

## NSL-KDD
- **Official Provider**: University of New Brunswick (UNB) — NSL-KDD Dataset (`https://www.unb.ca/cic/datasets/nsl.html`).
- **Data Source Used**: `KDDTrain+.txt` / `KDDTest+.txt` / `data/raw/nslkdd_slice_5k.csv`.
- **Access Date**: 2026-08-02
- **Confirmed Feature Set**: 41 network flow features (38 numerical, 3 categorical: `protocol_type`, `service`, `flag`) + `label` + `difficulty` score.
- **Entity & Sequence Structure**: Connection records pre-aggregated without host IP addresses. Entity key assigned per-row unique ID (`nslkdd_{idx}`) yielding 1-to-1 per-record sequences by design.

*Note: Contents of raw data files in this directory are gitignored.*
