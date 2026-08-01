"""Pipeline Orchestrator driving train, eval, stream, and ablate execution modes."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
import pandas as pd
import torch
import yaml

from hdlnn.common.config import Settings, load_settings
from hdlnn.common.logging import get_logger
from hdlnn.common.seed import set_global_seed
from hdlnn.contracts.decision import AnomalyDecision
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.data.loaders.ciciot import CICIoTLoader
from hdlnn.data.loaders.nslkdd import NSLKDDLoader
from hdlnn.data.loaders.unsw import UNSWLoader
from hdlnn.data.quality import check_quality, log_quality_report
from hdlnn.data.splitter import Splitter
from hdlnn.eval import StandardPipeline, run_ablation_suite, run_experiment
from hdlnn.eval.harness import ComparisonTable, get_git_commit_hash
from hdlnn.eval.metrics import (
    compute_detection_metrics,
    compute_latency_stats,
    compute_memory_stats,
)
from hdlnn.eval.run_result import RunResult


def load_and_split_data(
    config: Settings,
) -> tuple[list[CanonicalFlow], list[CanonicalFlow], list[CanonicalFlow]]:
    """Load raw dataset flows, execute quality checks, and apply or persist dataset splits.

    Note: Reuses existing split JSON file if present; never regenerates splits silently.

    Args:
        config: Application Settings configuration object.

    Returns:
        Tuple of (train_flows, val_flows, test_flows).
    """
    logger = get_logger("hdlnn.pipeline.load_and_split_data", run_id=config.run_id)

    ds_source = getattr(config, "dataset_source", "unsw").lower()
    if ds_source == "ciciot":
        loader = CICIoTLoader(run_id=config.run_id)
    elif ds_source == "nslkdd":
        loader = NSLKDDLoader(run_id=config.run_id)
    else:
        loader = UNSWLoader(run_id=config.run_id)

    raw_path = getattr(config, "dataset_raw_path", None)
    if not raw_path or not Path(raw_path).exists():
        repo_fixture = Path(__file__).parent.parent / "tests" / "fixtures" / "unsw_sample.csv"
        if repo_fixture.exists():
            raw_path = str(repo_fixture)
        else:
            raw_path = "tests/fixtures/unsw_sample.csv"
        logger.info("Using default dataset path '%s'", raw_path)

    flows = loader.load(raw_path)

    # Execute data quality checks
    q_report = check_quality(flows)
    log_quality_report(logger, q_report)

    # Manage dataset splits
    split_version = getattr(config, "split_version", "v1")
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_file = splits_dir / f"{ds_source}_{split_version}.json"

    splitter = Splitter()
    if split_file.exists():
        logger.info("Loading existing dataset split from '%s'...", split_file)
        split_map = splitter.load_split(dataset_name=ds_source, version=split_version, splits_dir=str(splits_dir))
        train_flows, val_flows, test_flows = [], [], []
        for f in flows:
            s = split_map.get(f.flow_id, "train")
            f.split = s
            if s == "val":
                val_flows.append(f)
            elif s == "test":
                test_flows.append(f)
            else:
                train_flows.append(f)
    else:
        logger.info("Creating new dataset split (seed=%d) and saving split version '%s'...", config.seed, split_version)
        train_flows, val_flows, test_flows = splitter.split(
            flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=config.seed
        )
        splitter.save_split(flows, dataset_name=ds_source, version=split_version, out_dir=str(splits_dir))

    return train_flows, val_flows, test_flows


def run_train(config: Settings) -> RunResult:
    """Execute 'train' mode: fit item memory, train sequence model, and compute metrics."""
    logger = get_logger("hdlnn.pipeline.run_train", run_id=config.run_id)
    logger.info("Executing Mode: TRAIN (run_id='%s')", config.run_id)

    train_flows, val_flows, test_flows = load_and_split_data(config)
    pipeline = StandardPipeline(settings=config)
    return run_experiment(config, pipeline, train_flows, val_flows, test_flows)


def run_eval(config: Settings, run_id_to_load: str | None = None) -> RunResult:
    """Execute 'eval' mode: load trained checkpoint if specified and compute evaluation metrics."""
    logger = get_logger("hdlnn.pipeline.run_eval", run_id=config.run_id)
    logger.info("Executing Mode: EVAL (run_id='%s')", config.run_id)

    train_flows, val_flows, test_flows = load_and_split_data(config)
    pipeline = StandardPipeline(settings=config)

    if run_id_to_load:
        ckpt_path = Path("runs") / run_id_to_load / "model_checkpoint.pt"
        if ckpt_path.exists():
            logger.info("Loading sequence model weights from checkpoint '%s'...", ckpt_path)
            model_module = getattr(
                pipeline.sequence_model,
                "cfc",
                getattr(
                    pipeline.sequence_model,
                    "ltc",
                    getattr(
                        pipeline.sequence_model,
                        "lstm",
                        getattr(pipeline.sequence_model, "conv", None),
                    ),
                ),
            )
            if isinstance(model_module, torch.nn.Module):
                state_dict = torch.load(ckpt_path, weights_only=True)
                model_module.load_state_dict(state_dict)
                model_module.eval()
        else:
            logger.warning("Checkpoint at '%s' not found; continuing with initial weights", ckpt_path)

    return run_experiment(config, pipeline, train_flows, val_flows, test_flows)


def run_stream(
    config: Settings,
    run_id_to_load: str,
    replay_rate: float = 1.0,
    check_consistency: bool = False,
) -> RunResult:
    """Execute 'stream' mode: replay test split in timestamp order using step() streaming inference.

    Replay Rate Semantics:
        replay_rate controls real-time stream pacing simulation (sleep dt / replay_rate).
        Capped at max 0.05s per record to prevent excessive delay during test runs.

    Consistency Check:
        When check_consistency=True, compares streaming step() decisions against batch forward() decisions.
        Raises ValueError if any divergence in anomaly decisions is detected.
    """
    logger = get_logger("hdlnn.pipeline.run_stream", run_id=config.run_id)
    logger.info("Executing Mode: STREAM (run_id='%s', loading='%s')", config.run_id, run_id_to_load)

    set_global_seed(config.seed)
    train_flows, val_flows, test_flows = load_and_split_data(config)
    pipeline = StandardPipeline(settings=config)

    # Load model checkpoint
    ckpt_path = Path("runs") / run_id_to_load / "model_checkpoint.pt"
    if ckpt_path.exists():
        logger.info("Loaded model checkpoint from '%s'", ckpt_path)
        model_module = getattr(
            pipeline.sequence_model,
            "cfc",
            getattr(
                pipeline.sequence_model,
                "ltc",
                getattr(
                    pipeline.sequence_model,
                    "lstm",
                    getattr(pipeline.sequence_model, "conv", None),
                ),
            ),
        )
        if isinstance(model_module, torch.nn.Module):
            state_dict = torch.load(ckpt_path, weights_only=True)
            model_module.load_state_dict(state_dict)
            model_module.eval()

    # Fit encoder and divergence manifold on train/val
    pipeline.encoder.fit(train_flows)
    val_encoded = [pipeline.encoder.encode(f) for f in val_flows]
    val_groups: dict[str, list[Any]] = {}
    val_map: dict[str, CanonicalFlow] = {}
    for hv, f in zip(val_encoded, val_flows):
        val_groups.setdefault(f.src_ip, []).append(hv)
        val_map[hv.flow_id] = f

    normal_val_trajs = []
    normal_val_labels = []
    for entity_id, seq in val_groups.items():
        trajs = pipeline.sequence_model.forward(seq, entity_id=entity_id)
        for traj in trajs:
            if val_map[traj.flow_id].label == "normal":
                normal_val_trajs.append(traj)
                normal_val_labels.append("normal")

    if normal_val_trajs:
        pipeline.divergence_scorer.fit(normal_val_trajs, normal_val_labels)

    # Order test flows by occurred_at timestamp
    test_flows_sorted = sorted(test_flows, key=lambda f: f.occurred_at)
    test_encoded = [pipeline.encoder.encode(f) for f in test_flows_sorted]

    # Perform streaming step() pass
    stream_decisions: list[AnomalyDecision] = []
    latencies_ms: list[float] = []

    prev_time: datetime | None = None
    prev_state_map: dict[str, Any] = {}

    for hv, flow in zip(test_encoded, test_flows_sorted):
        # Simulate replay rate delay if enabled
        if replay_rate > 0 and prev_time is not None:
            dt = (flow.occurred_at - prev_time).total_seconds()
            sleep_duration = min(dt / max(0.001, replay_rate), 0.05)
            if sleep_duration > 0:
                time.sleep(sleep_duration)
        prev_time = flow.occurred_at

        entity_id = flow.src_ip
        prev_state = prev_state_map.get(entity_id)

        t0 = time.perf_counter()
        curr_state = pipeline.sequence_model.step(hv, prev_state=prev_state, entity_id=entity_id)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        decision = pipeline.divergence_scorer.score(
            curr_state, ground_truth_label=flow.label, attack_category=flow.attack_category
        )
        stream_decisions.append(decision)
        prev_state_map[entity_id] = curr_state

    # Consistency verification: batch forward vs streaming step
    if check_consistency:
        logger.info("Running eval-vs-stream consistency verification check...")
        test_groups: dict[str, list[tuple[Any, CanonicalFlow, int]]] = {}
        for idx, (hv, f) in enumerate(zip(test_encoded, test_flows_sorted)):
            test_groups.setdefault(f.src_ip, []).append((hv, f, idx))

        batch_dec_map: dict[tuple[str, int], AnomalyDecision] = {}
        for entity_id, items in test_groups.items():
            seq = [item[0] for item in items]
            b_trajs = pipeline.sequence_model.forward(seq, entity_id=entity_id)
            for b_traj, item in zip(b_trajs, items):
                orig_flow = item[1]
                idx = item[2]
                b_dec = pipeline.divergence_scorer.score(
                    b_traj, ground_truth_label=orig_flow.label, attack_category=orig_flow.attack_category
                )
                batch_dec_map[(b_traj.flow_id, idx)] = b_dec

        for idx, (s_dec, orig_flow) in enumerate(zip(stream_decisions, test_flows_sorted)):
            b_dec = batch_dec_map[(s_dec.flow_id, idx)]
            if s_dec.is_anomalous != b_dec.is_anomalous:
                raise ValueError(
                    f"Consistency Check Failed! AnomalyDecision divergence at flow '{s_dec.flow_id}': "
                    f"stream (anomalous={s_dec.is_anomalous}) vs "
                    f"batch (anomalous={b_dec.is_anomalous})"
                )
        logger.info("Eval-vs-stream consistency verification PASSED: 100%% decision agreement!")

    # Assemble RunResult without retraining sequence model
    detection_metrics = compute_detection_metrics(stream_decisions)
    latency_stats = compute_latency_stats(latencies_ms, device=config.device)
    memory_stats = compute_memory_stats(pipeline.sequence_model, device=config.device)
    commit_hash = get_git_commit_hash()

    result = RunResult(
        run_id=config.run_id,
        config_snapshot=config.model_dump(),
        detection_metrics=detection_metrics,
        latency_stats=latency_stats,
        memory_stats=memory_stats,
        decisions=stream_decisions,
        git_commit_hash=commit_hash,
    )

    # Persist run artifacts to runs/<run_id>/
    run_dir = Path("runs") / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)

    with open(run_dir / "git_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{commit_hash}\n")

    metrics_payload = {
        "run_id": config.run_id,
        "detection_metrics": detection_metrics,
        "latency_stats": latency_stats,
        "memory_stats": memory_stats,
        "git_commit_hash": commit_hash,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    dec_dicts = [d.model_dump() for d in stream_decisions]
    df = pd.DataFrame(dec_dicts)
    if "t" in df.columns:
        df["t"] = df["t"].astype(str)
    df.to_parquet(run_dir / "decisions.parquet", index=False)

    return result


def run_ablate(
    configs: list[Settings] | None = None, configs_dir: str = "configs"
) -> ComparisonTable:
    """Execute 'ablate' mode: run ablation suite across all specified or discovered config YAMLs."""
    logger = get_logger("hdlnn.pipeline.run_ablate")
    logger.info("Executing Mode: ABLATE")

    if configs is None:
        yaml_files = sorted(Path(configs_dir).glob("*.yaml"))
        if not yaml_files:
            repo_configs = Path(__file__).parent.parent / "configs"
            yaml_files = sorted(repo_configs.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(f"No YAML configuration files found in '{configs_dir}'")
        logger.info("Discovered %d configuration files", len(yaml_files))
        configs = [load_settings(str(p)) for p in yaml_files]

    train_flows, val_flows, test_flows = load_and_split_data(configs[0])
    return run_ablation_suite(configs, train_flows, val_flows, test_flows)
