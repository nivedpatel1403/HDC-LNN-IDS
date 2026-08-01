"""Evaluation and harness package exporting pipeline adapters, ablation suite, and metrics."""

from hdlnn.eval.harness import (
    ComparisonTable,
    run_ablation_suite,
    run_cross_dataset_transfer,
    run_experiment,
)
from hdlnn.eval.interfaces import IPipeline
from hdlnn.eval.metrics import (
    compute_detection_metrics,
    compute_latency_stats,
    compute_memory_stats,
)
from hdlnn.eval.pipeline_adapter import StandardPipeline
from hdlnn.eval.run_result import RunResult

__all__ = [
    "IPipeline",
    "StandardPipeline",
    "RunResult",
    "ComparisonTable",
    "run_experiment",
    "run_ablation_suite",
    "run_cross_dataset_transfer",
    "compute_detection_metrics",
    "compute_latency_stats",
    "compute_memory_stats",
]
