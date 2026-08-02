"""Shared metrics calculation module for detection performance, latencies, and memory."""

from typing import Any
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import torch

from hdlnn.contracts.decision import AnomalyDecision


def compute_detection_metrics(decisions: list[AnomalyDecision]) -> dict[str, Any]:
    """Compute detection performance metrics (F1, Precision, Recall, Accuracy, AUROC, FPR@95%TPR).

    Args:
        decisions: List of AnomalyDecision contract objects.

    Returns:
        Dictionary containing calculated performance metrics.
    """
    if not decisions:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "auroc": 0.0,
            "fpr_at_95_tpr": 0.0,
            "total_decisions": 0,
            "anomalies_detected": 0,
        }

    y_true = np.array([1 if d.ground_truth_label == "attack" else 0 for d in decisions])
    y_pred = np.array([1 if d.is_anomalous else 0 for d in decisions])
    scores = np.array([float(d.drift_score) for d in decisions])

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    # AUROC calculation with single-class fallback
    if len(np.unique(y_true)) > 1:
        try:
            auroc = float(roc_auc_score(y_true, scores))
        except Exception:
            auroc = 0.5
    else:
        # Single class present
        auroc = 1.0 if np.array_equal(y_true, y_pred) else 0.5

    # FPR at 95% TPR calculation
    fpr_at_95_tpr = 0.0
    if len(np.unique(y_true)) > 1 and np.sum(y_true == 1) > 0:
        try:
            fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
            # Find indices where TPR >= 0.95
            tpr_mask = tpr_arr >= 0.95
            if np.any(tpr_mask):
                fpr_at_95_tpr = float(fpr_arr[tpr_mask][0])
        except Exception:
            fpr_at_95_tpr = 0.0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "auroc": auroc,
        "fpr_at_95_tpr": fpr_at_95_tpr,
        "total_decisions": len(decisions),
        "anomalies_detected": int(np.sum(y_pred)),
    }


def compute_latency_stats(latencies_ms: list[float], device: str) -> dict[str, Any]:
    """Compute per-record inference latency statistics (mean, p95, min, max).

    Args:
        latencies_ms: List of per-record latency numbers in milliseconds.
        device: Device string (e.g. 'cpu', 'cuda:0'). Required argument.

    Returns:
        Dictionary containing latency statistics and attached device string.

    Raises:
        ValueError: If device argument is missing or empty.
    """
    if not device or not str(device).strip():
        raise ValueError("Device string is required for latency stats calculation.")

    if not latencies_ms:
        return {
            "mean_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "device": str(device).strip(),
        }

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "device": str(device).strip(),
    }


def compute_memory_stats(model: Any, device: str) -> dict[str, Any]:
    """Compute memory usage statistics (peak RSS and PyTorch model parameter size).

    Args:
        model: Model instance (PyTorch nn.Module or wrapper).
        device: Device string. Required argument.

    Returns:
        Dictionary containing peak RSS and model size statistics.

    Raises:
        ValueError: If device argument is missing or empty.
    """
    if not device or not str(device).strip():
        raise ValueError("Device string is required for memory stats calculation.")

    # Peak RSS process memory in MB
    process = psutil.Process()
    peak_rss_mb = float(process.memory_info().rss / (1024 * 1024))

    param_count = 0
    model_size_mb = 0.0

    # Calculate model parameter count and size if PyTorch Module
    target_module = getattr(model, "cfc", getattr(model, "ltc", getattr(model, "lstm", getattr(model, "conv", model))))
    if isinstance(target_module, torch.nn.Module):
        param_count = sum(p.numel() for p in target_module.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in target_module.parameters())
        model_size_mb = float(param_bytes / (1024 * 1024))

    return {
        "peak_rss_mb": peak_rss_mb,
        "model_size_mb": model_size_mb,
        "param_count": param_count,
        "device": str(device).strip(),
    }
