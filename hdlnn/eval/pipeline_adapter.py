"""StandardPipeline adapter orchestrating HDC encoding, sequence modeling, and divergence scoring."""

import time
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from hdlnn.baselines import CNNBaseline, LSTMBaseline
from hdlnn.common.config import Settings
from hdlnn.common.logging import get_logger
from hdlnn.contracts.decision import AnomalyDecision
from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.contracts.trajectory import TrajectoryState
from hdlnn.divergence import DivergenceScorer
from hdlnn.divergence.interfaces import IDivergenceScorer
from hdlnn.eval.ablation_adapters import HDCOffEncoder, PassthroughSequenceModel
from hdlnn.eval.interfaces import IPipeline
from hdlnn.eval.metrics import (
    compute_detection_metrics,
    compute_latency_stats,
    compute_memory_stats,
)
from hdlnn.eval.run_result import RunResult
from hdlnn.hdc import HDCEncoder
from hdlnn.hdc.interfaces import IEncoder
from hdlnn.lnn import CfCModel, LTCModel
from hdlnn.lnn.interfaces import ISequenceModel
from hdlnn.lnn.utils import load_encoded_vector


class StandardPipeline(IPipeline):
    """End-to-end pipeline adapter orchestrating HDC encoding, sequence model training, and divergence scoring.

    Sequence Model Training Objective:
    Sequence models (CfC, LTC, LSTM, CNN) are trained on normal-only training sequences using a
    Self-Supervised Next-Step Prediction Loss (MSE between linear projection readout(h_t) and x_{t+1}).
    """

    def __init__(
        self,
        settings: Settings,
        encoder: IEncoder | None = None,
        sequence_model: ISequenceModel | None = None,
        divergence_scorer: IDivergenceScorer | None = None,
        epochs: int | None = None,
        lr: float | None = None,
        device: str | None = None,
    ) -> None:
        self.settings = settings
        self.epochs = epochs if epochs is not None else settings.epochs
        self.lr = lr if lr is not None else settings.lr
        self.device = device if device is not None else settings.device
        self.logger = get_logger("hdlnn.eval.pipeline_adapter", run_id=settings.run_id)

        # 1. Resolve Encoder (HDC-on vs HDC-off ablation)
        if encoder is not None:
            self.encoder = encoder
        elif settings.hdc_encoder == "off":
            self.encoder = HDCOffEncoder(dim=settings.dimensionality)
        else:
            self.encoder = HDCEncoder(
                dim=settings.dimensionality,
                representation="MAP",
                num_levels=settings.num_levels,
            )

        # 2. Resolve Sequence Model (LNN-on vs LNN-off ablation, model type selection)
        if sequence_model is not None:
            self.sequence_model = sequence_model
        elif settings.lnn_stage == "off":
            self.sequence_model = PassthroughSequenceModel()
        else:
            m_type = settings.sequence_model_type.lower()
            if m_type == "ltc":
                self.sequence_model = LTCModel(input_size=settings.dimensionality, hidden_size=settings.hidden_size)
            elif m_type == "lstm":
                self.sequence_model = LSTMBaseline(input_size=settings.dimensionality, hidden_size=settings.hidden_size)
            elif m_type == "cnn":
                self.sequence_model = CNNBaseline(input_size=settings.dimensionality, hidden_size=settings.hidden_size, window_size=settings.window_size)
            else:
                self.sequence_model = CfCModel(input_size=settings.dimensionality, hidden_size=settings.hidden_size)

        # 3. Resolve Divergence Scorer
        if divergence_scorer is not None:
            self.divergence_scorer = divergence_scorer
        else:
            self.divergence_scorer = DivergenceScorer(target_fpr=settings.target_fpr)

    def _train_sequence_model(
        self, train_encoded_groups: dict[str, list[EncodedHypervector]]
    ) -> None:
        """Train sequence model using self-supervised next-step prediction MSE loss on normal sequences."""
        model_module = getattr(
            self.sequence_model,
            "cfc",
            getattr(
                self.sequence_model,
                "ltc",
                getattr(
                    self.sequence_model,
                    "lstm",
                    getattr(self.sequence_model, "conv", None),
                ),
            ),
        )

        if model_module is None or not isinstance(model_module, nn.Module):
            return

        input_size = getattr(self.sequence_model, "input_size", self.settings.dimensionality)
        hidden_size = getattr(self.sequence_model, "hidden_size", self.settings.hidden_size)

        readout = nn.Linear(hidden_size, input_size)
        params = list(model_module.parameters()) + list(readout.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # General sequence length capping (P8 compliance): split long entity sequences into sub-sequences <= max_sequence_length
        max_seq_len = max(2, self.settings.max_sequence_length)
        chunked_groups: dict[str, list[Any]] = {}
        for entity_id, seq in train_encoded_groups.items():
            if len(seq) <= max_seq_len:
                chunked_groups[entity_id] = seq
            else:
                for chunk_idx, i in enumerate(range(0, len(seq), max_seq_len)):
                    sub_seq = seq[i : i + max_seq_len]
                    if len(sub_seq) >= 2:
                        chunked_groups[f"{entity_id}_chunk_{chunk_idx}"] = sub_seq

        self.logger.info(
            "Training sequence model for %d epochs with lr=%f over %d chunked entity streams (max_len=%d)...",
            self.epochs,
            self.lr,
            len(chunked_groups),
            max_seq_len,
        )

        # Pre-load encoded tensors in memory once to eliminate synchronous disk I/O in the inner epoch loop
        self.logger.info("Pre-loading encoded hypervector tensors into RAM...")
        encoded_tensors_cache: dict[str, torch.Tensor] = {}
        for entity_id, seq in chunked_groups.items():
            if len(seq) >= 2:
                step_tensors = [load_encoded_vector(hv).squeeze() for hv in seq]
                encoded_tensors_cache[entity_id] = torch.stack(step_tensors, dim=0)  # Shape (N, input_size)

        model_module.train()
        start_time = time.time()
        for epoch in range(self.epochs):
            t_epoch_start = time.time()
            total_loss = 0.0
            num_batches = 0

            for entity_id, seq in chunked_groups.items():
                if len(seq) < 2 or entity_id not in encoded_tensors_cache:
                    continue

                step_tensor = encoded_tensors_cache[entity_id]
                x_seq = step_tensor.unsqueeze(0)  # Shape (1, N, input_size)

                optimizer.zero_grad()

                if isinstance(self.sequence_model, CNNBaseline):
                    x_conv = x_seq.transpose(1, 2)
                    out_conv = self.sequence_model.relu(self.sequence_model.conv(x_conv))
                    out_seq = out_conv.transpose(1, 2)
                elif hasattr(model_module, "forward"):
                    out_seq, _ = model_module(x_seq)
                else:
                    continue

                preds = readout(out_seq[:, :-1, :])
                targets = x_seq[:, 1:, :]

                loss = nn.functional.mse_loss(preds, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            t_elapsed = time.time() - start_time
            self.logger.info(
                "[run_id=%s] Epoch %d/%d - Loss: %.6f - Elapsed: %.2fs",
                self.settings.run_id,
                epoch + 1,
                self.epochs,
                avg_loss,
                t_elapsed,
            )

        model_module.eval()

    def run(
        self,
        train_flows: list[CanonicalFlow],
        val_flows: list[CanonicalFlow],
        test_flows: list[CanonicalFlow],
    ) -> RunResult:
        """Run full pipeline execution and return RunResult."""
        self.logger.info(
            "Starting pipeline execution: train=%d, val=%d, test=%d flows (hdc=%s, lnn=%s, model=%s)",
            len(train_flows),
            len(val_flows),
            len(test_flows),
            self.settings.hdc_encoder,
            self.settings.lnn_stage,
            self.settings.sequence_model_type,
        )

        # 1. Fit Encoder
        self.encoder.fit(train_flows)

        # 2. Encode flows
        train_encoded = [self.encoder.encode(f) for f in train_flows]
        val_encoded = [self.encoder.encode(f) for f in val_flows]
        test_encoded = [self.encoder.encode(f) for f in test_flows]

        # 3. Group train encoded hypervectors by entity_id
        train_groups: dict[str, list[EncodedHypervector]] = {}
        for hv, f in zip(train_encoded, train_flows):
            train_groups.setdefault(f.src_ip, []).append(hv)

        # 4. Train sequence model
        self._train_sequence_model(train_groups)

        # 5. Process validation flows & fit reference manifold on normal-only validation trajectories
        max_seq_len = max(2, self.settings.max_sequence_length)
        val_groups: dict[str, list[EncodedHypervector]] = {}
        val_flow_map: dict[str, CanonicalFlow] = {}
        for hv, f in zip(val_encoded, val_flows):
            val_groups.setdefault(f.src_ip, []).append(hv)
            val_flow_map[hv.flow_id] = f

        chunked_val_groups: dict[str, list[EncodedHypervector]] = {}
        for entity_id, seq in val_groups.items():
            if len(seq) <= max_seq_len:
                chunked_val_groups[entity_id] = seq
            else:
                for chunk_idx, i in enumerate(range(0, len(seq), max_seq_len)):
                    sub_seq = seq[i : i + max_seq_len]
                    if sub_seq:
                        chunked_val_groups[f"{entity_id}_chunk_{chunk_idx}"] = sub_seq

        normal_val_trajs: list[TrajectoryState] = []
        normal_val_labels: list[str] = []

        for entity_id, seq in chunked_val_groups.items():
            trajs = self.sequence_model.forward(seq, entity_id=entity_id)
            for traj in trajs:
                orig_flow = val_flow_map[traj.flow_id]
                if orig_flow.label == "normal":
                    normal_val_trajs.append(traj)
                    normal_val_labels.append("normal")

        # Fit divergence scorer on normal validation trajectories
        if normal_val_trajs:
            self.divergence_scorer.fit(normal_val_trajs, normal_val_labels)
        else:
            train_trajs: list[TrajectoryState] = []
            train_labels: list[str] = []
            for entity_id, seq in train_groups.items():
                trajs = self.sequence_model.forward(seq, entity_id=entity_id)
                train_trajs.extend(trajs)
                train_labels.extend(["normal"] * len(trajs))
            self.divergence_scorer.fit(train_trajs, train_labels)

        # 6. Process test flows & score test trajectories while timing latency
        test_groups: dict[str, list[EncodedHypervector]] = {}
        test_flow_map: dict[str, CanonicalFlow] = {}
        for hv, f in zip(test_encoded, test_flows):
            test_groups.setdefault(f.src_ip, []).append(hv)
            test_flow_map[hv.flow_id] = f

        decisions: list[AnomalyDecision] = []
        latencies_ms: list[float] = []

        for entity_id, seq in test_groups.items():
            prev_state: TrajectoryState | None = None
            for step_idx, hv in enumerate(seq):
                if step_idx > 0 and step_idx % max_seq_len == 0:
                    prev_state = None  # Symmetric state reset at max_sequence_length boundaries to match training structure

                t0 = time.perf_counter()
                curr_state = self.sequence_model.step(hv, prev_state=prev_state, entity_id=entity_id)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

                orig_flow = test_flow_map[hv.flow_id]
                decision = self.divergence_scorer.score(
                    curr_state,
                    ground_truth_label=orig_flow.label,
                    attack_category=orig_flow.attack_category,
                )
                decisions.append(decision)
                prev_state = curr_state

        # 7. Compute metrics
        detection_metrics = compute_detection_metrics(decisions)
        latency_stats = compute_latency_stats(latencies_ms, device=self.device)
        memory_stats = compute_memory_stats(self.sequence_model, device=self.device)

        # Store test OOV fields info in detection metrics
        oov_hits = sum(len(hv.oov_fields) for hv in test_encoded)
        total_cat_fields = sum(len(f.categorical_features) for f in test_flows)
        detection_metrics["oov_rate"] = float(oov_hits / max(1, total_cat_fields))

        config_snapshot = self.settings.model_dump()

        return RunResult(
            run_id=self.settings.run_id,
            config_snapshot=config_snapshot,
            detection_metrics=detection_metrics,
            latency_stats=latency_stats,
            memory_stats=memory_stats,
            decisions=decisions,
            git_commit_hash="unknown",
        )
