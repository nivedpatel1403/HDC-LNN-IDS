# PHASE 7c REPORT: Test Coverage Gap-Fill for Phase 7b

## Tests added

All new test coverage was implemented in `tests/test_ablation_coverage.py`:

1. `test_run_experiment_hybrid_yaml`: Proves `run_experiment()` executes end-to-end with `configs/hybrid.yaml` producing a valid `RunResult`.
2. `test_run_experiment_hdc_only_yaml`: Proves `run_experiment()` executes end-to-end with `configs/hdc_only.yaml` producing a valid `RunResult`.
3. `test_run_experiment_lnn_only_yaml`: Proves `run_experiment()` executes end-to-end with `configs/lnn_only.yaml` producing a valid `RunResult`.
4. `test_hdc_only_bypasses_lnn_and_uses_raw_hdc_vectors`: Proves `hdc_only` mode fits `ReferenceManifold` directly on raw $D=250$ HDC hypervectors and outputs bit-identical tensors without LNN accumulation.
5. `test_ablation_suite_6_configs_plus_cross_dataset_placeholder`: Proves `run_ablation_suite()` generates 6 real config rows with non-null `oov_rate` fields plus 1 cross-dataset transfer placeholder row with `status="not_run: CICIoTLoader not yet implemented (Phase 2 stub)"`.
6. `test_model_checkpoint_persistence_and_readout_exclusion`: Proves `model_checkpoint.pt` is persisted and does not contain linear readout head parameters (`"readout"` keys).
7. `test_hdc_off_mode_bypasses_hdc_encoder`: Proves `hdc_encoder="off"` mode uses `HDCOffEncoder` and never calls `HDCEncoder.encode()` (mock call count == 0).

## Result of the hdc_only "genuinely bypasses LNN" test specifically

- **Status**: **PASSED**
- **Assertions Code**:
  ```python
  # 1. Assert PassthroughSequenceModel is used for lnn_stage='off'
  assert isinstance(pipe.sequence_model, PassthroughSequenceModel)

  res = run_experiment(cfg, pipe, train_flows, val_flows, test_flows)

  # 2. Assert manifold dimensionality equals HDC dimensionality D (250), NOT LNN hidden_size (16)
  manifold_hidden_size = pipe.divergence_scorer.manifold.hidden_size
  assert manifold_hidden_size == dim_value
  assert manifold_hidden_size != hidden_size_value

  # 3. Assert PassthroughSequenceModel output tensors are bit-identical to EncodedHypervector raw arrays
  test_encoded_hv = pipe.encoder.encode(test_flows[0])
  traj = pipe.sequence_model.step(test_encoded_hv, prev_state=None, entity_id="test_ent")

  raw_vector = load_encoded_vector(test_encoded_hv).squeeze()
  passthrough_vector = load_hidden_state(traj.hidden_state_ref).squeeze()

  torch.testing.assert_close(raw_vector, passthrough_vector)
  ```

## Result of the hdc_off "genuinely bypasses HDC encoder" test specifically

- **Status**: **PASSED**
- **Assertions Code**:
  ```python
  # Confirm encoder is HDCOffEncoder and NOT HDCEncoder
  assert isinstance(pipe.encoder, HDCOffEncoder)
  assert not isinstance(pipe.encoder, HDCEncoder)

  with patch.object(HDCEncoder, "encode", autospec=True) as mock_hdc_encode:
      run_experiment(cfg, pipe, train_flows, val_flows, test_flows)
      # Assert HDCEncoder.encode was called 0 times
      assert mock_hdc_encode.call_count == 0
  ```

## Any bugs found and fixed

- **No implementation bugs found**: All Phase 7b implementations (`StandardPipeline`, `PassthroughSequenceModel`, `HDCOffEncoder`, `run_ablation_suite`, checkpoint readout head exclusion) functioned correctly without needing any source modifications.

## Full test suite results

```text
.\.venv\Scripts\pytest.exe
.................................................                        [100%]
49 passed in 19.86s
```

## Suggested next step

- Proceed to **Phase 8: Reporting, Visualization & CLI Layer (L5)** to implement `ComparisonTable` markdown/latex renderer (`hdlnn/report/`), command-line interface (`cli.py`), and end-to-end pipeline execution entrypoint (`pipeline.py`).
