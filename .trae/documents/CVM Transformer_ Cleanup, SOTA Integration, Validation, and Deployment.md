## Repository Audit Findings
- Structure: Core code in `cvm_translator/` with many ad‑hoc scripts at repo root; generated artifacts present.
- Misalignment: `compute_loss` expects `(logits, hidden, attn)` but `CVMTransformer.forward` returns only `logits` (`cvm_translator/kd_losses.py:23-30` vs `cvm_translator/cvm_transformer.py:76-82`).
- Training scripts: Multiple overlapping 10k-run variants; `training_10k_simple.py` implements KD via MSE on logits only and writes JSON/PNG outputs to repo root (`training_10k_simple.py:239-287`).
- Validation tooling: Rich protocol available in `cvm_translator/validation_protocol.py` for periodic validation, rollback, and reporting.
- Telemetry/logging: Basic dashboard exists (`cvm_translator/telemetry.py`), but structured logging and rotating files are not unified.
- .gitignore: Generic Python template plus `*.pt`; project-specific outputs not excluded (`.gitignore:1-208`).

## Refactor & Cleanup Plan
- Module layout: Reorganize into clear packages while preserving names where possible.
  - `cvm_translator/models/`: `cvm_transformer.py` (return logits, hidden states, and attention), attention kernels, quantization helpers.
  - `cvm_translator/losses/`: move `kd_losses.py` and align signatures with models.
  - `cvm_translator/data/`: datasets, tokenizers (`sp_tokenizer.py`, corpus prep).
  - `cvm_translator/training/`: training loop, schedulers, checkpoints; unify 10k runner.
  - `cvm_translator/eval/`: validation protocol and metrics integration.
  - `cvm_translator/serving/`: gRPC server/client, proto stubs.
  - `cvm_translator/utils/`: telemetry, language detection, buffering, misc utilities.
  - `configs/`: YAML configs for training/eval/deploy.
  - `scripts/`: thin CLIs (`train.py`, `validate.py`, `serve.py`, `benchmark.py`).
- API alignment:
  - Update `CVMTransformer` forward to return `(logits, hidden_states, attn_maps)` so `kd_losses` works out‑of‑the‑box (`cvm_translator/kd_losses.py:23-30`). Preserve a flag for `return_hidden=True`.
  - Centralize tokenizer interfaces (SentencePiece where available; fallback simple tokenizer used in `training_10k_simple.py:144-166`).
- Remove/rehome artifacts:
  - Move existing JSON/PNG reports to `reports/` and `artifacts/` then add ignore rules; keep a curated `docs/` index of final reports already present.
  - Consolidate duplicate training scripts into `scripts/train.py` with config‑driven behavior.
- Documentation:
  - Add module docstrings and concise function docstrings; auto‑generate API reference later.

## .gitignore Updates (project‑specific)
- Checkpoints & models: `checkpoints/`, `models/`, `*.pt`, `*.bin`, `*.onnx`.
- Experiment outputs: `artifacts/`, `reports/*.json`, `reports/*.png`, `benchmarks/`, `distillation_output/`, `test_distillation_output/`.
- Logs & telemetry: `logs/`, `telemetry.json`, `*.log`.
- Cache/runtime: `*.tmp`, `*.cache`, `__marimo__/`, `*.pb2.py` generated from `.proto` if re‑generated.
- Config overrides: `configs/local/*.yaml`.

## SOTA Research Summary & Integration Plan
- Efficient sequence models:
  - FlashAttention (Tri Dao et al.) for exact IO‑aware attention; integrate when available and gate by device capabilities (Ref: arXiv:2205.14135; deep2Read survey).
  - Mamba (Selective SSM) as long‑context alternative; consider hybrid layer for streaming segments where quadratic attention is a bottleneck (Ref: arXiv:2312.00752; AI21 Jamba blog).
  - Emerging alternatives: Hyena, RWKV, RetNet; evaluate for streaming robustness (gocodeo 2025 overview).
- Optimization:
  - Baseline AdamW; evaluate Lion and Adafactor‑with‑momentum on transformer tasks (ICLR 2025 optimizer comparisons; Google/UCLA Lion article). Keep SAM as optional regularizer for generalization.
  - Schedulers: cosine decay with warmup; one‑cycle for rapid 10k runs.
- Distillation & compression:
  - Multi‑objective KD already scaffolded; align hidden/attention supervision after model API update.
  - Quantization: INT8/INT4 via `onnxruntime`, `auto-gptq`, `autoawq`; apply per‑layer scale calibration (SmoothQuant‑style) where applicable.
- Streaming & physics‑inspired methods:
  - Use CVM reservoir sampling for token core‑sets (`cvm_translator/cvm_buffer.py`); evaluate Kalman‑style smoothing of decoder logits for stability.
  - State‑space (SSM) blocks echo control/system techniques for long‑range dependencies; pilot a small SSM layer in encoder for latency benefits.

## Experimental Validation Plan (10,000 iterations)
- Unified runner:
  - Implement `scripts/train.py` that reads `configs/train.yaml` (device, model dims, optimizer, data).
  - Iterations: 10,000 with validation every 1,000 via `DistillationValidator` (`cvm_translator/validation_protocol.py`).
- Validation protocol:
  - Metrics: BLEU/ROUGE/TER/BERTScore if installed; fallback metrics otherwise.
  - Rollback & early stopping thresholds from `ValidationConfig` (`validation_protocol.py:31-47`).
- Error checking & logging:
  - Structured logging with rotating file handlers; per‑epoch summaries plus batch‑level anomalies.
  - Telemetry: latency, throughput, loss trend using `TelemetryDashboard` (`cvm_translator/telemetry.py`).
- Output verification:
  - Compare student outputs against references and (optionally) teacher baseline; persist `validation_history.json` and a concise `training_report.json`.
- Hyperparameter search:
  - Systematic random/Bayesian search (Optuna optional) over LR, weight decay, temperature, alpha/beta/gamma KD weights; fixed budget with best config exported to `configs/best.yaml`.

## Full Deployment Plan
- Complete training cycle:
  - Run multi‑epoch training with best hyperparameters; save checkpoints and final weights to `checkpoints/`.
- Monitoring & logging:
  - Unified log directory; telemetry plots auto‑generated.
- Serving:
  - Package gRPC server/client; ensure `realtime_translation.proto` stubs up‑to‑date (`cvm_translator/realtime_translation.proto`).
- Final validation & reporting:
  - Execute comprehensive validation; generate final report using `generate_final_report.py` (already present), extended to include baseline comparisons.
  - Artifacts saved under `reports/` and ignored by VCS.

## Deliverables
- Cleaned repository with logical module structure and updated `.gitignore`.
- Aligned model/loss APIs enabling full KD (logits, hidden, attention).
- Config‑driven training/validation scripts with 10k iteration runner and periodic validation.
- SOTA integration toggles (FlashAttention, SSM/Mamba, quantization, optimizer options).
- Final report with metrics, baseline comparison, and recommendations for future work.

## Notes
- We will preserve existing files’ functionality where possible, consolidating overlapping scripts.
- Changes will be incremental with verification at each step; no secrets or keys will be logged.
- If certain optional libraries are missing, features will be gated and fall back gracefully.
