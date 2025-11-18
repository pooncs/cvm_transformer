## Objectives
- Achieve BLEU > 0.99 on 95% of validation runs; stability ±0.01 across seeds
- Maintain or improve inference speed; reduce decoding pathologies (repetitions, truncation)
- Ensure reproducible, versioned experiments and clean directory structure

## Phase 1: Training Extension
- Epochs: Increase by 50–100% from current baseline (e.g., 8 → 12–16). Use per‑iteration checkpoints.
- Learning Rate Scheduling:
  - Warmup: First 5–10% of total steps with linear warmup to base LR
  - Cosine decay: Cosine annealing warm restarts (T_0=10, T_mult=2, eta_min=1e-6)
- Early Stopping:
  - Patience 5–10 epochs on validation BLEU; save best checkpoint on improvement
- Logging & Checkpoints:
  - Save training logs to `./training_logs/<run_id>/` (metrics, LR, grad norms, speed)
  - Save checkpoints to `./checkpoints/<run_id>/epoch_{N}.pt` and `best.pt`
  - Include `args.json` and environment snapshot (seed, library versions)

## Phase 2: Model Architecture Enhancement
- Capacity Scaling (per iteration):
  - Increase `d_model`/`nhead`/`num_layers` by 20–50% (e.g., 768→1024, 12→16 heads, 8→12 layers), ensure GPU fit
  - Widen FFN (`dim_feedforward`) proportionally (e.g., 3072→4096/5120)
- Depth & Width Experiments:
  - Deeper enc/dec stacks vs. wider FFN layers; compare trade-offs
- Residual Connections:
  - Confirm residuals and layernorm placement; test Pre‑LN vs Post‑LN variants for stability
- Versioning:
  - Save model variants in `./models/` with versioned filenames: `nmt_v{major}.{minor}_{config_tag}.pt`
  - Maintain a `models/index.json` with config, metrics, creation date

## Phase 3: Data Quality Improvement
- Cleaning:
  - Deduplicate pairs, remove misaligned or empty strings, enforce BOS/EOS/PAD consistency
  - Normalize punctuation, whitespace, unicode; language ID check to ensure KR→EN mapping
- Augmentation:
  - Synonym substitution on English side; controlled back‑translation; noise injection within limits
- Diversity Expansion:
  - Add domain‑balanced corpora (daily, tech, business, education, health, news) with multiple references
- Automated Validation Pipeline:
  - Implement in `./data_validation/`:
    - Scripts: `schema_check.py`, `lang_id_check.py`, `alignment_check.py`, `duplicates_report.py`
  - Write processed datasets to `./processed_data/v{date}-{hash}/` with manifest and stats

## Phase 4: Advanced Techniques Implementation
- Ensembles (3–5 variants):
  - Train diversified configs (depth/width/regularization) and aggregate via weighted averaging or stacking
  - Implement in `./ensemble/` with `blend.py` (weights learned on validation)
- Knowledge Distillation:
  - Teacher: highest‑capacity model; Students: 2–3 smaller models
  - Temperature scaling (T=1–3), KL loss mixing with CE; orchestrate in `./distillation/`
- Decoding Optimization:
  - Beam search (beam=8–12, length_penalty=0.6–1.0), support `greedy`, `beam`, `diverse_beam`
  - Log beam scores/path probabilities for analysis; store to `./experiment_logs/beam_stats/`
- Training Stability Toolkit:
  - Gradient clipping (norm=1.0), AMP+GradScaler, LR warmup, scheduled sampling ramp‑up (0→0.3), dynamic batch sizing

## Phase 5: Validation & Maintenance
- Unit & Integration Tests:
  - Run `./tests/` (tokenizer alignment, BOS/EOS handling, decoding sanity, metrics correctness)
- Holdout Validation:
  - Maintain `./validation_sets/` with multiple references and domain balance; seed‑controlled splits
- Profiling:
  - GPU memory/time profiling per epoch and inference; record to `./experiment_logs/perf/`
- Documentation:
  - Each folder has `README.md` with purpose, usage, and structure
  - Log all experiments in `./experiment_logs/<run_id>/summary.json` + plots
- Cleanup & Version Control:
  - Automated cleanup of temp files after each run; retain essential artifacts
  - Version control with clear, atomic commit messages (config, code, data changes separated)

## Implementation Artifacts (Planned)
- Scripts:
  - `train.py` (extended epochs, warmup+cosine, early stop, logging, checkpoints)
  - `validate.py` (beam/diverse beam, BLEU/ROUGE/METEOR reporting, seed sweeps)
  - `data_validation/*` (automated checks)
  - `ensemble/blend.py` (weighted average/stacking)
  - `distillation/run_distill.py` (teacher→student training)
- Configs:
  - `configs/base.yaml`, `configs/large.yaml`, `configs/ensemble.yaml`, `configs/distill.yaml`

## Metrics & Success Criteria
- Primary: BLEU > 0.99 on ≥95% runs; secondary: ROUGE‑L & METEOR ≥0.99
- Stability: ±0.01 BLEU across seeds {42,1337,2025}
- Speed: Maintain or improve average inference latency baseline; report p95 latency
- Decoding Quality: No repetitive outputs; beam path entropy within expected range

## Timeline & Milestones
- Week 1: Training extension + logging/checkpoints; initial validation
- Week 2: Capacity scaling and residual variants; decode tuning
- Week 3: Data cleaning/augmentation; automated validation pipeline
- Week 4: Ensembles + distillation; comprehensive validation and ablations

## Risks & Mitigations
- GPU memory constraints → gradient accumulation, mixed precision, checkpointing
- Overfitting with extended epochs → stronger regularization, early stopping, multiple references
- Data noise → strict validation pipeline and language alignment checks

## Request for Confirmation
- Confirm directory structure, metric targets, and iteration cadence
- Upon approval, I will implement the scripts/configs, run extended training/validation, and deliver detailed reports