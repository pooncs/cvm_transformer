## Objectives
- Build a compact CVM-enhanced transformer (SLM) for real-time Korean↔English translation.
- Achieve <500 ms end-to-end latency with edge deployability and controlled memory footprint.
- Use CVM (Count–Vector–Merge) for parameter-efficient learning, KV-cache compaction, and selective forgetting.

## Algorithmic Foundations
- CVM estimator: Maintain buffer `B` of size `s` and threshold `p`; output `|B|/p` (unbiased for distinct counts).
- Accuracy: `(ε, δ)` guarantees via `s = O(ε^{-2} log(1/δ))`; per-item cost `O(1)` or `O(log s)` depending on buffer structure.
- Transformer integration: Per-layer CVM buffers over token/feature representations choose core representatives; attention computed over cores; non-core tokens reconstructed via convex weights; merge policy yields controlled forgetting.

## Model Architecture
- Encoder–decoder transformer (bi-directional) with:
- 12–16 layers, `d_model≈768`, 8 heads, RoPE embeddings.
- CVM-core attention module per layer: buffer size `s_ℓ` (e.g., 64–256), threshold `p_ℓ`, representative mapping `r(i)`, convex reconstruction weights.
- Low-rank shared projections (head tying) for parameter reduction; QAT-ready (`INT8/INT4`).

## Data & Preprocessing
- Corpora: OPUS (Korean↔English), AI Hub KR↔EN, TED talks, newswire; clean and deduplicate.
- Audio: Korean speech datasets (Common Voice, internal), paired texts for supervised S2T.
- Text normalization: Unicode NFC, punctuation repair, romanization handling; sentencepiece or BPE vocab with shared KR/EN subwords.
- Streaming segmentation: 200–300 ms windows with VAD; incremental punctuation and casing repair.

## Training Pipeline
- Stage A: Base MT pretraining (seq2seq)
- Train SLM (no CVM compression initially) on KR↔EN parallel text; establish baseline.
- Stage B: CVM integration
- Introduce per-layer CVM buffers; gradually increase compression (ramp `s_ℓ` down, tighten `p_ℓ`).
- Losses: CE for translation; auxiliary reconstruction loss for non-core tokens; attention alignment regularizer.
- Stage C: Distillation from teacher (LLM or NMT)
- Teacher provides logits, hidden states, and attention maps.
- KD losses: logit KL, hidden MSE, attention alignment (e.g., EMD); blend with CE.
- Stage D: Controlled forgetting curriculum
- Schedule `p_ℓ`/`s_ℓ` to promote forgetting of rarely-activated contexts; monitor performance changes and cap perturbation via bounds.
- Stage E: Quantization-aware training
- Insert fake quant nodes; calibrate per-channel scales; ensure CVM computations preserve estimator properties under quantization.

## Proofs & Theory (Deliverables)
- Unbiasedness/variance: Restate CVM estimator assumptions; derive `E[|B|/p]=D`, `Var` bounds `O(1/s)`.
- Attention perturbation bounds: Bound softmax weight change when replacing tokens by cores; derive error ≤ `Δ` per logit under convex reconstruction.
- Convergence: Show expected descent with compressed gradients when variance ≤ target; provide Lipschitz conditions and step-size constraints.
- Stability: Bound forgetting-induced drift using merge schedule; show bounded output perturbation across layers.

## Inference Pipeline (Real-Time)
- ASR Frontend: Whisper streaming (tiny/base) with VAD; partial hypotheses every 200–300 ms; timestamps and confidences.
- MT Engine: CVM-SLM decoder operates incrementally per ASR chunk; speculative decoding with short beams (beam=2–4) or greedy + grammar constraints.
- Bi-directional translation: Separate tokenization pipelines for KR and EN; shared SLM weights with language tags.
- KV-cache compaction: Store only core keys/values per segment; maintain `B_ℓ` buffers; reconstruct non-core contributions via convex weights when needed.
- Fallbacks: Low ASR confidence → re-ask or LM denoise; punctuation/segmentation post-processing; short-retry with higher compression for speed.
- Latency budget: ASR 200–250 ms; MT 150–200 ms; I/O 50 ms → <500 ms end-to-end.

## Deployment & Optimization
- Edge targets: Windows/Android/Linux x64/ARM; ONNX/TensorRT/DirectML runtimes.
- Quantization: INT8 per-channel for matmuls; INT4 for KV-cache; fused attention kernels over cores.
- Memory: KV-core caches `O(s·d·L)` vs `O(n·d·L)`; pooled allocators; treap or heap structures for `B_ℓ` with `O(log s)` updates.
- Parallelism: Thread pinning, micro-batching at 5–10 ms, speculative decode slots.

## Evaluation & Ablations
- Metrics: SacreBLEU/BLEU, chrF; latency distribution (P50/P95); memory footprint; parameter count; throughput; robustness under forgetting.
- Ablations: buffer size `s`, `ε/δ`, per-layer vs global cores, forgetting schedules, projection ranks, beam widths; KR→EN vs EN→KR symmetry.
- Baselines: Non-CVM SLM; HLL-based compaction; standard NMT.

## System Architecture (Services)
- Components: ASR service (Whisper), MT service (CVM-SLM), Orchestrator (chunking, routing), Telemetry (latency/memory), Fallback handler.
- APIs: gRPC/HTTP streaming endpoints; protobuf messages with chunk timestamps, confidences, language tags.
- Observability: Tracing per-chunk; CVM buffer stats; failure and retry counters; on-device dashboards.

## Deliverables
- Mathematical report: formal definitions, proofs, bounds.
- Model code: CVM-core attention module; training loops; KD scripts; quantization tooling.
- Services: Streaming ASR/MT endpoints; orchestrator; fallback handlers.
- Evaluation suite: metrics, benchmarks, ablation harness; latency instrumentation.

## Timeline
- Week 1–2: Baseline SLM and data pipeline; initial MT accuracy.
- Week 3–4: CVM-core attention integration; preliminary latency/memory reductions.
- Week 5–6: Distillation and proofs draft; edge quantization; streaming services.
- Week 7–8: Ablations; KR↔EN parity tuning; final report and benchmarks.

## Risks & Mitigations
- ASR noise: Use confidence gating, denoising, and retry logic.
- Over-compression: Monitor perturbation bounds; dynamically adjust `s/p`.
- Edge variability: Provide runtime-agnostic kernels; fall back to CPU with smaller `s`.

## Next Steps
- Implement CVM-core attention module; assemble training scripts with KD; stand up ASR/MT streaming services; produce initial latency and memory benchmarks for confirmation.