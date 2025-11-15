## Scope and Assumptions
- The provided URL points to Knuth’s “CVM” for distinct-element estimation, not Core Vector Machines. I proceed with Core Vector Machines by Tsang–Kwok–Cheung (JMLR 2005) and coreset/MEB literature.
- Goals: integrate CVM principles into transformers for parameter-efficient learning and controlled forgetting; design an SLM and a KR↔EN real-time translation pipeline.

## Research Foundations
- Review CVM’s MEB/coreset equivalence and guarantees: linear-in-data-time and data-size–independent memory with (1+ε)-approximation.
- Gather precise bounds from JMLR 2005 and Bădoiu–Clarkson coreset theory; catalog the mapping to kernel spaces and farthest-point iterations.
- Extract transformer-relevant insights: subset selection of keys/values, convex-combination representation, softmax stability under logit perturbations, per-layer core memory.

## CVM→Transformer Integration
- Define per-layer token representation sets and compute streaming (1+ε)-MEB core-sets of keys/values.
- Replace full attention with attention over core tokens; represent non-core tokens via convex combinations.
- Add controlled forgetting: age-weighted core maintenance; drop points inside ball with margin that provably do not shift the center beyond ε.
- Prove convergence/stability: bound attention logit perturbations, gradient Lipschitzness, and accumulated error across layers.
- Complexity: reduce per-layer attention from O(n²d) to O(n·c·d) (c≈O(ε⁻²)), and KV-cache memory to O(c·d·L).

## Small Language Model (SLM) Design & Distillation
- Architecture: 12–16 layers, hidden 768, 8 heads, rotary embeddings; CVM core attention module; shared low-rank projections.
- Distillation protocol: teacher LLM supplies logits, hidden states, and attention; student trains on teacher signals with CVM core selection.
- Attention compression: per-layer core-set building; softmax over cores; reconstruct non-core contributions via convex weights.
- Metrics: task accuracy, logit KL, attention Earth Mover’s distance, parameter count, memory, latency.
- Ablations: ε, core size c, per-layer vs global cores, forgetting thresholds, rank of projections.

## Real-Time KR↔EN Translation Pipeline
- Frontend: Whisper streaming (tiny/base) for Korean STT with VAD and incremental decoding.
- MT: CVM-enhanced transformer for bi-directional translation; streaming chunking (200–300 ms windows), speculative decoding.
- Fallbacks: confidence-based reask, punctuation repair, LM-based denoising, short-retry loop.
- Latency budget: STT 200–250 ms, MT 150–200 ms, I/O 50 ms ⇒ <500 ms end-to-end.
- Edge optimization: INT8/INT4 quantization, fused kernels, KV-core caches, memory pooling.

## Deliverables
- Mathematical report: formal definitions, theorems, proofs, and complexity bounds.
- SLM code skeleton: CVM-core attention module, training loop, distillation scripts.
- Translation service design: streaming APIs, latency instrumentation, bidirectional flows, fallback handlers.
- Evaluation suite: metrics, test datasets, and ablation scripts.

## Next Steps
- Confirm these plan components; I will then implement the CVM-core attention module, the distillation pipeline, and the KR↔EN service, and deliver proofs, benchmarks, and ablation results.