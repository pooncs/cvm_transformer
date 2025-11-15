## Goal
Achieve near-perfect (≈99%) exact-match translation on the project’s Korean→English benchmark by redesigning the model, scaling high-quality data, and upgrading training, decoding, and evaluation.

## Architecture Overview
- Text NMT Backbone: Transformer Big encoder–decoder
  - Embeddings: shared `src/tgt` embeddings (SentencePiece, vocab≈32k), positional embeddings (Sinusoidal or Rotary).
  - Encoder: 12 layers, `d_model=1024`, `n_heads=16`, `ff_dim=4096`, dropout 0.1.
  - Decoder: 12 layers, cross-attention to encoder states, identical dimensions.
  - Regularization: label smoothing 0.1, dropout 0.1, gradient clipping 1.0.
- Multimodal Extensions (optional but aligned with tests)
  - Image Encoder: ViT-B/16 pretrained; project CLS into `d_model` and add learned `[IMG]` token; gated cross-attention in decoder.
  - Audio Encoder: wav2vec2-base or Whisper-small; mean-pool or attentive-pool features → linear projection into `d_model`; learned `[AUDIO]` token; cross-attention.
  - Fusion: late fusion via per-modality cross-attention blocks + gating; modality dropout to avoid reliance on non-text inputs.
- Inference & Decoding
  - Beam search size 5–10, length penalty 1.0–1.2, coverage penalty to reduce omissions.
  - Lexical constraints for known named entities; detokenization post-processing rules (honorifics, spacing).
- Efficiency Options
  - Mixed precision (fp16/bf16), gradient accumulation, FlashAttention/xFormers when available, checkpointing for long sequences, tie embeddings and output softmax.

## Data Expansion & Quality
- Sources (KR↔EN parallel corpora)
  - AI Hub, ParaCrawl, OpenSubtitles, Tatoeba, Global Voices, JW300, News Commentary, TED Talks, UN, Wiki Titles.
- Cleaning & Filtering
  - Normalize punctuation/spacing, remove duplicates/near-duplicates, language ID filter, sentence length bounds (2–128), heuristic noise removal.
  - Alignment validation via bilingual sentence embeddings (LaBSE) to remove misaligned pairs.
- Domain Balancing
  - Create domain tags (news, conversational, technical, subtitles). Oversample domains relevant to the benchmark.
- Back-Translation
  - Bootstrap EN→KR synthetic data from a strong teacher (mBART-50/NLLB-200), weight bt_loss≈0.3.
- Scale Target
  - Target 10–50M clean sentence pairs; start with ≥5M for meaningful gains.

## Tokenizer & Vocabulary
- SentencePiece BPE
  - Vocab≈32k, char coverage 0.9995, shared across KR/EN; special tokens: `[BT]`, `[IMG]`, `[AUDIO]`, `[MASK]`, `[SEP]`, `<bos>=2`, `<eos>=3`, `<pad>=0`, `<unk>=1`.
- Train on combined cleaned corpus, enforce consistent normalization (NFKC) and Korean spacing rules.

## Training Strategy
- Baseline Text NMT
  - Optimizer: AdamW (or Lion after validation), `lr=2e-4`, cosine decay with warm restarts, warmup=8k steps, weight_decay=0.01.
  - Batch: tokens-per-batch 8k–16k with dynamic batcher; gradient_accumulation=4–8.
  - Epochs: train to 200–400k steps; early stopping on dev BLEU/chrF.
- Curriculum Learning
  - Phase 1: short/simple sentences; Phase 2: medium difficulty; Phase 3: complex + idioms and honorifics; Phase 4: domain specialization.
- Knowledge Distillation
  - Teacher: fine-tuned mBART-50/NLLB-200; distillation temperature T=2–4, alpha=0.5; loss = CE + KD.
- Adversarial/Noise
  - Token dropout, synonym replacement (KR/EN), slight punctuation noise; ensure non-degenerate augmentation.
- Regular Evaluation
  - Dev sets per domain; report BLEU, chrF, COMET; track exact-match on the project test list.

## Hyperparameter Tuning
- Grid/Population Search (W&B or in-house)
  - d_model: {768, 1024}, n_layers: {8, 12}, n_heads: {12, 16}, ff_dim: {3072, 4096}.
  - lr: {1e-4, 2e-4, 3e-4}, warmup: {4k, 8k, 12k}, dropout: {0.0, 0.1}.
  - beam size: {5, 8, 10}, length penalty: {1.0, 1.2}.
- Selection Metric
  - Primary: exact-match on curated KR→EN phrases; Secondary: BLEU/chrF averaged across domains.

## Quality Control & Human-in-the-Loop
- Error Analysis Loop
  - Categorize errors (honorifics, idioms, spacing, named entities, tense/politeness).
  - Add targeted micro-corpora and rules for frequent failure modes.
- Human Review
  - Periodic sampling for annotation; integrate corrections back into training via small fine-tunes.
- Safety & Robustness
  - Rejects for toxic content; consistency checks (round-trip & paraphrase tests); guardrails on named entities.

## Multimodal Testing Support
- Image
  - OCR (Tesseract or EasyOCR) to extract KR text; fuse ViT features for layout cues if necessary.
- Audio
  - ASR via Whisper-small to KR text; optionally train end-to-end with audio encoder to improve robustness.
- Unified Evaluation
  - Keep text as primary target; use multimodal inputs only when present; ensure modality-agnostic behavior.

## Repository Changes (Planned)
- New Model Modules
  - `src/models/nmt_transformer.py`: Transformer Big enc-dec with shared embeddings, cross-attention, beam search.
  - `src/models/multimodal_fusion.py`: ViT/Whisper encoders, fusion adapters, modality tokens.
- Training
  - `src/training/train_nmt.py`: text-only baseline training with curriculum, distillation, back-translation.
  - `configs/train_nmt.yaml`: canonical config; `configs/train_nmt_multimodal.yaml` for fusion.
- Data
  - `src/data/prepare_corpus.py`: ingestion, cleaning, filtering, domain tagging, SP training.
- Evaluation
  - `src/evaluation/metrics.py`: BLEU/chrF/COMET wrappers; exact-match suite; error analysis report.

## Milestones
- M1 (Week 1–2): Data pipeline + SP tokenizer; text-only Transformer Big baseline; achieve non-zero BLEU, >60% exact-match on curated phrases.
- M2 (Week 3–4): Distillation from mBART/NLLB; curriculum stages; reach >85% exact-match on curated phrases; BLEU>25 on dev.
- M3 (Week 5–6): Multimodal fusion, beam tuning, lexical constraints; >92% exact-match; BLEU>30.
- M4 (Week 7+): Scale data to 10M+ sentences, hyperparameter sweep, human loop; push toward ≈99% exact-match in scoped benchmark.

## Risks & Mitigations
- 99% global exact-match is unrealistic; scope to a well-defined benchmark and domains.
- Data quality dominates outcomes; invest in cleaning/alignment and high-quality sources.
- Compute demand: plan for multi-GPU training (A100/RTX 4090) with AMP and accumulation.

## Next Steps (Upon Approval)
1. Implement text NMT Transformer Big module and training script.
2. Build data preparation and tokenizer pipeline; ingest initial corpora.
3. Run baseline training with regular evaluations; wire the comprehensive test suite as a gate.
4. Add distillation from a strong teacher; tune decoding; iterate.
5. Extend to multimodal when text baseline stabilizes, then push metrics with curriculum + human-in-the-loop.