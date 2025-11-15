## SOTA Overview
- Small Language Models: DistilBERT, TinyBERT, MobileBERT, MiniLM, ALBERT; compact generative models (Llama 3.2 1B/3B, Gemma 2 2B, Phi-3-mini) tuned for edge tasks.
- Efficient Attention: Linformer, Performer, Reformer; FlashAttention-2 for GPU; KV-cache optimizations (PagedAttention in vLLM) and KV quantization (KIVI 2-bit) to reduce memory.
- Quantization: GPTQ (3–4 bit PTQ), AWQ (INT4 weight-aware) [AWQ MLSys’24, GitHub: mit-han-lab/llm-awq], SmoothQuant (activation migration), A8W8 / W4A16 mixed precision; edge stacks with TensorRT-LLM supporting INT4/FP8 and ONNX Runtime for portable INT8 [TensorRT-LLM review, 2024/2025].
- Edge Runtimes: ONNX Runtime (CPU/GPU/NPU), TensorRT-LLM (Jetson/RTX), Core ML (iOS), TFLite (Android), MNN/NCNN (mobile GPU). Survey reviews detail lifecycle design and runtime optimizations for edge LLMs [On-Device Language Models Review (2024), Edge LLMs Review (2024)].
- Language Detection:
  - Text: fastText lid.176 models (176 langs, 126MB or 917KB compressed) with high accuracy; alternatives CLD3/langid.py [fastText docs; short-text LangID review].
  - Audio: ECAPA‑TDNN trained on VoxLingua107 via SpeechBrain (107 langs, strong SLI baselines) [speechbrain/lang-id-voxlingua107-ecapa].
  - Images: OCR first (PaddleOCR/PP-OCRv4 or Tesseract) → run text LangID on extracted text; optional script detection by Unicode ranges for confidence bootstrapping.

## Current Repo Assessment
- Core model: `cvm_translator/cvm_transformer.py` (CVM-core attention), `slm_model.py`.
- Tokenizer: `sp_tokenizer.py` with SentencePiece; current prod vocab ~668.
- Streaming/ASR: `whisper_streaming.py`, `whisper_interface.py`.
- Serving: `grpc_server.py`, `grpc_int8_server.py`, `realtime_translation.proto`, client & telemetry.
- Training: `train_loop.py` (KD losses `kd_losses.py`), `prepare_corpus.py`, `ablation.py`, `metrics.py`.
- Quantization: `quantization.py` (INT8 scaffolding), with prior embedding qconfig limitation.

## Goals
- Add multimodal language detection (text, audio, image) and automatic routing.
- Upgrade SLM for higher quality at same or lower latency/memory.
- Edge-ready deployment across CPU/Jetson/mobile with INT8/INT4.
- Preserve <500 ms end-to-end latency target with robust telemetry.

## Upgrade Plan

### 1) Multimodal Language Detection Module
- Text LangID:
  - Integrate fastText `lid.176.bin` (high accuracy) with optional `lid.176.ftz` (917KB) for ultra-light edge; expose `detect_text_language(text) -> {lang, prob}`.
  - Fallback providers: CLD3 or langid.py; ensemble voting for short texts (<70 chars) per short-text benchmarks.
- Audio LangID:
  - Add SpeechBrain ECAPA‑TDNN VoxLingua107 model wrapper: `detect_audio_language(wave) -> {lang, prob}`; supports 16 kHz mono; include streaming gate with VAD windows.
  - Quantization strategy: ONNX export + ORT INT8 where feasible; cache embeddings per speaker/session.
- Image LangID:
  - Add OCR pipeline (PaddleOCR preferred for accuracy/speed; fallback Tesseract): `extract_text(image) -> text + boxes`.
  - Run Text LangID on OCR text; boost confidence via Unicode script distribution; multi-language detection if multiple regions.
- Unified API & Confidence:
  - `LanguageDetector.detect(input, modality=auto) -> {lang(s), confidence(s), modality}` with thresholds and abstain policy.

### 2) Pipeline & Protocol Integration
- gRPC proto updates: include `detected_language`, `modality`, `confidence`, `source_region` (for images), `audio_chunk_ms` metadata.
- Gating logic:
  - Auto-route KR↔EN translation direction from LangID; skip translation if source==target or confidence<τ.
  - Fallback to Whisper auto-language if audio LangID confidence low.
- Telemetry: record LangID decisions, confidences, and routing outcomes alongside latency/core-count.

### 3) SLM Quality Upgrade (CVM Transformer)
- Tokenizer:
  - Scale SentencePiece vocab to 8k–16k using OPUS/OpenSubtitles + AI‑Hub corpus via `prepare_corpus.py` (robust download, alignment, cleaning, de-dup).
  - Retain compact model path for edge (8k) with domain-specific merges for KR/EN.
- Distillation & Training:
  - Teacher options: NLLB/M2M100/MarianMT or a strong bilingual LLM; distill via losses already scaffolded (logit KL, hidden MSE, attention EMD).
  - Curriculum with short sentences, domain phrases, noisy text; evaluate BLEU/chrF; maintain CVM cores ablation.
- Attention & KV Cache:
  - Validate core capacity schedules (8 optimal from current ablations); support dynamic cores per sequence length.
  - Optional GPU path: FlashAttention-2; general path: efficient linear attention fallback.
  - Investigate KV cache quantization (KIVI-like per-channel keys/per-token values) to reduce memory while preserving quality.

### 4) Edge Inference Optimization
- Export:
  - Torch → ONNX export with dynamic axes (`batch`, `seq_len`) for encoder/decoder components.
  - CPU: ONNX Runtime with INT8 PTQ (per-channel weights); QAT for embeddings using float‑qparams configs.
  - Jetson: TensorRT (via ONNX or TensorRT‑LLM) with INT8 calibration or AWQ INT4 where applicable; test FP8 if supported.
  - Mobile: Core ML (iOS Metal) and TFLite (Android) conversion pipeline; evaluate MNN/NCNN for GPU acceleration where TFLite falls short.
- Quantization:
  - Weight-only GPTQ/AWQ for transformer blocks; A8W8 for activations if hardware supports.
  - Validate accuracy/latency trade-offs per device; maintain a per-target build matrix.

### 5) Detection & Translation Evaluation
- LangID metrics: accuracy@top‑1/top‑k, short-text robustness (<70 chars), audio LID accuracy on VoxLingua-style samples, OCR text coverage & confidence.
- Translation metrics: BLEU, chrF, latency distribution, memory footprint, core-count utilization.
- Continuous telemetry: JSON logs + plots (latency vs cores, memory vs cores); add CI benchmarks.

### 6) Deployment Targets
- Docker multi-stage images for CPU/GPU; feature flags for modality support.
- Jetson container with TensorRT and VAD; validate streaming at <500 ms E2E.
- Mobile SDK stubs (Android/iOS): local OCR + text LID + remote/edge translation; local-only path if models are sufficiently small.

### 7) Privacy & Security
- Process inputs on-device; avoid sending raw audio/images to remote unless explicitly configured.
- PII-safe logging; redact text and OCR snippets; store only metrics.
- gRPC TLS option; configurable retention window for telemetry.

## Milestones
- M1 (Week 1–2): Implement `LanguageDetector` (text/audio/image) with unit tests; proto extensions; telemetry hooks.
- M2 (Week 3–4): Corpus scaling & tokenizer 8k–16k; baseline distillation run; CVM core schedule finalized.
- M3 (Week 5–6): ONNX export; ORT INT8; Jetson TensorRT path; AWQ/GPTQ experiments; end-to-end latency <500 ms reconfirmed.
- M4 (Week 7–8): Mobile conversions (Core ML/TFLite/MNN); light client SDK; OCR+LangID device workflows.
- M5 (Week 9): Full benchmark suite; final plots & JSON; deployment docs and sample apps.

## References
- TensorRT‑LLM & edge quantization overview (AWQ, SmoothQuant, FP8/INT4): https://www.rohan-paul.com/p/edge-deployment-of-llms-and-ml-models
- AWQ (MLSys 2024 Best Paper), TinyChat for edge 4‑bit LLM/VLM: https://github.com/mit-han-lab/llm-awq
- On‑Device Language Models Survey (2024): https://arxiv.org/html/2409.00088v2
- Edge LLMs Review (design/execution/applications): https://arxiv.org/html/2410.11845v1
- Small LM optimization in practice (QLoRA + GPTQ/AWQ trade‑offs): https://arxiv.org/html/2510.21970
- Text LangID fastText models (176 langs): https://fasttext.cc/docs/en/language-identification.html
- Short‑text LangID comparison (fastText/CLD3/langid): https://medium.com/besedo-engineering/language-identification-for-very-short-texts-a-review-c9f2756773ad
- Audio LangID ECAPA‑TDNN on VoxLingua107 (SpeechBrain): https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa