# CVM-Enhanced Real-Time Korean↔English Translator

A compact, low-latency, memory-efficient translation system using the CVM (Count–Vector–Merge) algorithm for core token selection and controlled forgetting.

## Quick Start

```bash
# Run streaming demo (KR→EN)
python -m cvm_translator.realtime_fallback_demo

# Run benchmark
python -m cvm_translator.benchmark

# Train with KD
python -m cvm_translator.train_loop
```

## Architecture

- **WhisperStreamer**: Real-time Korean ASR with VAD and 300 ms chunks.
- **CVMTransformer**: 12-layer encoder-decoder with per-layer CVM-core attention (capacity 64–256).
- **CVMBuffer**: Unbiased distinct-element reservoir; merges maintain memory bound.
- **Fallback**: Confidence-based retry, LM denoising, punctuation repair.
- **Metrics**: Latency, memory, BLEU/chrF tracking.

## Training

- Distillation from teacher LLM (logits, hidden, attention) with KD, hidden, and attention losses.
- Controlled forgetting via CVM merge policy; accuracy vs memory trade-offs ablated.

## Performance Targets

- End-to-end latency <500 ms on edge devices.
- Memory footprint reduced via CVM KV-cache compaction (O(s·d·L) vs O(n·d·L)).
- Parameter-efficient: shared low-rank projections, INT8/INT4 quantization ready.

## Dependencies

- torch, whisper, sounddevice, psutil, numpy

## License

MIT