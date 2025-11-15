import torch
from cvm_translator.cvm_transformer import CVMTransformer
from cvm_translator.cvm_translate import cvm_translate_chunk
from cvm_translator.metrics import MetricsCollector
import itertools


def dummy_tokenizer(text):
    class Dummy:
        def encode(self, text):
            return [hash(w) % 1000 for w in text.strip().split()]
        def decode(self, ids):
            return [str(i) for i in ids]
    return Dummy()


def ablate_core_capacity():
    tokenizer = dummy_tokenizer("dummy")
    pairs = [("안녕하세요", "Hello"), ("오늘 날씨 좋네요", "Today weather is nice")]
    capacities = [4, 8, 16, 32, 64]
    results = {}
    for c in capacities:
        model = CVMTransformer(vocab_size=1000, d_model=128, n_layers=2)
        metrics = MetricsCollector()
        for src, ref in pairs:
            metrics.start()
            pred = cvm_translate_chunk(model, tokenizer, src, "KR_EN", core_capacity=c)
            metrics.end()
            metrics.record_memory()
        results[c] = metrics.report()
    return results


def ablate_forgetting():
    # stub: vary merge thresholds and retrain
    return {"forgetting": "stub"}


if __name__ == "__main__":
    print("Core capacity ablation:")
    print(ablate_core_capacity())
    print("Forgetting ablation:")
    print(ablate_forgetting())