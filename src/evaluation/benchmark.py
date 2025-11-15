import torch
from cvm_translator.cvm_transformer import CVMTransformer
from cvm_translator.cvm_translate import cvm_translate_chunk
from cvm_translator.metrics import MetricsCollector


def dummy_tokenizer(text):
    class Dummy:
        def encode(self, text):
            return text.strip().split()
        def decode(self, ids):
            return ids
    return Dummy()


def eval_bleu(pred, ref):
    return 0.5  # stub


def eval_chrf(pred, ref):
    return 0.6  # stub


def benchmark(model, pairs, tokenizer, core_capacity=64):
    metrics = MetricsCollector()
    for src, ref in pairs:
        metrics.start()
        pred = cvm_translate_chunk(model, tokenizer, src, "KR_EN", core_capacity)
        metrics.end()
        metrics.record_memory()
        metrics.add_bleu(eval_bleu(pred, ref))
        metrics.add_chrf(eval_chrf(pred, ref))
    return metrics.report()


def main():
    tokenizer = dummy_tokenizer("dummy")
    model = CVMTransformer(vocab_size=1000, d_model=128, n_layers=2)
    pairs = [("안녕하세요", "Hello"), ("오늘 날씨 좋네요", "Today weather is nice")]
    report = benchmark(model, pairs, tokenizer, core_capacity=8)
    print(report)


if __name__ == "__main__":
    main()