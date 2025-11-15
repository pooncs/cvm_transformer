import time
import psutil
import torch


class MetricsCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.latencies = []
        self.memories = []
        self.bleu_scores = []
        self.chrf_scores = []

    def start(self):
        self.start_time = time.perf_counter()

    def end(self):
        self.latencies.append((time.perf_counter() - self.start_time) * 1000)

    def record_memory(self):
        self.memories.append(psutil.virtual_memory().used / 1024 ** 2)

    def add_bleu(self, score):
        self.bleu_scores.append(score)

    def add_chrf(self, score):
        self.chrf_scores.append(score)

    def report(self):
        return {
            "latency_ms": {
                "mean": sum(self.latencies) / len(self.latencies) if self.latencies else 0,
                "max": max(self.latencies) if self.latencies else 0,
                "min": min(self.latencies) if self.latencies else 0,
            },
            "memory_MB": {
                "mean": sum(self.memories) / len(self.memories) if self.memories else 0,
                "max": max(self.memories) if self.memories else 0,
            },
            "bleu": {
                "mean": sum(self.bleu_scores) / len(self.bleu_scores) if self.bleu_scores else 0,
            },
            "chrf": {
                "mean": sum(self.chrf_scores) / len(self.chrf_scores) if self.chrf_scores else 0,
            }
        }