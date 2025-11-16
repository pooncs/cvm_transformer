#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from src.models.cvm_transformer import CVMTransformer
from src.evaluation.validation_protocol import ValidationConfig, DistillationValidator


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler(log_dir / "train.log", maxBytes=1_000_000, backupCount=3)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


class BiTextDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, max_len: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.tokenizer.encode(src)[:self.max_len]
        tgt_ids = self.tokenizer.encode(tgt)[:self.max_len]
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }


def collate(batch):
    src_ids = torch.nn.utils.rnn.pad_sequence([b["src_ids"] for b in batch], batch_first=True, padding_value=0)
    tgt_ids = torch.nn.utils.rnn.pad_sequence([b["tgt_ids"] for b in batch], batch_first=True, padding_value=0)
    return {"src_ids": src_ids, "tgt_ids": tgt_ids}


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1}

    def encode(self, text: str):
        return [self.vocab.get(ch, self.vocab["<unk>"]) for ch in text]


def run_training(cfg: dict):
    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    vocab_size = cfg["vocab_size"]
    d_model = cfg["d_model"]
    n_layers_student = cfg["n_layers_student"]
    n_layers_teacher = cfg["n_layers_teacher"]
    core_capacity_student = cfg["core_capacity_student"]
    core_capacity_teacher = cfg["core_capacity_teacher"]
    max_len = cfg["max_len"]
    batch_size = cfg["batch_size"]
    lr = float(cfg["learning_rate"])
    total_iters = cfg["total_iterations"]
    val_freq = cfg["validation_frequency"]

    log_dir = Path(cfg.get("log_dir", "logs"))
    reports_dir = Path(cfg.get("reports_dir", "reports"))
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    for d in [log_dir, reports_dir, ckpt_dir]:
        d.mkdir(exist_ok=True)

    logger = setup_logger(log_dir)
    logger.info(f"Using device: {device}")

    student = CVMTransformer(vocab_size, d_model=d_model, n_layers=n_layers_student, core_capacity=core_capacity_student).to(device)
    teacher = CVMTransformer(vocab_size, d_model=d_model, n_layers=n_layers_teacher, core_capacity=core_capacity_teacher).to(device)

    base_pairs = [
        ("안녕하세요", "Hello"),
        ("오늘 날씨 좋네요", "Today weather is nice"),
        ("실시간 번역", "real-time translation"),
        ("한국어 영어", "Korean English"),
    ]

    pairs = []
    for _ in range(2500):
        pairs.extend(base_pairs)

    tokenizer = SimpleTokenizer()
    dataset = BiTextDataset(pairs, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # Optimizer selection
    if cfg.get("use_lion_optimizer"):
        try:
            from lion_pytorch import Lion
            optimizer = Lion(student.parameters(), lr=lr)
            logger.info("Using Lion optimizer")
        except ImportError:
            logger.warning("Lion not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    elif cfg.get("use_adafactor_optimizer"):
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(student.parameters(), lr=lr, relative_step=False)
            logger.info("Using Adafactor optimizer")
        except ImportError:
            logger.warning("Adafactor not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
        logger.info("Using AdamW optimizer")

    vconfig = ValidationConfig(validation_frequency=val_freq)
    validator = DistillationValidator(vconfig, teacher_model=teacher)

    iters = 0
    losses: List[float] = []
    t0 = time.time()

    while iters < total_iters:
        for batch in loader:
            src = batch["src_ids"].to(device)
            tgt = batch["tgt_ids"].to(device)
            optimizer.zero_grad()

            s_logits, s_h, s_attn = student(src, return_hidden=True, return_attn=True)
            with torch.no_grad():
                t_logits, t_h, t_attn = teacher(src, return_hidden=True, return_attn=True)

            tgt_aligned = tgt[:, :s_logits.size(1)]
            ce = torch.nn.functional.cross_entropy(
                s_logits.reshape(-1, s_logits.size(-1)), tgt_aligned.reshape(-1), ignore_index=0
            )
            kd = torch.nn.functional.mse_loss(s_logits, t_logits)
            loss = ce + 0.5 * kd

            loss.backward()
            optimizer.step()

            iters += 1
            losses.append(float(loss.item()))
            if iters % 100 == 0:
                logger.info(f"iter={iters} loss={loss.item():.6f}")

            if iters % val_freq == 0:
                val_data = [(src_str, tgt_str) for src_str, tgt_str in base_pairs]
                result = validator.validate_distillation(student, val_data, iters, device=str(device))
                logger.info(f"validation iter={iters} quality={result.quality_score:.4f}")
                ckpt_path = ckpt_dir / f"student_iter_{iters}.pt"
                torch.save({"model_state": student.state_dict(), "optimizer_state": optimizer.state_dict(), "iters": iters}, ckpt_path)
                logger.info(f"checkpoint saved to {ckpt_path}")

            if iters >= total_iters:
                break

    elapsed = time.time() - t0
    report = {
        "config": cfg,
        "total_iterations": iters,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "elapsed_sec": elapsed,
    }
    with open(reports_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    final_ckpt_path = ckpt_dir / "student_final.pt"
    torch.save({"model_state": student.state_dict(), "optimizer_state": optimizer.state_dict(), "iters": iters}, final_ckpt_path)
    logger.info(f"final checkpoint saved to {final_ckpt_path}")
    logger.info(f"completed {iters} iterations in {elapsed:.2f}s avg_loss={report['avg_loss']:.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_training(cfg)