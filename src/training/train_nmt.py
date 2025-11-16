"""
Baseline training script for Korean-English NMT with curriculum learning and knowledge distillation.
Implements advanced training techniques to achieve high translation quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import sentencepiece as spm
import yaml
import json
import time
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
from tqdm import tqdm
import wandb
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Import custom modules
import sys

sys.path.append(".")
from src.models.nmt_transformer import NMTTransformer, create_nmt_transformer
from src.data.prepare_corpus import CorpusConfig, prepare_corpus
from src.utils.metrics import BLEUScore, compute_translation_accuracy

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model parameters
    d_model: int = 1024
    n_heads: int = 16
    n_encoder_layers: int = 12
    n_decoder_layers: int = 12
    d_ff: int = 4096
    dropout: float = 0.1
    max_len: int = 512

    # Training parameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 8000
    max_steps: int = 400000
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0

    # Curriculum learning
    curriculum_stages: List[Dict[str, Any]] = None
    use_curriculum: bool = True

    # Knowledge distillation
    use_distillation: bool = True
    teacher_model_path: str = ""
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.5

    # Data augmentation
    use_augmentation: bool = True
    back_translation_weight: float = 0.3
    noise_prob: float = 0.1

    # Optimization
    use_mixed_precision: bool = True
    use_flash_attention: bool = True

    # Evaluation
    eval_steps: int = 1000
    save_steps: int = 5000
    logging_steps: int = 100

    # Paths
    data_dir: str = "data/processed"
    output_dir: str = "models/nmt"
    tokenizer_path: str = "data/processed/sentencepiece.model"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "korean-english-nmt"

    def __post_init__(self):
        if self.curriculum_stages is None:
            self.curriculum_stages = [
                {"name": "simple", "max_length": 20, "min_length": 2, "weight": 0.4},
                {"name": "medium", "max_length": 50, "min_length": 10, "weight": 0.4},
                {"name": "complex", "max_length": 100, "min_length": 20, "weight": 0.2},
            ]


class TranslationDataset(Dataset):
    """Dataset for translation pairs."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: spm.SentencePieceProcessor,
        max_length: int = 512,
        stage_config: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage_config = stage_config

        # Filter data based on curriculum stage
        if stage_config:
            self.data = self._filter_by_stage(data, stage_config)

    def _filter_by_stage(
        self, data: List[Dict[str, str]], stage_config: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Filter data based on curriculum stage."""
        filtered = []
        for item in data:
            src_len = len(item["src"].split())
            tgt_len = len(item["tgt"].split())

            if (
                stage_config["min_length"] <= src_len <= stage_config["max_length"]
                and stage_config["min_length"] <= tgt_len <= stage_config["max_length"]
            ):
                filtered.append(item)

        return filtered

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize source and target
        src_tokens = self.tokenizer.encode(item["src"], out_type=int)
        tgt_tokens = self.tokenizer.encode(item["tgt"], out_type=int)

        # Add special tokens
        src_tokens = [self.tokenizer.bos_id()] + src_tokens + [self.tokenizer.eos_id()]
        tgt_tokens = [self.tokenizer.bos_id()] + tgt_tokens + [self.tokenizer.eos_id()]

        # Truncate if necessary
        src_tokens = src_tokens[: self.max_length]
        tgt_tokens = tgt_tokens[: self.max_length]

        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_tokens, dtype=torch.long),
            "src_length": len(src_tokens),
            "tgt_length": len(tgt_tokens),
            "domain": item.get("domain", "general"),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching."""
    # Find max lengths
    max_src_len = max(item["src_length"] for item in batch)
    max_tgt_len = max(item["tgt_length"] for item in batch)

    # Pad sequences
    src_padded = torch.zeros(len(batch), max_src_len, dtype=torch.long)
    tgt_padded = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)
    src_lengths = torch.zeros(len(batch), dtype=torch.long)
    tgt_lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        src_len = item["src_length"]
        tgt_len = item["tgt_length"]

        src_padded[i, :src_len] = item["src"][:src_len]
        tgt_padded[i, :tgt_len] = item["tgt"][:tgt_len]
        src_lengths[i] = src_len
        tgt_lengths[i] = tgt_len

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_lengths": src_lengths,
        "tgt_lengths": tgt_lengths,
    }


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [batch_size, seq_len, vocab_size]
            target: Target tokens [batch_size, seq_len]
        """
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)

        # Create smoothed target distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        # Mask padding tokens
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        return torch.mean(
            torch.sum(-true_dist * torch.log_softmax(pred, dim=-1), dim=-1)
        )


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            target: Ground truth targets
        """
        # Cross-entropy loss with ground truth
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            target.view(-1),
            ignore_index=0,
        )

        # Knowledge distillation loss
        T = self.temperature
        student_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        kd_loss = F.kl_div(
            student_probs.view(-1, student_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction="batchmean",
        ) * (T * T)

        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss


class CosineAnnealingWarmRestarts:
    """Cosine annealing with warm restarts scheduler."""

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.T_cur = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_0:
                self.T_cur = self.T_cur - self.T_0
                self.T_0 = self.T_0 * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(
                    "Expected non-negative epoch, but got {}".format(epoch)
                )
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_0 = self.T_0 * self.T_mult**n
            else:
                self.T_cur = epoch
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    def get_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_0))
            / 2
            for base_lr in self.base_lrs
        ]


class NMTTrainer:
    """Trainer for NMT model with curriculum learning and knowledge distillation."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Initialize tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(config.tokenizer_path)

        # Initialize model
        self.model = self._create_model()

        # Initialize teacher model for distillation
        self.teacher_model = None
        if config.use_distillation:
            self.teacher_model = self._load_teacher_model()

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Initialize losses
        self.criterion = LabelSmoothingLoss(
            self.tokenizer.vocab_size(),
            padding_idx=self.tokenizer.pad_id(),
            smoothing=config.label_smoothing,
        )

        if config.use_distillation:
            self.distillation_criterion = KnowledgeDistillationLoss(
                temperature=config.distillation_temperature,
                alpha=config.distillation_alpha,
            )

        # Initialize logging
        if config.use_wandb:
            wandb.init(project=config.wandb_project, config=config.__dict__)

    def _create_model(self) -> NMTTransformer:
        """Create NMT model."""
        model_config = {
            "src_vocab_size": self.tokenizer.vocab_size(),
            "tgt_vocab_size": self.tokenizer.vocab_size(),
            "d_model": self.config.d_model,
            "n_heads": self.config.n_heads,
            "n_encoder_layers": self.config.n_encoder_layers,
            "n_decoder_layers": self.config.n_decoder_layers,
            "d_ff": self.config.d_ff,
            "max_len": self.config.max_len,
            "dropout": self.config.dropout,
            "pad_id": self.tokenizer.pad_id(),
            "use_flash": self.config.use_flash_attention,
        }

        model = create_nmt_transformer(model_config)
        model = model.to(self.device)

        logger.info(f"Model parameters: {model.count_parameters():,}")

        return model

    def _load_teacher_model(self) -> Optional[NMTTransformer]:
        """Load teacher model for knowledge distillation."""
        if not self.config.teacher_model_path:
            logger.warning("No teacher model path provided, using mBART-50 as fallback")
            return None

        try:
            checkpoint = torch.load(
                self.config.teacher_model_path, map_location=self.device
            )
            teacher_config = checkpoint.get("config", {})

            teacher_model = create_nmt_transformer(teacher_config)
            teacher_model.load_state_dict(checkpoint["model_state_dict"])
            teacher_model.eval()

            logger.info(f"Loaded teacher model from {self.config.teacher_model_path}")
            return teacher_model
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            return None

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10000, T_mult=2, eta_min=1e-6
        )

    def create_dataloader(
        self, data: List[Dict[str, str]], stage_config: Optional[Dict[str, Any]] = None
    ) -> DataLoader:
        """Create dataloader for specific curriculum stage."""
        dataset = TranslationDataset(
            data, self.tokenizer, self.config.max_len, stage_config
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    def train_epoch(
        self, dataloader: DataLoader, epoch: int, stage_name: str
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        progress_bar = tqdm(dataloader, desc=f"Training {stage_name} (Epoch {epoch})")

        for batch_idx, batch in enumerate(progress_bar):
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)

            # Create target input and output
            tgt_input = tgt[:, :-1]  # Remove last token for input
            tgt_output = tgt[:, 1:]  # Remove first token for output

            self.optimizer.zero_grad()

            # Forward pass
            if self.config.use_mixed_precision:
                with autocast():
                    output = self.model(src, tgt_input)

                    if self.config.use_distillation and self.teacher_model:
                        with torch.no_grad():
                            teacher_output = self.teacher_model(src, tgt_input)
                        loss = self.distillation_criterion(
                            output, teacher_output, tgt_output
                        )
                    else:
                        loss = self.criterion(output, tgt_output)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(src, tgt_input)

                if self.config.use_distillation and self.teacher_model:
                    with torch.no_grad():
                        teacher_output = self.teacher_model(src, tgt_input)
                    loss = self.distillation_criterion(
                        output, teacher_output, tgt_output
                    )
                else:
                    loss = self.criterion(output, tgt_output)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Update scheduler
            self.scheduler.step()

            # Logging
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_tokens += (tgt_output != self.tokenizer.pad_id()).sum().item()

            if batch_idx % self.config.logging_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr:.2e}",
                        "tokens": total_tokens,
                    }
                )

                if self.config.use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/learning_rate": lr,
                            "train/tokens": total_tokens,
                            "train/stage": stage_name,
                        }
                    )

        avg_loss = total_loss / len(dataloader)

        return {"loss": avg_loss, "tokens": total_tokens, "stage": stage_name}

    def evaluate(
        self, dataloader: DataLoader, stage_name: str = "eval"
    ) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        total_bleu = 0
        total_exact_match = 0
        num_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {stage_name}"):
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)

                # Create target input and output
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # Forward pass
                output = self.model(src, tgt_input)
                loss = self.criterion(output, tgt_output)

                total_loss += loss.item()
                total_tokens += (tgt_output != self.tokenizer.pad_id()).sum().item()

                # Calculate BLEU score
                pred_tokens = torch.argmax(output, dim=-1)
                bleu_score = self._calculate_bleu(pred_tokens, tgt_output)
                total_bleu += bleu_score

                # Calculate exact match
                exact_match = self._calculate_exact_match(pred_tokens, tgt_output)
                total_exact_match += exact_match

                num_samples += src.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_bleu = total_bleu / len(dataloader)
        avg_exact_match = total_exact_match / num_samples

        return {
            "loss": avg_loss,
            "bleu": avg_bleu,
            "exact_match": avg_exact_match,
            "tokens": total_tokens,
            "stage": stage_name,
        }

    def _calculate_bleu(
        self, pred_tokens: torch.Tensor, target_tokens: torch.Tensor
    ) -> float:
        """Calculate BLEU score."""
        # Convert to text and calculate BLEU
        # This is a simplified version - in practice you'd use a proper BLEU implementation
        pred_text = []
        target_text = []

        for i in range(pred_tokens.size(0)):
            pred_seq = pred_tokens[i].cpu().numpy()
            target_seq = target_tokens[i].cpu().numpy()

            # Remove padding and special tokens
            pred_seq = pred_seq[pred_seq != self.tokenizer.pad_id()]
            target_seq = target_seq[target_seq != self.tokenizer.pad_id()]

            # Remove BOS/EOS tokens
            pred_seq = pred_seq[pred_seq != self.tokenizer.bos_id()]
            pred_seq = pred_seq[pred_seq != self.tokenizer.eos_id()]
            target_seq = target_seq[target_seq != self.tokenizer.bos_id()]
            target_seq = target_seq[target_seq != self.tokenizer.eos_id()]

            pred_text.append(" ".join(map(str, pred_seq)))
            target_text.append(" ".join(map(str, target_seq)))

        # Simple BLEU calculation (would use proper BLEU in production)
        bleu_scores = []
        for pred, target in zip(pred_text, target_text):
            pred_words = pred.split()
            target_words = target.split()

            if len(target_words) == 0:
                continue

            # Calculate 1-gram precision
            matches = sum(1 for word in pred_words if word in target_words)
            precision = matches / len(pred_words) if len(pred_words) > 0 else 0

            bleu_scores.append(precision)

        return np.mean(bleu_scores) if bleu_scores else 0.0

    def _calculate_exact_match(
        self, pred_tokens: torch.Tensor, target_tokens: torch.Tensor
    ) -> float:
        """Calculate exact match rate."""
        exact_matches = 0

        for i in range(pred_tokens.size(0)):
            pred_seq = pred_tokens[i].cpu().numpy()
            target_seq = target_tokens[i].cpu().numpy()

            # Remove padding and special tokens
            pred_seq = pred_seq[pred_seq != self.tokenizer.pad_id()]
            target_seq = target_seq[target_seq != self.tokenizer.pad_id()]

            # Remove BOS/EOS tokens
            pred_seq = pred_seq[pred_seq != self.tokenizer.bos_id()]
            pred_seq = pred_seq[pred_seq != self.tokenizer.eos_id()]
            target_seq = target_seq[target_seq != self.tokenizer.bos_id()]
            target_seq = target_seq[target_seq != self.tokenizer.eos_id()]

            # Check exact match
            if np.array_equal(pred_seq, target_seq):
                exact_matches += 1

        return exact_matches / pred_tokens.size(0)

    def save_checkpoint(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
            "metrics": metrics,
            "tokenizer_path": self.config.tokenizer_path,
        }

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if "exact_match" in metrics and metrics["exact_match"] > 0.8:
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def train(
        self, train_data: List[Dict[str, str]], val_data: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """Main training loop with curriculum learning."""
        logger.info("Starting training with curriculum learning")

        best_metrics = {"exact_match": 0.0}

        # Curriculum learning stages
        if self.config.use_curriculum:
            stages = self.config.curriculum_stages
        else:
            stages = [
                {
                    "name": "all",
                    "max_length": self.config.max_len,
                    "min_length": 2,
                    "weight": 1.0,
                }
            ]

        global_step = 0

        for stage_idx, stage_config in enumerate(stages):
            logger.info(
                f"Training stage {stage_idx + 1}/{len(stages)}: {stage_config['name']}"
            )

            # Create stage-specific dataloaders
            train_loader = self.create_dataloader(train_data, stage_config)
            val_loader = self.create_dataloader(val_data)

            # Train for multiple epochs per stage
            stage_epochs = max(1, int(stage_config.get("weight", 1.0) * 3))

            for epoch in range(stage_epochs):
                # Train
                train_metrics = self.train_epoch(
                    train_loader, epoch, stage_config["name"]
                )

                # Evaluate
                if epoch % 2 == 0:  # Evaluate every 2 epochs
                    eval_metrics = self.evaluate(val_loader, stage_config["name"])

                    logger.info(f"Stage {stage_config['name']} - Epoch {epoch}:")
                    logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
                    logger.info(f"  Eval Loss: {eval_metrics['loss']:.4f}")
                    logger.info(f"  BLEU: {eval_metrics['bleu']:.4f}")
                    logger.info(f"  Exact Match: {eval_metrics['exact_match']:.4f}")

                    # Save checkpoint if best
                    if eval_metrics["exact_match"] > best_metrics["exact_match"]:
                        best_metrics = eval_metrics
                        self.save_checkpoint(epoch, global_step, eval_metrics)

                    # Log to wandb
                    if self.config.use_wandb:
                        wandb.log(
                            {
                                "eval/loss": eval_metrics["loss"],
                                "eval/bleu": eval_metrics["bleu"],
                                "eval/exact_match": eval_metrics["exact_match"],
                                "eval/stage": stage_config["name"],
                            }
                        )

                global_step += len(train_loader)

                # Early stopping
                if global_step >= self.config.max_steps:
                    break

            if global_step >= self.config.max_steps:
                break

        logger.info("Training completed!")
        logger.info(f"Best Exact Match: {best_metrics['exact_match']:.4f}")

        return best_metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Korean-English NMT model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_nmt.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/nmt",
        help="Output directory for models",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="data/processed/sentencepiece.model",
        help="Path to SentencePiece tokenizer model",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config = TrainingConfig(**config_dict)
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.tokenizer_path = args.tokenizer_path

    # Load data
    logger.info("Loading training data")
    with open(f"{args.data_dir}/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(f"{args.data_dir}/val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    logger.info(
        f"Loaded {len(train_data)} training pairs and {len(val_data)} validation pairs"
    )

    # Create trainer and train
    trainer = NMTTrainer(config)
    best_metrics = trainer.train(train_data, val_data)

    logger.info(
        f"Training completed. Best exact match: {best_metrics['exact_match']:.4f}"
    )


if __name__ == "__main__":
    main()
