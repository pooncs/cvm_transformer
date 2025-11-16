#!/usr/bin/env python3
"""
Extended NMT Training with Learning Rate Scheduling and Regularization
Optimized for achieving 99%+ Korean-English translation accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
import time
import os
import argparse
from typing import Dict, List, Tuple
import sentencepiece as spm
import math
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("Using CPU")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerNMT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer with proper configuration
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, lengths, max_length=None):
        if max_length is None:
            max_length = lengths.max().item()

        # Ensure both tensors are on the same device
        batch_size = lengths.size(0)
        mask = (
            torch.arange(max_length, device=lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_length)
        )
        mask = mask >= lengths.unsqueeze(1)
        return mask

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Embeddings
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encoding
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)

        # Create causal mask for decoder
        tgt_length = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_length).to(
            device
        )

        # Transpose for transformer (seq_len, batch, features)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)

        # Forward through transformer
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Output projection
        output = self.fc_out(output.transpose(0, 1))
        return output


class TranslationDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.piece_to_id("<pad>")
        self.bos_token_id = tokenizer.piece_to_id("<s>")
        self.eos_token_id = tokenizer.piece_to_id("</s>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        korean = item["korean"]
        english = item["english"]

        # Tokenize
        korean_tokens = (
            [self.bos_token_id] + self.tokenizer.encode(korean) + [self.eos_token_id]
        )
        english_tokens = (
            [self.bos_token_id] + self.tokenizer.encode(english) + [self.eos_token_id]
        )

        # Truncate if necessary
        korean_tokens = korean_tokens[: self.max_length]
        english_tokens = english_tokens[: self.max_length]

        # Pad sequences
        korean_padded = korean_tokens + [self.pad_token_id] * (
            self.max_length - len(korean_tokens)
        )
        english_padded = english_tokens + [self.pad_token_id] * (
            self.max_length - len(english_tokens)
        )

        # Calculate actual lengths
        korean_length = min(len(korean_tokens), self.max_length)
        english_length = min(len(english_tokens), self.max_length)

        return {
            "korean": torch.tensor(korean_padded, dtype=torch.long),
            "english": torch.tensor(english_padded, dtype=torch.long),
            "korean_length": torch.tensor(korean_length, dtype=torch.long),
            "english_length": torch.tensor(english_length, dtype=torch.long),
        }


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scheduler,
    epoch,
    device,
    grad_accumulation_steps=4,
):
    model.train()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        korean = batch["korean"].to(device)
        english = batch["english"].to(device)
        korean_lengths = batch["korean_length"].to(device)
        english_lengths = batch["english_length"].to(device)

        # Create masks
        src_key_padding_mask = model.create_padding_mask(korean_lengths, korean.size(1))
        tgt_key_padding_mask = model.create_padding_mask(
            english_lengths, english.size(1)
        )

        # Prepare target (shifted for teacher forcing)
        tgt_input = english[:, :-1]
        tgt_output = english[:, 1:]

        # Forward pass
        output = model(
            korean,
            tgt_input,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask[:, :-1],
            memory_key_padding_mask=src_key_padding_mask,
        )

        # Calculate loss (ignore padding)
        output_flat = output.reshape(-1, output.size(-1))
        tgt_flat = tgt_output.reshape(-1)

        # Create loss mask (ignore padding tokens)
        loss_mask = (tgt_flat != 0).float()
        loss = criterion(output_flat, tgt_flat)
        loss = (loss * loss_mask).sum() / loss_mask.sum()

        # Backward pass with gradient accumulation
        loss = loss / grad_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % grad_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Update scheduler
            if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step(epoch + batch_idx / len(dataloader))

        # Calculate accuracy
        pred = output.argmax(dim=-1)
        correct = (pred == tgt_output) & (tgt_output != 0)
        total_correct += correct.sum().item()
        total_tokens += (tgt_output != 0).sum().item()
        total_loss += loss.item() * grad_accumulation_steps

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                f"Loss: {loss.item() * grad_accumulation_steps:.4f}, "
                f"Accuracy: {correct.sum().item() / (tgt_output != 0).sum().item():.4f}"
            )

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, avg_accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            korean = batch["korean"].to(device)
            english = batch["english"].to(device)
            korean_lengths = batch["korean_length"].to(device)
            english_lengths = batch["english_length"].to(device)

            # Create masks
            src_key_padding_mask = model.create_padding_mask(
                korean_lengths, korean.size(1)
            )
            tgt_key_padding_mask = model.create_padding_mask(
                english_lengths, english.size(1)
            )

            # Prepare target
            tgt_input = english[:, :-1]
            tgt_output = english[:, 1:]

            # Forward pass
            output = model(
                korean,
                tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask[:, :-1],
                memory_key_padding_mask=src_key_padding_mask,
            )

            # Calculate loss
            output_flat = output.reshape(-1, output.size(-1))
            tgt_flat = tgt_output.reshape(-1)

            loss_mask = (tgt_flat != 0).float()
            loss = criterion(output_flat, tgt_flat)
            loss = (loss * loss_mask).sum() / loss_mask.sum()

            # Calculate accuracy
            pred = output.argmax(dim=-1)
            correct = (pred == tgt_output) & (tgt_output != 0)

            total_loss += loss.item()
            total_correct += correct.sum().item()
            total_tokens += (tgt_output != 0).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, avg_accuracy


def main():
    parser = argparse.ArgumentParser(description="Extended NMT Training")
    parser.add_argument(
        "--data-file", default="data/processed_large_simple/train_data.json"
    )
    parser.add_argument(
        "--val-file", default="data/processed_large_simple/val_data.json"
    )
    parser.add_argument(
        "--tokenizer-model",
        default="data/processed_large_simple/sentencepiece_large.model",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--save-dir", default="models/extended")
    parser.add_argument("--quick-validation", action="store_true")
    args = parser.parse_args()

    # Initialize wandb for experiment tracking (optional)
    try:
        import wandb

        wandb.init(project="korean-english-nmt-extended", config=args)
        use_wandb = True
    except:
        print("Wandb not available, running without tracking")
        use_wandb = False

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_model)
    vocab_size = tokenizer.get_piece_size()
    print(f"Vocabulary size: {vocab_size}")

    # Create datasets
    train_dataset = TranslationDataset(args.data_file, tokenizer, args.max_length)
    val_dataset = TranslationDataset(args.val_file, tokenizer, args.max_length)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    model = TransformerNMT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_length,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Training loop
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    print("Starting extended training...")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_dataloader, optimizer, criterion, scheduler, epoch, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_dataloader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Log to wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save model
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_accuracy": val_acc,
                    "args": args,
                },
                os.path.join(args.save_dir, "best_model.pt"),
            )

            print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Quick validation mode
        if args.quick_validation and epoch >= 4:
            print("Quick validation completed")
            break

    # Save final model
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "val_accuracy": val_acc,
            "args": args,
        },
        os.path.join(args.save_dir, "final_model.pt"),
    )

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
