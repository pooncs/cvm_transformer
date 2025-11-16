"""
Advanced Korean-English NMT training script with professional tokenization and attention mechanisms.
Implements Transformer Big architecture with proper regularization and optimization.
"""

import os
import json
import math
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import sentencepiece as spm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for Transformer Big model."""

    d_model: int = 1024  # Model dimension
    n_heads: int = 16  # Number of attention heads
    n_layers: int = 6  # Number of encoder/decoder layers
    d_ff: int = 4096  # Feed-forward dimension
    dropout: float = 0.1
    max_seq_length: int = 128
    vocab_size: int = 3000  # Our trained tokenizer vocab size
    pad_token_id: int = 0  # Using 0 as pad token since tokenizer.pad_id() is -1
    sos_token_id: int = 1  # BOS ID from tokenizer
    eos_token_id: int = 2  # EOS ID from tokenizer


class TranslationDataset(Dataset):
    """Dataset for Korean-English translation."""

    def __init__(self, data_file: str, tokenizer_model: str, max_length: int = 128):
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)

        # Load data
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} sentence pairs from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize source (Korean)
        src_tokens = self.tokenizer.encode(item["src"], out_type=int)
        # Tokenize target (English)
        tgt_tokens = self.tokenizer.encode(item["tgt"], out_type=int)

        # Bounds checking
        vocab_size = self.tokenizer.get_piece_size()
        if any(t >= vocab_size for t in src_tokens):
            print(f"WARNING: Source tokens exceed vocab size in sample {idx}")
            src_tokens = [min(t, vocab_size - 1) for t in src_tokens]

        if any(t >= vocab_size for t in tgt_tokens):
            print(f"WARNING: Target tokens exceed vocab size in sample {idx}")
            tgt_tokens = [min(t, vocab_size - 1) for t in tgt_tokens]

        # Truncate if necessary
        src_tokens = src_tokens[: self.max_length - 2]  # Reserve space for SOS/EOS
        tgt_tokens = tgt_tokens[: self.max_length - 2]

        # Add special tokens
        src_tokens = [self.tokenizer.bos_id()] + src_tokens + [self.tokenizer.eos_id()]
        tgt_tokens = [self.tokenizer.bos_id()] + tgt_tokens + [self.tokenizer.eos_id()]

        # Pad sequences - use 0 as pad token since tokenizer.pad_id() is -1
        src_length = len(src_tokens)
        tgt_length = len(tgt_tokens)

        src_tokens = src_tokens + [0] * (self.max_length - src_length)
        tgt_tokens = tgt_tokens + [0] * (self.max_length - tgt_length)

        # Final bounds check
        if any(t >= vocab_size for t in src_tokens + tgt_tokens):
            print(f"ERROR: Final tokens exceed vocab size in sample {idx}")
            print(
                f"Max token: {max(src_tokens + tgt_tokens)}, Vocab size: {vocab_size}"
            )

        return {
            "src_tokens": torch.tensor(src_tokens, dtype=torch.long),
            "tgt_tokens": torch.tensor(tgt_tokens, dtype=torch.long),
            "src_length": torch.tensor(src_length, dtype=torch.long),
            "tgt_length": torch.tensor(tgt_length, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    src_tokens = torch.stack([item["src_tokens"] for item in batch])
    tgt_tokens = torch.stack([item["tgt_tokens"] for item in batch])
    src_lengths = torch.stack([item["src_length"] for item in batch])
    tgt_lengths = torch.stack([item["tgt_length"] for item in batch])

    return {
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "src_lengths": src_lengths,
        "tgt_lengths": tgt_lengths,
    }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

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
        return x + self.pe[: x.size(0), :]


class TransformerNMT(nn.Module):
    """Advanced Transformer model for NMT."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, config.n_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.n_layers)

        # Output layer
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(
        self, lengths: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        """Create padding mask for attention."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_length, device=lengths.device).expand(
            batch_size, max_length
        ) >= lengths.unsqueeze(1)
        return mask.bool()  # Ensure boolean mask

    def create_look_ahead_mask(self, size: int) -> torch.Tensor:
        """Create look-ahead mask for decoder self-attention."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src_tokens: Source token IDs [batch_size, src_len]
            tgt_tokens: Target token IDs [batch_size, tgt_len]
            src_lengths: Source sequence lengths [batch_size]
            tgt_lengths: Target sequence lengths [batch_size]

        Returns:
            logits: Output logits [batch_size, tgt_len, vocab_size]
        """
        batch_size, src_len = src_tokens.size()
        _, tgt_len = tgt_tokens.size()

        # Bounds checking for token IDs
        if src_tokens.max() >= self.config.vocab_size:
            print(
                f"ERROR: Source tokens exceed vocab size! Max: {src_tokens.max()}, Vocab: {self.config.vocab_size}"
            )
            src_tokens = torch.clamp(src_tokens, max=self.config.vocab_size - 1)

        if tgt_tokens.max() >= self.config.vocab_size:
            print(
                f"ERROR: Target tokens exceed vocab size! Max: {tgt_tokens.max()}, Vocab: {self.config.vocab_size}"
            )
            tgt_tokens = torch.clamp(tgt_tokens, max=self.config.vocab_size - 1)

        # Create masks
        src_padding_mask = self.create_padding_mask(src_lengths, src_len)
        tgt_padding_mask = self.create_padding_mask(tgt_lengths, tgt_len)
        look_ahead_mask = self.create_look_ahead_mask(tgt_len).to(src_tokens.device)

        # Combine padding and look-ahead masks for decoder self-attention
        tgt_mask = look_ahead_mask | tgt_padding_mask.unsqueeze(1)

        # Embeddings + positional encoding
        src_embedded = self.pos_encoding(self.src_embedding(src_tokens))
        tgt_embedded = self.pos_encoding(self.tgt_embedding(tgt_tokens))

        # Transpose for transformer (seq_len, batch_size, d_model)
        src_embedded = src_embedded.transpose(0, 1)
        tgt_embedded = tgt_embedded.transpose(0, 1)

        # Encoder
        memory = self.encoder(src_embedded, src_key_padding_mask=~src_padding_mask)

        # Decoder
        decoder_output = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=~tgt_padding_mask,
            memory_key_padding_mask=~src_padding_mask,
        )

        # Transpose back (batch_size, seq_len, d_model)
        decoder_output = decoder_output.transpose(0, 1)

        # Output projection
        logits = self.output_projection(decoder_output)

        return logits

    def generate(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        max_length: int = 128,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate translations using greedy decoding.

        Args:
            src_tokens: Source token IDs [batch_size, src_len]
            src_lengths: Source sequence lengths [batch_size]
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            generated_tokens: Generated token IDs [batch_size, max_length]
        """
        self.eval()
        batch_size = src_tokens.size(0)
        device = src_tokens.device

        # Create source mask
        src_padding_mask = self.create_padding_mask(src_lengths, src_tokens.size(1))

        # Encode source
        with torch.no_grad():
            src_embedded = self.pos_encoding(self.src_embedding(src_tokens))
            memory = self.encoder(src_embedded, src_key_padding_mask=src_padding_mask)

        # Initialize target sequence with SOS token
        generated_tokens = torch.full(
            (batch_size, 1), self.config.sos_token_id, dtype=torch.long, device=device
        )

        # Generate tokens one by one
        for _ in range(max_length - 1):
            with torch.no_grad():
                # Create target mask
                tgt_len = generated_tokens.size(1)
                look_ahead_mask = self.create_look_ahead_mask(tgt_len).to(device)

                # Target embeddings
                tgt_embedded = self.pos_encoding(self.tgt_embedding(generated_tokens))

                # Decode
                decoder_output = self.decoder(
                    tgt_embedded,
                    memory,
                    tgt_mask=look_ahead_mask,
                    memory_key_padding_mask=src_padding_mask,
                )

                # Get next token logits
                next_token_logits = self.output_projection(decoder_output[:, -1, :])

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Sample next token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

                # Check for EOS token
                if torch.all(next_token == self.config.eos_token_id):
                    break

        return generated_tokens


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Move to device
        src_tokens = batch["src_tokens"].to(device)
        tgt_tokens = batch["tgt_tokens"].to(device)
        src_lengths = batch["src_lengths"].to(device)
        tgt_lengths = batch["tgt_lengths"].to(device)

        # Prepare input and target for decoder
        # Input: exclude last token, Target: exclude first token
        decoder_input = tgt_tokens[:, :-1]
        decoder_target = tgt_tokens[:, 1:]

        # Adjust lengths
        decoder_lengths = tgt_lengths - 1

        # Forward pass
        logits = model(src_tokens, decoder_input, src_lengths, decoder_lengths)

        # Calculate loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / num_batches


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            src_tokens = batch["src_tokens"].to(device)
            tgt_tokens = batch["tgt_tokens"].to(device)
            src_lengths = batch["src_lengths"].to(device)
            tgt_lengths = batch["tgt_lengths"].to(device)

            # Prepare input and target for decoder
            decoder_input = tgt_tokens[:, :-1]
            decoder_target = tgt_tokens[:, 1:]
            decoder_lengths = tgt_lengths - 1

            # Forward pass
            logits = model(src_tokens, decoder_input, src_lengths, decoder_lengths)

            # Calculate loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    """Main training function."""
    # Configuration
    config = TransformerConfig()

    # Training parameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0001
    warmup_steps = 4000

    # Data paths
    train_file = "data/processed_large_simple/train.json"
    val_file = "data/processed_large_simple/val.json"
    tokenizer_model = "data/processed_large_simple/sentencepiece_large.model"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = TranslationDataset(
        train_file, tokenizer_model, config.max_seq_length
    )
    val_dataset = TranslationDataset(val_file, tokenizer_model, config.max_seq_length)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Create model
    model = TransformerNMT(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=config.pad_token_id, label_smoothing=0.1
    )

    # Optimizer with warmup
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return (warmup_steps**0.5) * (step**-0.5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    logger.info("Starting training...")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_dataloader, optimizer, criterion, device, epoch
        )
        train_losses.append(train_loss)

        # Evaluate
        val_loss = evaluate(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "config": config,
                    "best_val_loss": best_val_loss,
                },
                "best_nmt_model.pt",
            )
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "config": config,
                    "best_val_loss": best_val_loss,
                },
                f"checkpoint_epoch_{epoch}.pt",
            )

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
