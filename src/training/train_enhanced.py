"""
Enhanced training script with SentencePiece BPE tokenization and diverse dataset.
This addresses the tokenization issues identified in the previous analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import os
import json
import time
from pathlib import Path

# Import the existing SentencePiece tokenizer
import sys

sys.path.append(".")
from cvm_translator.sp_tokenizer import SPTokenizer


class TranslationDataset(Dataset):
    """Dataset class using SentencePiece tokenization."""

    def __init__(self, src_file, tgt_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read source and target sentences
        with open(src_file, "r", encoding="utf-8") as f:
            self.src_sentences = [line.strip() for line in f if line.strip()]

        with open(tgt_file, "r", encoding="utf-8") as f:
            self.tgt_sentences = [line.strip() for line in f if line.strip()]

        assert len(self.src_sentences) == len(
            self.tgt_sentences
        ), "Source and target files must have the same number of lines"

        print(f"Loaded {len(self.src_sentences)} sentence pairs")

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        # Tokenize source (Korean)
        src_tokens = self.tokenizer.encode(src_text)

        # Tokenize target (English) - add BOS and EOS
        tgt_tokens = [2] + self.tokenizer.encode(tgt_text) + [3]  # bos_id=2, eos_id=3

        # Truncate if necessary
        if len(src_tokens) > self.max_length:
            src_tokens = src_tokens[: self.max_length]

        if len(tgt_tokens) > self.max_length:
            tgt_tokens = tgt_tokens[: self.max_length]

        # Create tensors
        src_tensor = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)

        return {
            "src": src_tensor,
            "tgt": tgt_tensor,
            "src_len": len(src_tensor),
            "tgt_len": len(tgt_tensor),
        }


def collate_fn(batch):
    """Custom collate function for padding."""
    src_tensors = [item["src"] for item in batch]
    tgt_tensors = [item["tgt"] for item in batch]

    # Pad sequences
    src_padded = nn.utils.rnn.pad_sequence(
        src_tensors, batch_first=True, padding_value=0
    )
    tgt_padded = nn.utils.rnn.pad_sequence(
        tgt_tensors, batch_first=True, padding_value=0
    )

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_lengths": torch.tensor([item["src_len"] for item in batch]),
        "tgt_lengths": torch.tensor([item["tgt_len"] for item in batch]),
    }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class EnhancedTransformer(nn.Module):
    """Enhanced transformer model with 12 layers and improved architecture."""

    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=12,
        num_decoder_layers=12,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Enhanced transformer with 12 layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):

        # Embeddings with scaling
        src_emb = self.embedding(src) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float)
        )
        tgt_emb = self.embedding(tgt) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float)
        )

        # Add positional encoding
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)

        # Apply dropout
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        # Create causal mask for decoder if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Transformer forward pass
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # Create masks
        src_padding_mask = src == 0
        tgt_padding_mask = tgt == 0

        # Prepare decoder input (remove last token) and target (remove first token)
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        # Forward pass
        optimizer.zero_grad()
        output = model(
            src,
            tgt_input,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask[:, :-1],
        )

        # Calculate loss
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            # Create masks
            src_padding_mask = src == 0
            tgt_padding_mask = tgt == 0

            # Prepare decoder input and target
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            # Forward pass
            output = model(
                src,
                tgt_input,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask[:, :-1],
            )

            # Calculate loss
            loss = criterion(
                output.reshape(-1, output.size(-1)), tgt_target.reshape(-1)
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    """Main training function."""

    # Configuration
    config = {
        "vocab_size": 1000,  # Reduced to reasonable size for small dataset
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "num_epochs": 50,
        "max_length": 128,
        "train_split": 0.8,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    print(f"Training configuration: {config}")

    # Create SentencePiece model if it doesn't exist
    if not os.path.exists("kr_en_diverse.model"):
        print("Training SentencePiece model...")
        spm.SentencePieceTrainer.train(
            input=["data/kr_diverse.txt", "data/en_diverse.txt"],
            model_prefix="kr_en_diverse",
            vocab_size=config["vocab_size"],
            character_coverage=0.9995,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        print("SentencePiece model training completed!")

    # Initialize tokenizer
    tokenizer = SPTokenizer("kr_en_diverse.model")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")

    # Create dataset
    dataset = TranslationDataset(
        "data/kr_diverse.txt",
        "data/en_diverse.txt",
        tokenizer,
        max_length=config["max_length"],
    )

    # Split into train and validation
    train_size = int(config["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(
        f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )

    # Initialize model
    model = EnhancedTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
    ).to(config["device"])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training loop
    best_val_loss = float("inf")
    training_history = []

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # Train
        train_loss = train_epoch(
            model, train_dataloader, optimizer, criterion, config["device"]
        )

        # Validate
        val_loss = validate(model, val_dataloader, criterion, config["device"])

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = "best_enhanced_model.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                best_model_path,
            )
            print("Saved best model!")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_history": training_history,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save training history
    with open("training_history_enhanced.json", "w") as f:
        json.dump(training_history, f, indent=2)

    print("Training history saved to training_history_enhanced.json")


if __name__ == "__main__":
    main()
