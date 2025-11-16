#!/usr/bin/env python3
"""
Minimal NMT training script to bypass mask issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTranslationDataset(Dataset):
    """Simple dataset for translation pairs."""
    
    def __init__(self, data_file, tokenizer_model, max_length=64):
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)
        self.max_length = max_length
        self.pad_id = self.tokenizer.pad_id()
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize source (Korean)
        src_tokens = self.tokenizer.encode_as_ids(item['korean'])
        src_tokens = src_tokens[:self.max_length-2]
        src_tokens = [self.tokenizer.bos_id()] + src_tokens + [self.tokenizer.eos_id()]
        
        # Tokenize target (English)
        tgt_tokens = self.tokenizer.encode_as_ids(item['english'])
        tgt_tokens = tgt_tokens[:self.max_length-2]
        tgt_tokens = [self.tokenizer.bos_id()] + tgt_tokens + [self.tokenizer.eos_id()]
        
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens, dtype=torch.long)
        }

def collate_fn(batch):
    """Collate function to pad sequences in a batch."""
    max_src_len = max(len(item['src']) for item in batch)
    max_tgt_len = max(len(item['tgt']) for item in batch)
    
    src_padded = []
    tgt_padded = []
    
    for item in batch:
        src = item['src']
        tgt = item['tgt']
        
        # Pad source
        src_pad = torch.cat([src, torch.zeros(max_src_len - len(src), dtype=torch.long)])
        src_padded.append(src_pad)
        
        # Pad target
        tgt_pad = torch.cat([tgt, torch.zeros(max_tgt_len - len(tgt), dtype=torch.long)])
        tgt_padded.append(tgt_pad)
    
    return {
        'src': torch.stack(src_padded),
        'tgt': torch.stack(tgt_padded)
    }

# Simple encoder-decoder model without complex attention
class SimpleNMT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*4),
            num_layers=n_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*4),
            num_layers=n_layers
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt):
        # Transpose for transformer (seq_len, batch_size, d_model)
        src_embed = self.embedding(src).transpose(0, 1) * (self.d_model ** 0.5)
        tgt_embed = self.embedding(tgt).transpose(0, 1) * (self.d_model ** 0.5)
        
        # Encode
        memory = self.encoder(src_embed)
        
        # Decode (teacher forcing)
        output = self.decoder(tgt_embed, memory)
        
        # Project to vocab and transpose back (batch_size, seq_len, vocab_size)
        return self.output_proj(output).transpose(0, 1)

def train_minimal():
    """Minimal training function."""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 10
    max_length = 64
    
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file='data/processed/sentencepiece.model')
    vocab_size = tokenizer.vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Load data
    train_dataset = SimpleTranslationDataset(
        'data/processed/train.json', 
        'data/processed/sentencepiece.model',
        max_length
    )
    val_dataset = SimpleTranslationDataset(
        'data/processed/val.json', 
        'data/processed/sentencepiece.model',
        max_length
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = SimpleNMT(vocab_size).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad_id = 0
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # Create target for loss (shifted by one position)
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_output = tgt[:, 1:]    # Remove first token
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src, tgt_input)
            
            # Calculate loss
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_steps
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/nmt_model/minimal_model.pt')
            logger.info(f"Saved best model with validation loss: {avg_val_loss:.4f}")
    
    logger.info("Training completed!")
    return model

if __name__ == "__main__":
    model = train_minimal()