#!/usr/bin/env python3
"""
Simple training script for Korean-English NMT
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import pandas as pd
import json
import time
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTranslationDataset(Dataset):
    def __init__(self, data_file: str, tokenizer_path: str, max_length: int = 128):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.max_length = max_length
        self.pad_id = self.tokenizer.pad_id()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize source (Korean)
        src_tokens = self.tokenizer.encode_as_ids(item['korean'])
        src_tokens = src_tokens[:self.max_length-2]  # Make room for BOS/EOS
        src_tokens = [self.tokenizer.bos_id()] + src_tokens + [self.tokenizer.eos_id()]
        
        # Tokenize target (English)
        tgt_tokens = self.tokenizer.encode_as_ids(item['english'])
        tgt_tokens = tgt_tokens[:self.max_length-2]
        tgt_tokens = [self.tokenizer.bos_id()] + tgt_tokens + [self.tokenizer.eos_id()]
        
        # Create tensors
        src_tensor = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)
        
        return {
            'src': src_tensor,
            'tgt': tgt_tensor
        }

def collate_fn(batch):
    """Collate function to pad sequences in a batch."""
    # Find max length in this batch
    max_src_len = max(len(item['src']) for item in batch)
    max_tgt_len = max(len(item['tgt']) for item in batch)
    
    # Pad sequences
    padded_batch = []
    for item in batch:
        src = item['src']
        tgt = item['tgt']
        
        # Pad source
        src_padded = torch.cat([src, torch.zeros(max_src_len - len(src), dtype=torch.long)])
        
        # Pad target
        tgt_padded = torch.cat([tgt, torch.zeros(max_tgt_len - len(tgt), dtype=torch.long)])
        
        padded_batch.append({
            'src': src_padded,
            'tgt': tgt_padded
        })
    
    # Stack into batch tensors
    return {
        'src': torch.stack([item['src'] for item in padded_batch]),
        'tgt': torch.stack([item['tgt'] for item in padded_batch])
    }

def train_simple():
    """Simple training function."""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 10
    max_length = 64
    
    logger.info(f"Using device: {device}")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    from src.models.nmt_transformer import create_nmt_transformer
    
    vocab_size = 600  # Our tokenizer vocab size
    model_config = {
        'src_vocab_size': vocab_size,
        'tgt_vocab_size': vocab_size,
        'd_model': 256,  # Smaller model for quick training
        'n_heads': 4,
        'n_encoder_layers': 4,
        'n_decoder_layers': 4,
        'd_ff': 1024,
        'max_len': max_length,
        'dropout': 0.1,
        'use_flash_attention': False  # Disable for CPU
    }
    model = create_nmt_transformer(model_config).to(device)
    
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
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
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
            torch.save(model.state_dict(), 'models/nmt_model/best_model.pt')
            logger.info(f"Saved best model with validation loss: {avg_val_loss:.4f}")
    
    logger.info("Training completed!")
    return model

if __name__ == "__main__":
    os.makedirs('models/nmt_model', exist_ok=True)
    model = train_simple()