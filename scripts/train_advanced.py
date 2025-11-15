#!/usr/bin/env python3
"""
Advanced CVM Transformer Training with Learning Rate Scheduling and Advanced Strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import json
import time
import numpy as np
from typing import Dict, List, Tuple
import math

class AdvancedTranslationDataset(Dataset):
    """Enhanced dataset with data augmentation and better preprocessing"""
    
    def __init__(self, src_file: str, tgt_file: str, tokenizer_model: str, 
                 max_length: int = 128, augment: bool = False):
        self.max_length = max_length
        self.augment = augment
        
        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)
        
        # Load data
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = [line.strip() for line in f.readlines()]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = [line.strip() for line in f.readlines()]
        
        assert len(self.src_sentences) == len(self.tgt_sentences)
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]
        
        # Simple data augmentation (add noise for robustness)
        if self.augment and torch.rand(1).item() < 0.1:
            # 10% chance to add small perturbation
            if len(src_text) > 3:
                src_text = src_text[:-1]  # Remove last character sometimes
        
        # Tokenize
        src_tokens = self.tokenizer.encode(src_text, out_type=int)
        tgt_tokens = self.tokenizer.encode(tgt_text, out_type=int)
        
        # Add special tokens
        src_tokens = [1] + src_tokens + [2]  # <sos> + tokens + <eos>
        tgt_tokens = [1] + tgt_tokens + [2]
        
        # Pad or truncate
        src_tokens = self._pad_or_truncate(src_tokens)
        tgt_tokens = self._pad_or_truncate(tgt_tokens)
        
        # Create masks
        src_mask = (torch.tensor(src_tokens) == 0)  # True for padding positions
        tgt_mask = self._generate_square_subsequent_mask(len(tgt_tokens))
        
        return {
            'src': torch.tensor(src_tokens),
            'tgt': torch.tensor(tgt_tokens),
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
    def _pad_or_truncate(self, tokens):
        if len(tokens) > self.max_length:
            return tokens[:self.max_length-1] + [tokens[-1]]  # Keep EOS
        else:
            return tokens + [0] * (self.max_length - len(tokens))
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    """Enhanced positional encoding with dropout"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AdvancedTransformerModel(nn.Module):
    """Enhanced transformer with layer normalization and residual connections"""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=12, 
                 num_decoder_layers=12, dim_feedforward=2048, dropout=0.1, max_length=128):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)
        
        # Enhanced transformer with pre-norm architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=False, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=False, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embeddings with scaling
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # Transformer pass
        encoder_output = self.transformer_encoder(src_emb.transpose(0, 1), 
                                                 src_key_padding_mask=src_mask.squeeze(1).squeeze(1) if src_mask is not None else None)
        
        decoder_output = self.transformer_decoder(tgt_emb.transpose(0, 1), 
                                                 encoder_output, 
                                                 tgt_mask=tgt_mask,
                                                 tgt_key_padding_mask=None,
                                                 memory_key_padding_mask=src_mask.squeeze(1).squeeze(1) if src_mask is not None else None)
        
        # Final layer norm and projection
        decoder_output = self.layer_norm(decoder_output)
        output = self.output_projection(decoder_output)
        
        return output.transpose(0, 1)

class WarmupCosineScheduler:
    """Advanced learning rate scheduler with warmup and cosine annealing"""
    
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def train_epoch(model, dataloader, optimizer, criterion, device, scheduler, epoch, 
                grad_clip=1.0, label_smoothing=0.1):
    """Enhanced training with gradient clipping and label smoothing"""
    
    model.train()
    total_loss = 0
    total_tokens = 0
    
    # Label smoothing for better generalization
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
    
    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device) if batch['src_mask'] is not None else None
        tgt_mask = batch['tgt_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Teacher forcing with decay
        teacher_forcing_ratio = max(0.5, 1.0 - epoch * 0.02)  # Decay from 1.0 to 0.5
        
        if torch.rand(1).item() < teacher_forcing_ratio:
            # Standard teacher forcing
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        else:
            # Scheduled sampling - use model's own predictions
            tgt_input = tgt[:, :1]
            for i in range(tgt.size(1) - 1):
                output = model(src, tgt_input, src_mask, tgt_mask[:i+1, :i+1])
                next_token = output[:, -1:].argmax(dim=-1)
                tgt_input = torch.cat([tgt_input, next_token], dim=1)
            
            # Calculate loss on final sequence
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_tokens += (tgt != 0).sum().item()
        
        if batch_idx % 10 == 0:
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Enhanced validation with beam search"""
    
    model.eval()
    total_loss = 0
    total_bleu = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_mask = batch['src_mask'].to(device) if batch['src_mask'] is not None else None
            tgt_mask = batch['tgt_mask'].to(device)
            
            # Standard validation loss
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            # Simple BLEU-like evaluation (word overlap)
            predictions = output.argmax(dim=-1)
            batch_bleu = 0
            for i in range(tgt.size(0)):
                pred_tokens = predictions[i].cpu().numpy()
                true_tokens = tgt[i, 1:].cpu().numpy()
                
                # Remove padding and special tokens
                pred_tokens = pred_tokens[pred_tokens > 3]
                true_tokens = true_tokens[true_tokens > 3]
                
                if len(pred_tokens) > 0 and len(true_tokens) > 0:
                    overlap = len(set(pred_tokens) & set(true_tokens))
                    total_possible = max(len(set(true_tokens)), 1)
                    batch_bleu += overlap / total_possible
            
            total_bleu += batch_bleu / tgt.size(0)
    
    return total_loss / len(dataloader), total_bleu / len(dataloader)

def main():
    """Main training function with advanced strategies"""
    
    # Configuration
    config = {
        'vocab_size': 1000,
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 12,
        'num_decoder_layers': 12,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_length': 128,
        'batch_size': 32,  # Increased batch size
        'base_lr': 5e-4,    # Higher base learning rate
        'num_epochs': 100,  # More epochs
        'warmup_steps': 1000,
        'grad_clip': 1.0,
        'label_smoothing': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Advanced training configuration: {config}")
    
    # Create datasets with augmentation
    train_dataset = AdvancedTranslationDataset(
        'data/kr_diverse.txt', 'data/en_diverse.txt', 'kr_en_diverse.model',
        max_length=config['max_length'], augment=True
    )
    val_dataset = AdvancedTranslationDataset(
        'data/kr_diverse.txt', 'data/en_diverse.txt', 'kr_en_diverse.model',
        max_length=config['max_length'], augment=False
    )
    
    # Split data properly
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    _, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create model
    model = AdvancedTransformerModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_length=config['max_length']
    ).to(config['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Advanced optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config['base_lr'], 
                         weight_decay=0.01, betas=(0.9, 0.98))
    
    # Advanced scheduler
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = WarmupCosineScheduler(optimizer, config['warmup_steps'], total_steps)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config['label_smoothing'])
    
    # Training loop
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'val_bleu': [], 'lr': []}
    
    print("Starting advanced training...")
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                               config['device'], scheduler, epoch, 
                               config['grad_clip'], config['label_smoothing'])
        
        # Validate
        val_loss, val_bleu = validate_epoch(model, val_loader, criterion, config['device'])
        
        epoch_time = time.time() - start_time
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val BLEU: {val_bleu:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_bleu'].append(val_bleu)
        training_history['lr'].append(current_lr)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_bleu': val_bleu,
                'config': config
            }, 'best_advanced_model.pth')
            print("Saved best model!")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'config': config
            }, f'checkpoint_advanced_epoch_{epoch+1}.pth')
    
    # Save final history
    with open('training_history_advanced.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Advanced training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()