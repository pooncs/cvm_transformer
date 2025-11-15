#!/usr/bin/env python3
"""
Simplified Advanced CVM Transformer Training with Learning Rate Scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import json
import time
import math

class SimpleTranslationDataset(Dataset):
    """Simple dataset for advanced training"""
    
    def __init__(self, src_file: str, tgt_file: str, tokenizer_model: str, max_length: int = 128):
        self.max_length = max_length
        
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
        
        # Tokenize
        src_tokens = self.tokenizer.encode(src_text, out_type=int)
        tgt_tokens = self.tokenizer.encode(tgt_text, out_type=int)
        
        # Add special tokens
        src_tokens = [1] + src_tokens + [2]  # <sos> + tokens + <eos>
        tgt_tokens = [2] + tgt_tokens + [3]  # <sos> + tokens + <eos>
        
        # Pad or truncate
        src_tokens = self._pad_or_truncate(src_tokens)
        tgt_tokens = self._pad_or_truncate(tgt_tokens)
        
        return {
            'src': torch.tensor(src_tokens),
            'tgt': torch.tensor(tgt_tokens),
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
    def _pad_or_truncate(self, tokens):
        if len(tokens) > self.max_length:
            return tokens[:self.max_length-1] + [tokens[-1]]  # Keep EOS
        else:
            return tokens + [0] * (self.max_length - len(tokens))

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

def train_epoch_advanced(model, dataloader, optimizer, criterion, device, scheduler, grad_clip=1.0):
    """Enhanced training with gradient clipping and label smoothing"""
    
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 5 == 0:
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
    
    return total_loss / len(dataloader)

def validate_epoch_advanced(model, dataloader, criterion, device):
    """Enhanced validation"""
    
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # Forward pass
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            mask = (tgt[:, 1:] != 0)  # Ignore padding
            correct = (predictions == tgt[:, 1:]) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return total_loss / len(dataloader), accuracy

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
        'batch_size': 16,
        'base_lr': 5e-4,    # Higher base learning rate
        'num_epochs': 50,   # Moderate epochs
        'warmup_steps': 200,
        'grad_clip': 1.0,
        'label_smoothing': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Advanced training configuration: {config}")
    
    # Create datasets
    train_dataset = SimpleTranslationDataset(
        'data/kr_diverse.txt', 'data/en_diverse.txt', 'kr_en_diverse.model',
        max_length=config['max_length']
    )
    val_dataset = SimpleTranslationDataset(
        'data/kr_diverse.txt', 'data/en_diverse.txt', 'kr_en_diverse.model',
        max_length=config['max_length']
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
    
    # Create model (reuse the working enhanced model)
    from train_enhanced import EnhancedTransformer
    model = EnhancedTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
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
    best_val_acc = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    print("Starting advanced training with learning rate scheduling...")
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch_advanced(model, train_loader, optimizer, criterion, 
                                        config['device'], scheduler, config['grad_clip'])
        
        # Validate
        val_loss, val_acc = validate_epoch_advanced(model, val_loader, criterion, config['device'])
        
        epoch_time = time.time() - start_time
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['lr'].append(current_lr)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config
            }, 'best_advanced_simple_model.pth')
            print("Saved best model!")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'config': config
            }, f'checkpoint_advanced_simple_epoch_{epoch+1}.pth')
    
    # Save final history
    with open('training_history_advanced_simple.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Advanced training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()