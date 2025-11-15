"""
Simplified robust training to test the core improvements.
Focuses on preventing model collapse with basic techniques.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import os
import json
import time
import sys
sys.path.append('.')

from cvm_translator.sp_tokenizer import SPTokenizer

class SimpleTranslationDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, src_file, tgt_file, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read source and target sentences
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = [line.strip() for line in f if line.strip()]
        
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = [line.strip() for line in f if line.strip()]
        
        # Use first 80 sentences for training
        self.src_sentences = self.src_sentences[:80]
        self.tgt_sentences = self.tgt_sentences[:80]
        
        assert len(self.src_sentences) == len(self.tgt_sentences), \
            "Source and target files must have the same number of lines"
        
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
            src_tokens = src_tokens[:self.max_length]
        
        if len(tgt_tokens) > self.max_length:
            tgt_tokens = tgt_tokens[:self.max_length]
        
        # Create tensors
        src_tensor = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)
        
        return {
            'src': src_tensor,
            'tgt': tgt_tensor,
            'src_len': len(src_tensor),
            'tgt_len': len(tgt_tensor)
        }

def collate_fn(batch):
    """Custom collate function for padding."""
    src_tensors = [item['src'] for item in batch]
    tgt_tensors = [item['tgt'] for item in batch]
    
    # Pad sequences
    src_padded = nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_tensors, batch_first=True, padding_value=0)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lengths': torch.tensor([item['src_len'] for item in batch]),
        'tgt_lengths': torch.tensor([item['tgt_len'] for item in batch])
    }

class SimpleTransformer(nn.Module):
    """Simple transformer that prevents collapse."""
    
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, 
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Simple transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize with small weights to prevent initial bias
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small random weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(self, src, tgt):
        # Embeddings
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        
        # Create causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(output)

def train_simple(model, dataloader, optimizer, criterion, device, epoch):
    """Simple training with teacher forcing."""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        # Prepare decoder input (remove last token) and target (remove first token)
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # Calculate loss
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))
        
        # Add penalty for predicting only EOS or PAD tokens (prevents collapse)
        predictions = output.argmax(dim=-1)
        eos_pad_ratio = ((predictions == 0) | (predictions == 3)).float().mean()
        if eos_pad_ratio > 0.7:  # If more than 70% are PAD/EOS tokens
            loss = loss * (1.0 + eos_pad_ratio)  # Increase loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, EOS/PAD ratio: {eos_pad_ratio:.3f}')
    
    return total_loss / len(dataloader)

def validate_simple(model, dataloader, criterion, device):
    """Simple validation."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # Prepare decoder input and target
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Calculate loss
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def test_translation_simple(model, tokenizer, device):
    """Test translation with the simple model."""
    model.eval()
    
    test_pairs = [
        ("안녕하세요", "Hello"),
        ("감사합니다", "Thank you"),
        ("오늘 날씨가 정말 좋네요", "The weather is really nice today"),
    ]
    
    print("\nTranslation Test:")
    print("-" * 30)
    
    with torch.no_grad():
        for korean, expected in test_pairs:
            # Tokenize input
            src_tokens = tokenizer.encode(korean)
            src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
            
            # Generate translation
            tgt_tokens = [2]  # BOS
            
            for _ in range(20):  # Max 20 tokens
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
                output = model(src_tensor, tgt_tensor)
                
                # Get next token
                next_token = output[0, -1, :].argmax().item()
                tgt_tokens.append(next_token)
                
                if next_token == 3:  # EOS
                    break
            
            # Decode
            english_tokens = tgt_tokens[1:-1]  # Remove BOS and EOS
            translation = tokenizer.decode(english_tokens)
            
            print(f"Korean: {korean}")
            print(f"Expected: {expected}")
            print(f"Translation: {translation}")
            print(f"Tokens: {english_tokens}")
            print()

def main():
    """Main simple training function."""
    
    # Configuration
    config = {
        'vocab_size': 1000,
        'd_model': 256,
        'nhead': 4,
        'num_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'batch_size': 8,
        'learning_rate': 1e-3,
        'num_epochs': 20,
        'max_length': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Simple training configuration: {config}")
    
    # Initialize tokenizer
    tokenizer = SPTokenizer("kr_en_diverse.model")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")
    
    # Create dataset
    dataset = SimpleTranslationDataset(
        "data/kr_diverse.txt", 
        "data/en_diverse.txt", 
        tokenizer, 
        max_length=config['max_length']
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = SimpleTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(config['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_loss = float('inf')
    training_history = []
    
    print("\nStarting simple training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_simple(model, dataloader, optimizer, criterion, config['device'], epoch)
        
        # Validate
        val_loss = validate_simple(model, dataloader, criterion, config['device'])
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'tokenizer_model': 'kr_en_diverse.model'
            }, 'simple_best_model.pth')
            print("Saved best model!")
        
        # Test translation every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_translation_simple(model, tokenizer, config['device'])
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {best_loss:.4f}")
    
    # Final test
    print("\nFinal translation test:")
    test_translation_simple(model, tokenizer, config['device'])
    
    # Save training history
    with open('simple_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("Training history saved to simple_training_history.json")

if __name__ == "__main__":
    main()