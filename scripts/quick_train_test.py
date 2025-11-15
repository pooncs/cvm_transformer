"""
Quick training test with the improved tokenization system.
This runs a shorter training to validate the improvements.
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

class QuickTranslationDataset(Dataset):
    """Quick dataset for testing."""
    
    def __init__(self, src_file, tgt_file, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read source and target sentences
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = [line.strip() for line in f if line.strip()]
        
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = [line.strip() for line in f if line.strip()]
        
        # Use first 50 sentences for quick testing
        self.src_sentences = self.src_sentences[:50]
        self.tgt_sentences = self.tgt_sentences[:50]
        
        assert len(self.src_sentences) == len(self.tgt_sentences), \
            "Source and target files must have the same number of lines"
        
        print(f"Loaded {len(self.src_sentences)} sentence pairs for quick testing")
    
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
    """Simplified transformer for quick testing."""
    
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=6, 
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Proper encoder-decoder transformer
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
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(self, src, tgt):
        # Embeddings with scaling
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        
        # Apply dropout
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Create causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(output)

def train_quick(model, dataloader, optimizer, criterion, device):
    """Quick training function."""
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
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def test_inference(model, tokenizer, device):
    """Test inference with the trained model."""
    model.eval()
    
    test_sentences = [
        "안녕하세요",
        "오늘 날씨가 정말 좋네요", 
        "감사합니다",
        "한국에 온 지 3개월이 되었어요"
    ]
    
    print("\nTesting inference:")
    print("-" * 30)
    
    with torch.no_grad():
        for sentence in test_sentences:
            # Tokenize
            tokens = tokenizer.encode(sentence)
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Create dummy target (for model input)
            tgt_tokens = [2]  # BOS token
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            
            # Forward pass
            output = model(input_tensor, tgt_tensor)
            
            # Get predictions
            predictions = output.argmax(dim=-1)
            
            print(f"Input: {sentence}")
            print(f"Tokens: {tokens}")
            print(f"Predictions: {predictions[0].cpu().numpy()}")
            
            # Try to decode predictions
            try:
                decoded = tokenizer.decode(predictions[0].cpu().numpy())
                print(f"Decoded: {decoded}")
            except:
                print("Decoding failed - predictions may be invalid")
            
            print()

def main():
    """Main quick training function."""
    
    # Configuration
    config = {
        'vocab_size': 1000,
        'd_model': 256,
        'nhead': 4,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'batch_size': 8,
        'learning_rate': 1e-3,
        'num_epochs': 10,
        'max_length': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Quick training configuration: {config}")
    
    # Initialize tokenizer
    tokenizer = SPTokenizer("kr_en_diverse.model")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")
    
    # Create dataset
    dataset = QuickTranslationDataset(
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
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("\nStarting quick training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_quick(model, dataloader, optimizer, criterion, config['device'])
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Test inference every 2 epochs
        if (epoch + 1) % 2 == 0:
            test_inference(model, tokenizer, config['device'])
    
    training_time = time.time() - start_time
    print(f"\nQuick training completed in {training_time:.2f} seconds")
    
    # Final test
    print("\nFinal inference test:")
    test_inference(model, tokenizer, config['device'])
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_model': 'kr_en_diverse.model'
    }, 'quick_test_model.pth')
    
    print("Model saved as: quick_test_model.pth")

if __name__ == "__main__":
    main()