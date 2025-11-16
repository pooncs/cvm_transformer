#!/usr/bin/env python3
"""
Test a single translation to identify the issue.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json

# Simple encoder-decoder model (same as training)
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
    
    def translate(self, src_tokens, tokenizer, max_length=64):
        """Translate source tokens to target tokens."""
        self.eval()
        device = next(self.parameters()).device
        
        # Convert to tensor
        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
        
        # Start with BOS token
        tgt_tokens = [tokenizer.bos_id()]
        
        with torch.no_grad():
            for step in range(max_length):
                # Convert current target to tensor
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long, device=device)
                
                # Get model output
                output = self.forward(src_tensor, tgt_tensor)
                
                # Get next token (greedy decoding)
                next_token_logits = output[0, -1, :]
                next_token = int(torch.argmax(next_token_logits).item())
                
                # Add to sequence
                tgt_tokens.append(next_token)
                
                # Check for EOS
                if next_token == tokenizer.eos_id():
                    break
        
        return tgt_tokens

def test_single_translation():
    """Test a single translation from validation data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file='data/processed/sentencepiece.model')
    vocab_size = tokenizer.vocab_size()
    
    # Create and load model
    model = SimpleNMT(vocab_size).to(device)
    model.load_state_dict(torch.load('models/nmt_model/minimal_model.pt', map_location=device))
    model.eval()
    
    # Load validation data
    with open('data/processed/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # Test first sample
    if len(val_data) > 0:
        item = val_data[0]
        korean_text = item['korean']
        english_reference = item['english']
        
        print(f"Korean: {korean_text}")
        print(f"English Reference: {english_reference}")
        
        # Tokenize source
        src_tokens = tokenizer.encode_as_ids(korean_text)
        src_tokens = src_tokens[:62]  # Make room for BOS/EOS
        src_tokens = [tokenizer.bos_id()] + src_tokens + [tokenizer.eos_id()]
        
        print(f"Source tokens: {src_tokens}")
        print(f"Source tokens type: {type(src_tokens)}")
        print(f"First token: {src_tokens[0]} (type: {type(src_tokens[0])})")
        
        # Try translation
        try:
            tgt_tokens = model.translate(src_tokens, tokenizer)
            print(f"Translation tokens: {tgt_tokens}")
            translation = tokenizer.decode_ids(tgt_tokens)
            print(f"Translation: {translation}")
            
            # Check types
            print(f"Translation tokens type: {type(tgt_tokens)}")
            if len(tgt_tokens) > 0:
                print(f"First translation token: {tgt_tokens[0]} (type: {type(tgt_tokens[0])})")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_single_translation()