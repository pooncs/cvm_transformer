#!/usr/bin/env python3
"""
Debug translation to understand the token issue.
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
        
        print(f"Source tokens: {src_tokens}")
        print(f"Device: {device}")
        
        # Convert to tensor
        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
        print(f"Source tensor shape: {src_tensor.shape}")
        
        # Start with BOS token
        tgt_tokens = [tokenizer.bos_id()]
        print(f"Starting with BOS: {tgt_tokens}")
        
        with torch.no_grad():
            for step in range(max_length):
                print(f"Step {step}: Current target tokens: {tgt_tokens}")
                
                # Convert current target to tensor
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long, device=device)
                print(f"Target tensor shape: {tgt_tensor.shape}")
                
                # Get model output
                output = self.forward(src_tensor, tgt_tensor)
                print(f"Output shape: {output.shape}")
                
                # Get next token (greedy decoding)
                next_token_logits = output[0, -1, :]
                print(f"Next token logits shape: {next_token_logits.shape}")
                print(f"Next token logits (first 10): {next_token_logits[:10]}")
                
                next_token_tensor = torch.argmax(next_token_logits)
                print(f"Next token (tensor): {next_token_tensor}")
                print(f"Next token (tensor type): {type(next_token_tensor)}")
                print(f"Next token (tensor item): {next_token_tensor.item()}")
                print(f"Next token (tensor item type): {type(next_token_tensor.item())}")
                
                next_token = int(next_token_tensor.item())
                print(f"Next token (final): {next_token}")
                print(f"Next token (final type): {type(next_token)}")
                
                # Add to sequence
                tgt_tokens.append(next_token)
                print(f"Updated target tokens: {tgt_tokens}")
                
                # Check for EOS
                if next_token == tokenizer.eos_id():
                    print("Found EOS token, breaking")
                    break
                
                if step >= 3:  # Limit for debugging
                    break
        
        return tgt_tokens

def test_translation():
    """Test translation with debug output."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file='data/processed/sentencepiece.model')
    vocab_size = tokenizer.vocab_size()
    
    # Create and load model
    model = SimpleNMT(vocab_size).to(device)
    model.load_state_dict(torch.load('models/nmt_model/minimal_model.pt', map_location=device))
    model.eval()
    
    # Test with a simple example
    test_text = "hello"
    src_tokens = tokenizer.encode_as_ids(test_text)
    src_tokens = [tokenizer.bos_id()] + src_tokens + [tokenizer.eos_id()]
    
    print(f"Test text: {test_text}")
    print(f"Encoded tokens: {src_tokens}")
    
    # Try translation
    try:
        tgt_tokens = model.translate(src_tokens, tokenizer, max_length=10)
        print(f"Translation tokens: {tgt_tokens}")
        translation = tokenizer.decode_ids(tgt_tokens)
        print(f"Translation: {translation}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_translation()