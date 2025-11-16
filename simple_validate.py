#!/usr/bin/env python3
"""
Simple validation to check the exact match rate.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def simple_validate():
    """Simple validation to check exact match rate."""
    logger.info("Starting simple validation...")
    
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
    
    logger.info(f"Validating on {len(val_data)} samples")
    
    exact_matches = 0  # Explicitly initialize as int
    total_samples = len(val_data)
    
    # Test first 5 samples
    for i in range(min(5, len(val_data))):
        item = val_data[i]
        korean_text = item['korean']
        english_reference = item['english']
        
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Korean: {korean_text}")
        logger.info(f"Reference: {english_reference}")
        
        # Tokenize source
        src_tokens = tokenizer.encode_as_ids(korean_text)
        src_tokens = src_tokens[:62]  # Make room for BOS/EOS
        src_tokens = [tokenizer.bos_id()] + src_tokens + [tokenizer.eos_id()]
        
        # Translate
        try:
            tgt_tokens = model.translate(src_tokens, tokenizer)
            english_hypothesis = tokenizer.decode_ids(tgt_tokens)
            
            logger.info(f"Hypothesis: {english_hypothesis}")
            
            # Check exact match
            if english_reference.lower().strip() == english_hypothesis.lower().strip():
                exact_matches += 1
                logger.info("✅ EXACT MATCH!")
            else:
                logger.info("❌ No exact match")
                
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
    
    # Calculate exact match rate
    exact_match_rate = (exact_matches / total_samples) * 100
    
    logger.info("\n" + "="*50)
    logger.info("SIMPLE VALIDATION RESULTS")
    logger.info("="*50)
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Exact Matches: {exact_matches}")
    logger.info(f"Exact Match Rate: {exact_match_rate:.2f}%")
    
    # Check types
    logger.info(f"exact_matches type: {type(exact_matches)}")
    logger.info(f"total_samples type: {type(total_samples)}")
    logger.info(f"exact_match_rate type: {type(exact_match_rate)}")

if __name__ == "__main__":
    simple_validate()