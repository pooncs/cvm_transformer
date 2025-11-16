#!/usr/bin/env python3
"""
Translation validation script to test accuracy.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

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
    
    def translate(self, src_tokens: List[int], tokenizer, max_length: int = 64) -> List[int]:
        """Translate source tokens to target tokens."""
        self.eval()
        device = next(self.parameters()).device
        
        # Convert to tensor
        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
        
        # Start with BOS token
        tgt_tokens = [tokenizer.bos_id()]
        
        with torch.no_grad():
            for _ in range(max_length):
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

def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file='data/processed/sentencepiece.model')
    vocab_size = tokenizer.vocab_size()
    
    # Create and load model
    model = SimpleNMT(vocab_size).to(device)
    model.load_state_dict(torch.load('models/nmt_model/minimal_model.pt', map_location=device))
    model.eval()
    
    return model, tokenizer

def calculate_bleu_score(reference: str, hypothesis: str) -> float:
    """Calculate simple BLEU-like score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0
    
    # Calculate precision for different n-grams
    scores = []
    for n in range(1, min(5, len(ref_tokens) + 1)):
        ref_ngrams = set([' '.join(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        hyp_ngrams = set([' '.join(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
        
        if len(hyp_ngrams) > 0:
            precision = len(ref_ngrams.intersection(hyp_ngrams)) / len(hyp_ngrams)
            scores.append(precision)
        else:
            scores.append(0.0)
    
    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    
    # Geometric mean of n-gram precisions
    if len(scores) > 0:
        scores_array = np.array(scores)
        geo_mean = np.exp(np.mean(np.log(scores_array + 1e-10)))
        return bp * geo_mean
    else:
        return 0.0

def validate_translation():
    """Run comprehensive translation validation."""
    logger.info("Starting translation validation...")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load validation data
    with open('data/processed/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    logger.info(f"Validating on {len(val_data)} samples")
    
    results = []
    total_bleu = 0.0
    exact_matches = 0  # Explicitly initialize as integer
    
    for i, item in enumerate(val_data):
        korean_text = item['korean']
        english_reference = item['english']
        
        # Tokenize source
        src_tokens = tokenizer.encode_as_ids(korean_text)
        src_tokens = src_tokens[:62]  # Make room for BOS/EOS
        src_tokens = [tokenizer.bos_id()] + src_tokens + [tokenizer.eos_id()]
        
        # Translate
        try:
            logger.debug(f"Translating sample {i}: {korean_text}")
            tgt_tokens = model.translate(src_tokens, tokenizer)
            logger.debug(f"Translation tokens: {tgt_tokens}")
            
            # Decode translation
            english_hypothesis = tokenizer.decode_ids(tgt_tokens)
            logger.debug(f"Translation: {english_hypothesis}")
            
            # Calculate BLEU score
            bleu_score = calculate_bleu_score(english_reference, english_hypothesis)
            total_bleu += bleu_score
            
            # Check exact match
            if english_reference.lower().strip() == english_hypothesis.lower().strip():
                exact_matches += 1
            
            results.append({
                'korean': korean_text,
                'reference': english_reference,
                'hypothesis': english_hypothesis,
                'bleu_score': bleu_score
            })
            
            if i % 10 == 0:
                logger.info(f"Sample {i+1}/{len(val_data)} - BLEU: {bleu_score:.4f}")
                logger.info(f"Reference: {english_reference}")
                logger.info(f"Hypothesis: {english_hypothesis}")
                logger.info("---")
                
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'korean': korean_text,
                'reference': english_reference,
                'hypothesis': '[ERROR]',
                'bleu_score': 0.0
            })
    
    # Calculate overall metrics
    avg_bleu = total_bleu / len(val_data)
    exact_match_rate = exact_matches / len(val_data) * 100
    
    logger.info("="*50)
    logger.info("VALIDATION RESULTS")
    logger.info("="*50)
    logger.info(f"Average BLEU Score: {avg_bleu:.4f}")
    logger.info(f"Exact Match Rate: {exact_match_rate:.2f}%")
    logger.info(f"Total Samples: {len(val_data)}")
    logger.info(f"Exact Matches: {exact_matches}")
    
    # Check if we achieved 99% accuracy
    if exact_match_rate >= 99.0:
        logger.info("✅ SUCCESS: Achieved 99%+ translation accuracy!")
    else:
        logger.info(f"❌ Below 99% accuracy target by {99.0 - exact_match_rate:.2f}%")
    
    # Save detailed results
    with open('validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("Detailed results saved to validation_results.json")
    
    return {
        'avg_bleu': avg_bleu,
        'exact_match_rate': exact_match_rate,
        'total_samples': len(val_data),
        'exact_matches': exact_matches
    }

if __name__ == "__main__":
    results = validate_translation()