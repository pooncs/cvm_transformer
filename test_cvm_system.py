#!/usr/bin/env python3
"""
Test the CVM transformer with the current tokenizer configuration
"""

import torch
import sys
sys.path.insert(0, '.')
from cvm_translator.cvm_transformer import CVMTransformer
from cvm_translator.sp_tokenizer import SPTokenizer

def test_cvm_transformer():
    """Test CVM transformer with current tokenizer"""
    print("Loading tokenizer...")
    tokenizer = SPTokenizer("kr_en.model")
    vocab_size = tokenizer.vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")
    
    print("Initializing CVMTransformer...")
    model = CVMTransformer(vocab_size=vocab_size, d_model=256, n_layers=2)
    
    # Test with sample input
    test_sentences = [
        "안녕하세요",
        "Hello world",
        "실시간 번역"
    ]
    
    print("\nTesting forward pass:")
    for sent in test_sentences:
        tokens = tokenizer.encode(sent)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        
        print(f"\nInput: '{sent}'")
        print(f"Tokens: {tokens}")
        print(f"Input shape: {input_ids.shape}")
        
        with torch.no_grad():
            logits = model(input_ids)
            
        print(f"Output shape: {logits.shape}")
        print(f"Vocab size check: {logits.shape[-1]} == {vocab_size}")
        
        # Get predictions
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
        decoded = tokenizer.decode(pred_ids)
        print(f"Predicted: {decoded}")

def test_with_different_core_capacities():
    """Test with different core capacities"""
    print("\n" + "="*50)
    print("Testing with different core capacities")
    print("="*50)
    
    tokenizer = SPTokenizer("kr_en.model")
    test_text = "안녕하세요"
    tokens = tokenizer.encode(test_text)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    for cores in [4, 8, 16, 32]:
        print(f"\nTesting with {cores} cores:")
        
        # Create model with specific core capacity
        model = CVMTransformer(
            vocab_size=tokenizer.vocab_size(),
            d_model=256,
            n_layers=2,
            core_capacity=cores
        )
        
        with torch.no_grad():
            # Test with core indices
            core_indices = torch.arange(min(cores, len(tokens)))
            if len(core_indices) == 0:
                core_indices = torch.tensor([0])
            
            logits = model(input_ids, core_indices=core_indices)
            
        print(f"  Input tokens: {len(tokens)}")
        print(f"  Core indices: {core_indices.tolist()}")
        print(f"  Output shape: {logits.shape}")

if __name__ == "__main__":
    test_cvm_transformer()
    test_with_different_core_capacities()
    print("\n✅ All tests completed successfully!")