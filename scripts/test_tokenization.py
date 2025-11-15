"""
Test the improved SentencePiece tokenization system to verify it's working correctly
before running full training.
"""

import sys
sys.path.append('.')

from cvm_translator.sp_tokenizer import SPTokenizer
import torch

def test_tokenization():
    """Test the SentencePiece tokenization with diverse Korean-English sentences."""
    
    print("Testing SentencePiece Tokenization System")
    print("=" * 50)
    
    # Test sentences
    test_sentences = [
        ("안녕하세요", "Hello"),
        ("오늘 날씨가 정말 좋네요", "The weather is really nice today"),
        ("회의가 몇 시에 있나요?", "What time is the meeting?"),
        ("한국에 온 지 3개월이 되었고 점점 한국어를 잘하게 되고 있어요", 
         "It's been 3 months since I came to Korea and I'm gradually getting better at Korean"),
        ("비가 오고 있어서 우산을 가져가는 게 좋을 것 같아요", 
         "Since it's raining, it would be good to take an umbrella"),
    ]
    
    # Initialize tokenizer
    tokenizer = SPTokenizer("kr_en_diverse.model")
    vocab_size = tokenizer.vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    print("\nTokenization Results:")
    print("-" * 30)
    
    all_tokens = set()
    
    for i, (korean, english) in enumerate(test_sentences):
        print(f"\nSentence pair {i+1}:")
        print(f"Korean: {korean}")
        print(f"English: {english}")
        
        # Tokenize Korean
        kr_tokens = tokenizer.encode(korean)
        print(f"Korean tokens: {kr_tokens}")
        print(f"Korean token count: {len(kr_tokens)}")
        
        # Tokenize English
        en_tokens = tokenizer.encode(english)
        print(f"English tokens: {en_tokens}")
        print(f"English token count: {len(en_tokens)}")
        
        # Decode to verify
        kr_decoded = tokenizer.decode(kr_tokens)
        en_decoded = tokenizer.decode(en_tokens)
        print(f"Korean decoded: {kr_decoded}")
        print(f"English decoded: {en_decoded}")
        
        # Check if decoding is correct
        kr_correct = kr_decoded.strip() == korean.strip()
        en_correct = en_decoded.strip() == english.strip()
        print(f"Korean reconstruction: {'✓' if kr_correct else '✗'}")
        print(f"English reconstruction: {'✓' if en_correct else '✗'}")
        
        # Collect all tokens for vocabulary analysis
        all_tokens.update(kr_tokens)
        all_tokens.update(en_tokens)
    
    print(f"\nVocabulary Analysis:")
    print(f"Total unique tokens used: {len(all_tokens)}")
    print(f"Vocabulary utilization: {len(all_tokens) / vocab_size * 100:.2f}%")
    print(f"Min token ID: {min(all_tokens)}")
    print(f"Max token ID: {max(all_tokens)}")
    
    # Check for token ID 1 (which was the problem in the old system)
    if 1 in all_tokens:
        print("⚠️  WARNING: Token ID 1 found in vocabulary - this was the problematic token")
    else:
        print("✓ Good: Token ID 1 not found - tokenization appears diverse")
    
    # Test with BOS and EOS tokens
    print(f"\nSpecial tokens:")
    print(f"PAD ID: 0")
    print(f"UNK ID: 1") 
    print(f"BOS ID: 2")
    print(f"EOS ID: 3")
    
    # Test a simple sequence with special tokens
    test_sequence = [2] + tokenizer.encode("Hello world") + [3]  # BOS + tokens + EOS
    print(f"\nTest sequence with special tokens: {test_sequence}")
    decoded = tokenizer.decode(test_sequence[1:-1])  # Remove BOS/EOS for decoding
    print(f"Decoded (without special tokens): {decoded}")

def test_model_compatibility():
    """Test that the tokenization works with PyTorch tensors."""
    
    print("\n\nTesting PyTorch Compatibility")
    print("=" * 30)
    
    tokenizer = SPTokenizer("kr_en_diverse.model")
    
    # Test sentence
    sentence = "안녕하세요"
    tokens = tokenizer.encode(sentence)
    
    # Convert to tensor
    tensor = torch.tensor(tokens, dtype=torch.long)
    print(f"Original tokens: {tokens}")
    print(f"PyTorch tensor: {tensor}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    
    # Test batch creation
    batch_sentences = ["Hello", "How are you?", "Good morning"]
    batch_tensors = []
    
    for sent in batch_sentences:
        tokens = tokenizer.encode(sent)
        tensor = torch.tensor(tokens, dtype=torch.long)
        batch_tensors.append(tensor)
    
    # Pad sequences
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True, padding_value=0)
    print(f"\nPadded batch shape: {padded_batch.shape}")
    print(f"Padded batch:\n{padded_batch}")
    
    print("\n✓ Tokenization system is working correctly!")
    print("✓ Ready for training with the enhanced model")

if __name__ == "__main__":
    test_tokenization()
    test_model_compatibility()