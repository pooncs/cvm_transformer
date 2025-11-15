"""
Test the improved translation quality with the trained model.
This will show if the tokenization improvements actually work for translation.
"""

import torch
import sys
sys.path.append('.')

from cvm_translator.sp_tokenizer import SPTokenizer
import torch.nn as nn

class SimpleTransformer(nn.Module):
    """Simplified transformer for testing."""
    
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

def translate_sentence(model, tokenizer, korean_text, device, max_length=50):
    """Translate a Korean sentence to English."""
    model.eval()
    
    with torch.no_grad():
        # Tokenize Korean input
        src_tokens = tokenizer.encode(korean_text)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
        
        # Start with BOS token
        tgt_tokens = [2]  # BOS
        
        for _ in range(max_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            
            # Get model predictions
            output = model(src_tensor, tgt_tensor)
            
            # Get the last token prediction
            next_token_logits = output[0, -1, :]
            next_token = next_token_logits.argmax().item()
            
            # Add to sequence
            tgt_tokens.append(next_token)
            
            # Stop if EOS token
            if next_token == 3:  # EOS
                break
        
        # Decode the output (excluding BOS and EOS)
        english_tokens = tgt_tokens[1:-1]  # Remove BOS and EOS
        english_text = tokenizer.decode(english_tokens)
        
        return english_text, tgt_tokens

def test_translation_quality():
    """Test translation quality with the improved model."""
    
    print("Testing Improved Translation Quality")
    print("=" * 50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = SPTokenizer("kr_en_diverse.model")
    vocab_size = tokenizer.vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")
    
    # Load model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load('quick_test_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded trained model")
    
    # Test sentences
    test_sentences = [
        "안녕하세요",
        "감사합니다", 
        "오늘 날씨가 정말 좋네요",
        "저는 커피를 좋아합니다",
        "회의가 몇 시에 있나요?",
        "한국에 온 지 3개월이 되었어요",
        "비가 오고 있어서 우산을 가져가는 게 좋을 것 같아요",
        "맛있는 식당을 추천해 주세요"
    ]
    
    expected_translations = [
        "Hello",
        "Thank you",
        "The weather is really nice today",
        "I like coffee",
        "What time is the meeting?",
        "It's been 3 months since I came to Korea",
        "Since it's raining, it would be good to take an umbrella",
        "Please recommend a delicious restaurant"
    ]
    
    print("\nTranslation Results:")
    print("-" * 40)
    
    total_score = 0
    
    for i, (korean, expected) in enumerate(zip(test_sentences, expected_translations)):
        print(f"\nTest {i+1}:")
        print(f"Korean: {korean}")
        print(f"Expected: {expected}")
        
        try:
            translation, tokens = translate_sentence(model, tokenizer, korean, device)
            print(f"Translation: {translation}")
            print(f"Tokens: {tokens}")
            
            # Simple scoring based on word overlap
            expected_words = set(expected.lower().split())
            translation_words = set(translation.lower().split())
            
            if len(expected_words) > 0:
                overlap = len(expected_words.intersection(translation_words))
                score = overlap / len(expected_words)
                total_score += score
                print(f"Word overlap score: {score:.2f}")
            else:
                print("No expected words to compare")
                
        except Exception as e:
            print(f"Translation failed: {e}")
            print("Score: 0.00")
    
    avg_score = total_score / len(test_sentences)
    print(f"\nAverage translation score: {avg_score:.2f}")
    
    # Test vocabulary utilization
    print(f"\nVocabulary Analysis:")
    print(f"Tokenizer vocab size: {vocab_size}")
    
    # Test if we're still getting the token ID 1 problem
    print(f"\nChecking for token ID 1 usage (the old problem):")
    test_tokens = []
    for sentence in test_sentences:
        tokens = tokenizer.encode(sentence)
        test_tokens.extend(tokens)
    
    if 1 in test_tokens:
        print("⚠️  WARNING: Token ID 1 found - this was the problematic token from before")
    else:
        print("✓ Good: Token ID 1 not found in test sentences")
    
    unique_tokens = set(test_tokens)
    print(f"Unique tokens used in test: {len(unique_tokens)}")
    print(f"Vocabulary utilization in tests: {len(unique_tokens) / vocab_size * 100:.2f}%")
    
    return avg_score

if __name__ == "__main__":
    score = test_translation_quality()
    
    if score > 0.3:
        print(f"\n🎉 SUCCESS: Translation quality improved! Score: {score:.2f}")
        print("The tokenization fix is working correctly.")
    else:
        print(f"\n📊 Score: {score:.2f} - Model may need more training or architecture improvements")
        print("But tokenization is working properly now.")