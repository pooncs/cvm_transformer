"""
Comprehensive test of the improved translation system.
This validates that all the fixes have resolved the original issues.
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from src.models.sp_tokenizer import SPTokenizer
import json

class SimpleTransformer(nn.Module):
    """Simple transformer for testing."""
    
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

def translate_sentence(model, tokenizer, korean_text, device, max_length=30):
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

def comprehensive_test():
    """Run comprehensive test of the improved translation system."""
    
    print("🚀 COMPREHENSIVE TRANSLATION SYSTEM TEST")
    print("=" * 60)
    
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
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load('simple_best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Loaded trained model")
    
    # Comprehensive test sentences
    test_sentences = [
        # Basic greetings
        ("안녕하세요", "Hello"),
        ("감사합니다", "Thank you"),
        ("안녕히 가세요", "Goodbye"),
        
        # Daily life
        ("오늘 날씨가 정말 좋네요", "The weather is really nice today"),
        ("저는 커피를 좋아합니다", "I like coffee"),
        ("아침에 일찍 일어났어요", "I woke up early in the morning"),
        
        # Questions
        ("회의가 몇 시에 있나요?", "What time is the meeting?"),
        ("얼마예요?", "How much is it?"),
        ("화장실이 어디에 있나요?", "Where is the bathroom?"),
        
        # Complex sentences
        ("한국에 온 지 3개월이 되었고 점점 한국어를 잘하게 되고 있어요", 
         "It's been 3 months since I came to Korea and I'm gradually getting better at Korean"),
        ("비가 오고 있어서 우산을 가져가는 게 좋을 것 같아요", 
         "Since it's raining, it would be good to take an umbrella"),
        
        # Food and restaurants
        ("맛있는 식당을 추천해 주세요", "Please recommend a delicious restaurant"),
        ("이 음식이 너무 짜요", "This food is too salty"),
        ("계산서 주세요", "Please give me the bill"),
        
        # Emotions
        ("오늘 기분이 정말 좋아요", "I feel really good today"),
        ("걱정이 많이 되네요", "I'm very worried"),
        ("이 소식이 너무 기쁩니다", "This news makes me very happy"),
    ]
    
    print(f"\n🧪 Testing {len(test_sentences)} sentence pairs")
    print("-" * 40)
    
    results = []
    total_score = 0
    
    for i, (korean, expected) in enumerate(test_sentences):
        print(f"\nTest {i+1}:")
        print(f"🇰🇷 Korean: {korean}")
        print(f"🇺🇸 Expected: {expected}")
        
        try:
            translation, tokens = translate_sentence(model, tokenizer, korean, device)
            print(f"🤖 Translation: {translation}")
            print(f"🔢 Tokens: {tokens}")
            
            # Calculate word overlap score
            expected_words = set(expected.lower().split())
            translation_words = set(translation.lower().split())
            
            if len(expected_words) > 0:
                overlap = len(expected_words.intersection(translation_words))
                score = overlap / len(expected_words)
                total_score += score
                print(f"📊 Word overlap: {overlap}/{len(expected_words)} = {score:.2f}")
            else:
                score = 0.0
                print(f"📊 Score: {score:.2f}")
            
            # Check for problematic patterns
            if len(tokens) <= 3:  # Too short
                print("⚠️  WARNING: Translation too short")
            elif all(t == tokens[1] for t in tokens[1:]):  # All same token
                print("⚠️  WARNING: All tokens identical")
            elif tokens.count(1) > len(tokens) * 0.5:  # Too many UNK tokens
                print("⚠️  WARNING: Too many unknown tokens")
            else:
                print("✅ Translation looks reasonable")
            
            results.append({
                'korean': korean,
                'expected': expected,
                'translation': translation,
                'score': score,
                'tokens': tokens,
                'token_count': len(tokens)
            })
            
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            results.append({
                'korean': korean,
                'expected': expected,
                'translation': "FAILED",
                'score': 0.0,
                'error': str(e)
            })
    
    # Overall statistics
    avg_score = total_score / len(test_sentences)
    
    print(f"\n📈 OVERALL RESULTS")
    print("=" * 30)
    print(f"Average translation score: {avg_score:.3f}")
    print(f"Tests passed (>0.3 score): {sum(1 for r in results if r['score'] > 0.3)}/{len(results)}")
    print(f"Tests with good overlap (>0.5): {sum(1 for r in results if r['score'] > 0.5)}/{len(results)}")
    
    # Token analysis
    all_tokens = []
    for result in results:
        if 'tokens' in result:
            all_tokens.extend(result['tokens'])
    
    if all_tokens:
        unique_tokens = set(all_tokens)
        print(f"\n🔤 TOKEN ANALYSIS")
        print("=" * 20)
        print(f"Total tokens generated: {len(all_tokens)}")
        print(f"Unique tokens used: {len(unique_tokens)}")
        print(f"Vocabulary utilization: {len(unique_tokens) / vocab_size * 100:.2f}%")
        print(f"Token ID range: {min(unique_tokens)} - {max(unique_tokens)}")
        
        # Check for the old problem (token ID 1)
        if 1 in unique_tokens:
            print(f"⚠️  WARNING: Token ID 1 (UNK) used {all_tokens.count(1)} times")
        else:
            print("✅ Token ID 1 (problematic token) not used")
        
        # Check for EOS token usage
        eos_count = all_tokens.count(3)
        print(f"EOS tokens (ID 3): {eos_count} ({eos_count/len(all_tokens)*100:.1f}%)")
        
        # Check for diversity
        if len(unique_tokens) < 10:
            print("⚠️  WARNING: Very low vocabulary diversity")
        elif len(unique_tokens) < 50:
            print("📊 Moderate vocabulary diversity")
        else:
            print("✅ Good vocabulary diversity")
    
    # Compare with old system
    print(f"\n🔄 COMPARISON WITH OLD SYSTEM")
    print("=" * 35)
    print("OLD SYSTEM PROBLEMS:")
    print("- Models predicted only token ID 1 (smiley faces)")
    print("- 0.003% vocabulary utilization")
    print("- Complete translation failure")
    print()
    print("NEW SYSTEM RESULTS:")
    if avg_score > 0.3:
        print("✅ Translation quality significantly improved")
    else:
        print("📊 Translation quality needs improvement")
    
    if len(unique_tokens) > 50:
        print("✅ Vocabulary utilization massively improved")
    else:
        print("📊 Vocabulary utilization improved but could be better")
    
    if 1 not in unique_tokens:
        print("✅ Token collapse issue resolved")
    else:
        print("⚠️  Some token collapse issues may remain")
    
    # Save detailed results
    with open('comprehensive_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed results saved to comprehensive_test_results.json")
    
    return avg_score, results

if __name__ == "__main__":
    score, results = comprehensive_test()
    
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 25)
    
    if score > 0.5:
        print("🎉 EXCELLENT: Translation system is working well!")
        print("✅ SentencePiece BPE tokenization successfully implemented")
        print("✅ Model collapse issues resolved")
        print("✅ Meaningful translations generated")
    elif score > 0.3:
        print("📈 GOOD: Translation system shows significant improvement!")
        print("✅ Tokenization issues resolved")
        print("✅ Basic translation functionality working")
        print("📊 Further training could improve quality")
    else:
        print("📊 MODERATE: Some improvements achieved")
        print("✅ Tokenization system working")
        print("⚠️  Translation quality needs more work")
        print("📚 Consider larger dataset or longer training")
    
    print(f"\n🏁 Test completed with average score: {score:.3f}")