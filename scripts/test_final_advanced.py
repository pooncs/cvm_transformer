#!/usr/bin/env python3
"""
Final test of the advanced translation system with clean model export
"""

import torch
import sentencepiece as spm
import json
import time

def load_clean_model(model_path: str, device: str):
    """Load the clean exported model"""
    
    # Import the model architecture
    from train_enhanced import EnhancedTransformer
    
    # Model configuration
    config = {
        'vocab_size': 1000,
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 12,
        'num_decoder_layers': 12,
        'dim_feedforward': 2048,
        'dropout': 0.1
    }
    
    model = EnhancedTransformer(**config).to(device)
    
    # Load clean checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded {checkpoint['training_info']['model_type']} model")
        print(f"📊 Best validation loss: {checkpoint['training_info']['best_val_loss']}")
    except Exception as e:
        print(f"⚠️ Loading as state dict: {e}")
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def translate_clean(model, tokenizer, korean_text: str, device: str, max_length: int = 128):
    """Translate Korean to English using the clean model"""
    
    # Tokenize input
    src_tokens = tokenizer.encode(korean_text, out_type=int)
    src_tokens = [1] + src_tokens + [2]  # Add SOS and EOS
    
    # Pad to max_length
    while len(src_tokens) < max_length:
        src_tokens.append(0)
    src_tokens = src_tokens[:max_length]
    
    src_tensor = torch.tensor([src_tokens]).to(device)
    
    # Generate translation
    with torch.no_grad():
        # Initialize target with SOS token
        tgt_tokens = [2]  # SOS token
        
        for _ in range(max_length - 1):
            # Prepare target tensor
            tgt_input = tgt_tokens + [0] * (max_length - len(tgt_tokens))
            tgt_input = tgt_input[:max_length]
            tgt_tensor = torch.tensor([tgt_input]).to(device)
            
            # Get model output
            output = model(src_tensor, tgt_tensor)
            
            # Get next token
            next_token_logits = output[0, len(tgt_tokens) - 1]
            next_token = next_token_logits.argmax().item()
            
            # Check for EOS token
            if next_token == 3:  # EOS token
                break
                
            tgt_tokens.append(next_token)
    
    # Decode translation
    translation_tokens = tgt_tokens[1:]  # Remove SOS
    translation = tokenizer.decode(translation_tokens)
    
    return translation

def calculate_word_overlap_score(predicted: str, expected: str) -> float:
    """Calculate word overlap score between predicted and expected translations"""
    
    pred_words = set(predicted.lower().split())
    expected_words = set(expected.lower().split())
    
    if not expected_words:
        return 1.0 if not pred_words else 0.0
    
    overlap = len(pred_words & expected_words)
    return overlap / len(expected_words)

def main():
    """Main test function"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file='kr_en_diverse.model')
    print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")
    
    # Load clean model
    print("Loading clean advanced model...")
    model = load_clean_model('clean_advanced_model.pth', device)
    print("✅ Clean advanced model loaded successfully")
    
    # Test sentences
    test_sentences = [
        ("안녕하세요", "Hello"),
        ("감사합니다", "Thank you"),
        ("안녕히 가세요", "Goodbye"),
        ("오늘 날씨가 정말 좋네요", "The weather is really nice today"),
        ("저는 커피를 좋아합니다", "I like coffee"),
        ("회의가 몇 시에 있나요?", "What time is the meeting?"),
        ("얼마예요?", "How much is it?"),
        ("화장실이 어디에 있나요?", "Where is the bathroom?"),
        ("계산서 주세요", "Please give me the bill"),
        ("오늘 기분이 정말 좋아요", "I feel really good today"),
        ("걱정이 많이 되네요", "I'm very worried"),
        ("이 소식이 너무 기쁩니다", "This news makes me very happy")
    ]
    
    print(f"\n🧪 Testing {len(test_sentences)} sentence pairs with Clean Advanced Model")
    print("=" * 70)
    
    results = []
    total_score = 0
    
    for i, (korean, expected) in enumerate(test_sentences, 1):
        print(f"\nTest {i}:")
        print(f"🇰🇷 Korean: {korean}")
        print(f"🇺🇸 Expected: {expected}")
        
        # Translate
        start_time = time.time()
        translation = translate_clean(model, tokenizer, korean, device)
        translation_time = time.time() - start_time
        
        # Calculate score
        score = calculate_word_overlap_score(translation, expected)
        total_score += score
        
        print(f"🤖 Clean Translation: {translation}")
        print(f"📊 Word overlap: {score:.2f}")
        print(f"⏱️ Translation time: {translation_time:.3f}s")
        
        results.append({
            'korean': korean,
            'expected': expected,
            'translation': translation,
            'score': score,
            'time': translation_time
        })
    
    # Summary
    avg_score = total_score / len(test_sentences)
    print(f"\n📈 FINAL ADVANCED MODEL RESULTS")
    print("=" * 50)
    print(f"Average translation score: {avg_score:.3f}")
    print(f"Tests passed (>0.3 score): {sum(1 for r in results if r['score'] > 0.3)}/{len(results)}")
    print(f"Tests with good overlap (>0.5): {sum(1 for r in results if r['score'] > 0.5)}/{len(results)}")
    
    # Compare with previous results
    try:
        with open('comprehensive_test_results.json', 'r', encoding='utf-8') as f:
            previous_results = json.load(f)
            previous_score = previous_results.get('average_score', 0)
            print(f"\n🔄 COMPARISON WITH ENHANCED MODEL:")
            print(f"Enhanced model score: {previous_score:.3f}")
            print(f"Advanced model score: {avg_score:.3f}")
            print(f"Improvement: {avg_score - previous_score:.3f}")
            
            if avg_score > previous_score:
                print("✅ Advanced model shows improvement!")
            else:
                print("⚠️ Advanced model similar to enhanced model")
                
    except Exception as e:
        print(f"\nℹ️ Could not compare with previous results: {e}")
    
    # Save results
    with open('final_advanced_model_test_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'average_score': avg_score,
            'results': results,
            'model': 'Clean Advanced model with LR scheduling'
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to final_advanced_model_test_results.json")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 30)
    if avg_score > 0.5:
        print("🎉 EXCELLENT: Translation system is working very well!")
    elif avg_score > 0.3:
        print("✅ GOOD: Translation system is working well!")
    elif avg_score > 0.1:
        print("⚠️ FAIR: Translation system is functional but needs improvement")
    else:
        print("❌ POOR: Translation system needs major fixes")
    
    return avg_score

if __name__ == "__main__":
    main()