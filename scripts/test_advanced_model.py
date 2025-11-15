#!/usr/bin/env python3
"""
Test the advanced trained model with learning rate scheduling
"""

import torch
import sentencepiece as spm
import json
from typing import List, Dict
import time

def load_advanced_model(model_path: str, device: str):
    """Load the advanced trained model"""
    
    # Import the model architecture
    from train_enhanced import EnhancedTransformer
    
    # Model configuration (must match training)
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
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def translate_advanced(model, tokenizer, korean_text: str, device: str, max_length: int = 128):
    """Translate Korean to English using the advanced model"""
    
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
    
    # Load model
    print("Loading advanced model...")
    model = load_advanced_model('best_advanced_simple_model.pth', device)
    print("✅ Advanced model loaded successfully")
    
    # Test sentences
    test_sentences = [
        ("안녕하세요", "Hello"),
        ("감사합니다", "Thank you"),
        ("안녕히 가세요", "Goodbye"),
        ("오늘 날씨가 정말 좋네요", "The weather is really nice today"),
        ("저는 커피를 좋아합니다", "I like coffee"),
        ("아침에 일찍 일어났어요", "I woke up early in the morning"),
        ("회의가 몇 시에 있나요?", "What time is the meeting?"),
        ("얼마예요?", "How much is it?"),
        ("화장실이 어디에 있나요?", "Where is the bathroom?"),
        ("한국에 온 지 3개월이 되었고 점점 한국어를 잘하게 되고 있어요", "It's been 3 months since I came to Korea and I'm gradually getting better at Korean"),
        ("비가 오고 있어서 우산을 가져가는 게 좋을 것 같아요", "Since it's raining, it would be good to take an umbrella"),
        ("맛있는 식당을 추천해 주세요", "Please recommend a delicious restaurant"),
        ("이 음식이 너무 짜요", "This food is too salty"),
        ("계산서 주세요", "Please give me the bill"),
        ("오늘 기분이 정말 좋아요", "I feel really good today"),
        ("걱정이 많이 되네요", "I'm very worried"),
        ("이 소식이 너무 기쁩니다", "This news makes me very happy")
    ]
    
    print(f"\n🧪 Testing {len(test_sentences)} sentence pairs")
    print("=" * 60)
    
    results = []
    total_score = 0
    
    for i, (korean, expected) in enumerate(test_sentences, 1):
        print(f"\nTest {i}:")
        print(f"🇰🇷 Korean: {korean}")
        print(f"🇺🇸 Expected: {expected}")
        
        # Translate
        start_time = time.time()
        translation = translate_advanced(model, tokenizer, korean, device)
        translation_time = time.time() - start_time
        
        # Calculate score
        score = calculate_word_overlap_score(translation, expected)
        total_score += score
        
        print(f"🤖 Translation: {translation}")
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
    print(f"\n📈 OVERALL RESULTS")
    print("=" * 40)
    print(f"Average translation score: {avg_score:.3f}")
    print(f"Tests passed (>0.3 score): {sum(1 for r in results if r['score'] > 0.3)}/{len(results)}")
    print(f"Tests with good overlap (>0.5): {sum(1 for r in results if r['score'] > 0.5)}/{len(results)}")
    
    # Compare with previous results
    try:
        with open('comprehensive_test_results.json', 'r') as f:
            previous_results = json.load(f)
            previous_score = previous_results.get('average_score', 0)
            print(f"\n🔄 COMPARISON WITH PREVIOUS MODEL:")
            print(f"Previous average score: {previous_score:.3f}")
            print(f"Advanced model score: {avg_score:.3f}")
            print(f"Improvement: {avg_score - previous_score:.3f}")
    except:
        print("\nℹ️ No previous results to compare with")
    
    # Save results
    with open('advanced_model_test_results.json', 'w') as f:
        json.dump({
            'average_score': avg_score,
            'results': results,
            'model': 'best_advanced_simple_model.pth'
        }, f, indent=2)
    
    print(f"\n💾 Results saved to advanced_model_test_results.json")
    
    return avg_score

if __name__ == "__main__":
    main()