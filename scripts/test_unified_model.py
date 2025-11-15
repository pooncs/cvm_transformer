#!/usr/bin/env python3
"""
Unified test script that can handle both enhanced and advanced models
"""

import torch
import sentencepiece as spm
import json
import time
import sys
import os

# Add the current directory to path for imports
sys.path.append('.')

def load_unified_model(model_path: str, device: str):
    """Load any model with unified architecture"""
    
    # Import the model architecture
    try:
        from train_enhanced import EnhancedTransformer
        ModelClass = EnhancedTransformer
        print("✅ Using EnhancedTransformer architecture")
    except ImportError:
        print("❌ Could not import EnhancedTransformer")
        return None
    
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
    
    model = ModelClass(**config).to(device)
    
    # Load checkpoint with error handling
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model from dict checkpoint")
                if 'val_loss' in checkpoint:
                    print(f"📊 Validation loss: {checkpoint['val_loss']:.4f}")
            else:
                # Try loading as state dict directly
                model.load_state_dict(checkpoint)
                print("✅ Loaded model state dict directly")
        else:
            print("❌ Unexpected checkpoint format")
            return None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    model.eval()
    return model

def translate_unified(model, tokenizer, korean_text: str, device: str, max_length: int = 128):
    """Translate Korean to English using unified approach"""
    
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

def test_model(model_path: str, model_name: str):
    """Test a specific model"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🧪 TESTING {model_name.upper()} MODEL")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file='kr_en_diverse.model')
    print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")
    
    # Load model
    print(f"Loading {model_name} model from {model_path}...")
    model = load_unified_model(model_path, device)
    
    if model is None:
        print(f"❌ Failed to load {model_name} model")
        return None
    
    print(f"✅ {model_name} model loaded successfully")
    
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
        ("이 소식이 너무 기쁩니다", "This news makes me very happy"),
        ("내일 봐요", "See you tomorrow"),
        ("잘 지내세요?", "How are you?"),
        ("만나서 반갑습니다", "Nice to meet you"),
        ("죄송합니다", "I'm sorry"),
        ("도와주세요", "Please help me")
    ]
    
    print(f"\n🧪 Testing {len(test_sentences)} sentence pairs")
    print("-" * 70)
    
    results = []
    total_score = 0
    
    for i, (korean, expected) in enumerate(test_sentences, 1):
        print(f"\nTest {i}:")
        print(f"🇰🇷 Korean: {korean}")
        print(f"🇺🇸 Expected: {expected}")
        
        # Translate
        start_time = time.time()
        translation = translate_unified(model, tokenizer, korean, device)
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
    print(f"\n📈 {model_name.upper()} MODEL RESULTS")
    print("=" * 50)
    print(f"Average translation score: {avg_score:.3f}")
    print(f"Tests passed (>0.3 score): {sum(1 for r in results if r['score'] > 0.3)}/{len(results)}")
    print(f"Tests with good overlap (>0.5): {sum(1 for r in results if r['score'] > 0.5)}/{len(results)}")
    print(f"Perfect translations (>0.8): {sum(1 for r in results if r['score'] > 0.8)}/{len(results)}")
    
    # Save results
    results_file = f'{model_name}_model_test_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': model_name,
            'average_score': avg_score,
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results if r['score'] > 0.3),
            'good_tests': sum(1 for r in results if r['score'] > 0.5),
            'perfect_tests': sum(1 for r in results if r['score'] > 0.8),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {results_file}")
    
    return {
        'model_name': model_name,
        'average_score': avg_score,
        'results': results
    }

def main():
    """Main function to test all available models"""
    
    print("🚀 UNIFIED MODEL TESTING SYSTEM")
    print("=" * 60)
    
    # Test all available models
    models_to_test = [
        ('best_enhanced_model.pth', 'enhanced'),
        ('best_advanced_simple_model.pth', 'advanced_simple'),
        ('clean_advanced_model.pth', 'clean_advanced')
    ]
    
    results = {}
    
    for model_path, model_name in models_to_test:
        if os.path.exists(model_path):
            result = test_model(model_path, model_name)
            if result:
                results[model_name] = result
        else:
            print(f"\n⚠️ Model {model_path} not found, skipping...")
    
    # Compare results
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("📊 MODEL COMPARISON")
        print(f"{'='*60}")
        
        for model_name, result in results.items():
            print(f"{model_name}: {result['average_score']:.3f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['average_score'])
        print(f"\n🏆 BEST MODEL: {best_model[0]} (Score: {best_model[1]['average_score']:.3f})")
    
    # Final assessment
    if results:
        best_score = max(result['average_score'] for result in results.values())
        print(f"\n🎯 FINAL ASSESSMENT")
        print("=" * 30)
        if best_score > 0.5:
            print("🎉 EXCELLENT: Translation system is working very well!")
        elif best_score > 0.3:
            print("✅ GOOD: Translation system is working well!")
        elif best_score > 0.1:
            print("⚠️ FAIR: Translation system is functional but needs improvement")
        else:
            print("❌ POOR: Translation system needs major fixes")
    
    return results

if __name__ == "__main__":
    main()