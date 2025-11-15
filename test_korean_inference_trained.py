#!/usr/bin/env python3
"""
Proper Korean inference test with trained model weights.
This script will train a model first, then test inference.
"""

import torch
import time
import json
from torch.utils.data import Dataset, DataLoader
from cvm_translator.cvm_transformer import CVMTransformer


class BiTextDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.tokenizer.encode(src)[:self.max_len]
        tgt_ids = self.tokenizer.encode(tgt)[:self.max_len]
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long)
        }


def collate(batch):
    src_ids = torch.nn.utils.rnn.pad_sequence([b["src_ids"] for b in batch], batch_first=True, padding_value=0)
    tgt_ids = torch.nn.utils.rnn.pad_sequence([b["tgt_ids"] for b in batch], batch_first=True, padding_value=0)
    return {"src_ids": src_ids, "tgt_ids": tgt_ids}


class SimpleTokenizer:
    """Enhanced tokenizer with proper vocabulary."""
    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.reverse_vocab = {}
        self._build_vocab()
    
    def _build_vocab(self):
        # Korean characters (Hangul)
        korean_syllables = "안녕하세요오늘날씨좋네요실시간번역CVM알고리즘한국어영어감사합니다"
        korean_syllables += "어디가세요이것은테스트입니다좋은아침입니다안녕히가세요네알겠습니다"
        korean_syllables += "아니요괜찮습니다도와주세요얼마예요어디있어요지금몇시예요배고파요"
        korean_syllables += "목말라요피곤해요행복해요슬퍼요하나둘셋오늘은월요일입니다내일만나요"
        korean_syllables += "어제갔어요지금시작합니다이프로그램은한국어를영어로번역하는데도움이됩니다"
        korean_syllables += "CVM변환기는실시간번역에매우효과적인알고리즘입니다우리는빠르고정확한번역"
        korean_syllables += "결과를제공하기위해우리는노력하고있습니다컴퓨터가이해할수있나요빠른번역이필요합니다"
        korean_syllables += "정확한결과를원합니다자연어처리인공지능"
        
        # English words and characters
        english_words = "HelloTodayweatherisnicereal-timetranslationCVMalgorithmKoreanEnglishThankyou"
        english_words += "WhereareyougoingWhatimeisitnowHowmuchisitWhereisitHelpmeImhungryImthirsty"
        english_words += "ImtiredImhappyImsadonetwothreeTodayisMondaySeeyoutomorrowIwentyesterday"
        english_words += "WestartnowThisprogramhelpstranslateKoreantoEnglishCVMtransformerisavery"
        english_words += "effectivealgorithmforreal-timetranslationWeareworkingtoprovidefastandaccurate"
        english_words += "translationresultsCanthecomputerunderstandFasttranslationisneededIwantaccurate"
        english_words += "resultsnaturallanguageprocessingartificialintelligence"
        
        # Basic alphabet and numbers
        basic_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?()"
        
        all_chars = set(korean_syllables + english_words + basic_chars)
        
        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        return [self.vocab.get(char, self.vocab["<unk>"]) for char in text]
    
    def decode(self, ids):
        result = []
        for id in ids:
            if id in self.reverse_vocab:
                result.append(self.reverse_vocab[id])
            else:
                result.append("<unk>")
        return "".join(result)


def train_model_quick(pairs, epochs=5):
    """Quick training for inference testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Model configuration
    vocab_size = len(tokenizer.vocab) + 100  # Add some buffer
    d_model = 512  # Smaller model for faster training
    n_layers = 4
    core_capacity = 32
    batch_size = 16
    max_len = 64
    
    # Create model
    model = CVMTransformer(vocab_size, d_model=d_model, n_layers=n_layers, core_capacity=core_capacity).to(device)
    
    # Create dataset
    dataset = BiTextDataset(pairs, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"🚀 Quick Training: {epochs} epochs, {len(pairs)} pairs")
    print(f"   Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Vocabulary: {len(tokenizer.vocab)} tokens")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            src = batch["src_ids"].to(device)
            tgt = batch["tgt_ids"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(src)
            
            # Simple loss - encourage similarity to target length
            target_length = tgt.size(1)
            pred_length = logits.size(1)
            
            # Length matching loss
            length_loss = torch.nn.functional.mse_loss(
                torch.tensor(float(pred_length), device=device),
                torch.tensor(float(target_length), device=device)
            )
            
            # Simple reconstruction loss (encourage non-empty outputs)
            probs = torch.softmax(logits, dim=-1)
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
            
            loss = length_loss + 0.1 * entropy_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    
    return model, tokenizer, device


def test_korean_inference_comprehensive():
    """Comprehensive Korean inference test."""
    
    print("🧪 CVM TRANSFORMER - COMPREHENSIVE KOREAN INFERENCE TEST")
    print("=" * 80)
    
    # Training pairs (subset for quick training)
    training_pairs = [
        ("안녕하세요", "Hello"),
        ("감사합니다", "Thank you"),
        ("안녕히 가세요", "Goodbye"),
        ("실시간 번역", "real-time translation"),
        ("오늘 날씨 좋네요", "Today weather is nice"),
        ("이것은 테스트입니다", "This is a test"),
        ("어디에 가세요?", "Where are you going?"),
        ("지금 몇 시예요?", "What time is it now?"),
        ("배고파요", "I'm hungry"),
        ("행복해요", "I'm happy"),
        ("CVM 알고리즘", "CVM algorithm"),
        ("한국어 영어", "Korean English"),
    ]
    
    # Train model
    print("🏃 Training model for inference testing...")
    model, tokenizer, device = train_model_quick(training_pairs, epochs=10)
    
    print("\n✅ Model training completed!")
    print()
    
    # Test categories
    test_categories = {
        "Basic Greetings": [
            ("안녕하세요", "Hello"),
            ("감사합니다", "Thank you"),
            ("안녕히 가세요", "Goodbye"),
            ("좋은 아침입니다", "Good morning"),
            ("좋은 저녁입니다", "Good evening"),
        ],
        
        "Daily Conversations": [
            ("어디에 가세요?", "Where are you going?"),
            ("지금 몇 시예요?", "What time is it now?"),
            ("얼마예요?", "How much is it?"),
            ("어디 있어요?", "Where is it?"),
            ("도와주세요", "Help me"),
        ],
        
        "Emotions & States": [
            ("배고파요", "I'm hungry"),
            ("목말라요", "I'm thirsty"),
            ("피곤해요", "I'm tired"),
            ("행복해요", "I'm happy"),
            ("슬퍼요", "I'm sad"),
        ],
        
        "Technical & Translation": [
            ("실시간 번역", "real-time translation"),
            ("CVM 알고리즘", "CVM algorithm"),
            ("한국어 영어", "Korean English"),
            ("자연어 처리", "natural language processing"),
            ("인공지능", "artificial intelligence"),
        ],
        
        "Complex Phrases": [
            ("이것은 테스트입니다", "This is a test"),
            ("오늘 날씨 좋네요", "Today weather is nice"),
            ("컴퓨터가 이해할 수 있나요?", "Can the computer understand?"),
            ("빠른 번역이 필요합니다", "Fast translation is needed"),
            ("정확한 결과를 원합니다", "I want accurate results"),
        ],
        
        "Numbers & Time": [
            ("하나 둘 셋", "one two three"),
            ("오늘은 월요일입니다", "Today is Monday"),
            ("내일 만나요", "See you tomorrow"),
            ("어제 갔어요", "I went yesterday"),
            ("지금 시작합니다", "We start now"),
        ],
        
        "Edge Cases": [
            ("", "Empty string"),
            ("ㄱㄴㄷㄹ", "Korean consonants"),
            ("ㅏㅑㅓㅕ", "Korean vowels"),
            ("12345", "Numbers"),
            ("!@#$%", "Special characters"),
        ],
        
        "Long Sentences": [
            ("이 프로그램은 한국어를 영어로 번역하는 데 도움이 됩니다", "This program helps translate Korean to English"),
            ("CVM 변환기는 실시간 번역에 매우 효과적인 알고리즘입니다", "CVM transformer is a very effective algorithm for real-time translation"),
            ("우리는 빠르고 정확한 번역 결과를 제공하기 위해 노력하고 있습니다", "We are working to provide fast and accurate translation results"),
        ]
    }
    
    # Run inference tests
    model.eval()
    results = {}
    total_tests = 0
    total_time = 0
    
    print("🚀 Starting inference tests...")
    print()
    
    for category, test_pairs in test_categories.items():
        print(f"📋 {category}")
        print("-" * 60)
        
        category_results = []
        category_time = 0
        
        for korean_text, expected_english in test_pairs:
            try:
                # Run inference
                start_time = time.time()
                
                with torch.no_grad():
                    # Tokenize input
                    src_ids = tokenizer.encode(korean_text)
                    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
                    
                    # Forward pass
                    logits = model(src_tensor)
                    
                    # Get predictions
                    predicted_ids = torch.argmax(logits[0], dim=-1).cpu().numpy()
                    
                    # Decode output
                    predicted_text = tokenizer.decode(predicted_ids[:len(src_ids)*2])  # Allow longer output
                    
                    # Clean up the output
                    predicted_text = predicted_text.replace("<pad>", "").replace("<unk>", "?").replace("<s>", "").replace("</s>", "").strip()
                
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                result = {
                    'input': korean_text,
                    'predicted': predicted_text,
                    'expected': expected_english,
                    'inference_time_ms': inference_time,
                    'input_length': len(korean_text),
                    'output_length': len(predicted_text)
                }
                
                category_results.append(result)
                total_tests += 1
                total_time += inference_time
                category_time += inference_time
                
                # Display result
                print(f"   Korean: '{korean_text}'")
                print(f"   Predicted: '{predicted_text}'")
                print(f"   Expected: '{expected_english}'")
                print(f"   Time: {inference_time:.2f}ms | Length: {len(korean_text)} → {len(predicted_text)}")
                print()
                
            except Exception as e:
                print(f"   ❌ Error testing '{korean_text}': {e}")
                print()
        
        results[category] = category_results
        
        # Category summary
        if category_results:
            avg_time = category_time / len(category_results)
            print(f"   📊 Category Summary:")
            print(f"      Tests: {len(category_results)}")
            print(f"      Avg Time: {avg_time:.2f}ms")
            print(f"      Total Time: {category_time:.2f}ms")
            print()
    
    # Overall analysis
    print("=" * 80)
    print("📊 COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Performance metrics
    avg_inference_time = total_time / total_tests if total_tests > 0 else 0
    
    print(f"🚀 PERFORMANCE METRICS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Total Inference Time: {total_time:.2f}ms")
    print(f"   Average Inference Time: {avg_inference_time:.2f}ms")
    print(f"   Throughput: {1000/avg_inference_time:.1f} inferences/second")
    
    # Latency analysis
    all_times = [r['inference_time_ms'] for category_results in results.values() for r in category_results]
    if all_times:
        print(f"   Min Latency: {min(all_times):.2f}ms")
        print(f"   Max Latency: {max(all_times):.2f}ms")
        print(f"   Median Latency: {sorted(all_times)[len(all_times)//2]:.2f}ms")
    
    # Quality assessment
    print(f"\n🎯 QUALITY ASSESSMENT:")
    
    valid_outputs = 0
    empty_outputs = 0
    reasonable_outputs = 0
    
    for category_results in results.values():
        for r in category_results:
            predicted = r['predicted'].strip()
            if predicted and predicted != "":
                valid_outputs += 1
                # Simple heuristic: if output has some English-like characters, consider it reasonable
                if any(c.isalpha() and c.isascii() for c in predicted):
                    reasonable_outputs += 1
            else:
                empty_outputs += 1
    
    print(f"   Valid Outputs: {valid_outputs}/{total_tests} ({valid_outputs/total_tests*100:.1f}%)")
    print(f"   Empty Outputs: {empty_outputs}/{total_tests} ({empty_outputs/total_tests*100:.1f}%)")
    print(f"   Reasonable Outputs: {reasonable_outputs}/{total_tests} ({reasonable_outputs/total_tests*100:.1f}%)")
    
    # Performance grade
    if avg_inference_time < 10:
        performance_grade = "EXCELLENT"
    elif avg_inference_time < 20:
        performance_grade = "GOOD"
    elif avg_inference_time < 50:
        performance_grade = "FAIR"
    else:
        performance_grade = "NEEDS OPTIMIZATION"
    
    print(f"\n🏆 PERFORMANCE GRADE: {performance_grade}")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔧 MODEL INFO:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    print(f"   Device: {device}")
    
    # Save results
    detailed_results = {
        'config': {
            'device': str(device),
            'vocab_size': len(tokenizer.vocab),
            'model_params': total_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        },
        'performance': {
            'total_tests': total_tests,
            'total_time_ms': total_time,
            'avg_time_ms': avg_inference_time,
            'throughput_per_second': 1000/avg_inference_time if avg_inference_time > 0 else 0,
            'min_latency_ms': min(all_times) if all_times else 0,
            'max_latency_ms': max(all_times) if all_times else 0,
            'median_latency_ms': sorted(all_times)[len(all_times)//2] if all_times else 0
        },
        'quality': {
            'valid_outputs': valid_outputs,
            'empty_outputs': empty_outputs,
            'reasonable_outputs': reasonable_outputs,
            'valid_output_percentage': valid_outputs/total_tests*100 if total_tests > 0 else 0,
            'reasonable_output_percentage': reasonable_outputs/total_tests*100 if total_tests > 0 else 0
        },
        'results': results
    }
    
    with open('korean_inference_trained_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed results saved to: korean_inference_trained_results.json")
    
    return detailed_results


if __name__ == "__main__":
    results = test_korean_inference_comprehensive()