#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

from cvm_translator.cvm_transformer import CVMTransformer


def load_model(checkpoint_path: str, vocab_size: int, d_model: int, n_layers: int, core_capacity: int, device: str):
    """Load a trained model from checkpoint."""
    model = CVMTransformer(vocab_size, d_model=d_model, n_layers=n_layers, core_capacity=core_capacity)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def create_tokenizer(vocab_size: int = 32000):
    """Create a simple character-level tokenizer."""
    def tokenize(text: str) -> List[int]:
        # Map characters to vocabulary range safely
        return [min(max(ord(c) % (vocab_size - 1) + 1, 1), vocab_size - 1) for c in text]
    
    def detokenize(ids: List[int]) -> str:
        # Simple reverse mapping (approximate)
        return ''.join([chr(max(1, min(id_, 1000))) for id_ in ids if id_ > 0])
    
    return tokenize, detokenize


def translate_text(model, text: str, tokenize, detokenize, device: str, max_length: int = 64) -> Tuple[str, float]:
    """Translate text using the model."""
    start_time = time.time()
    
    with torch.no_grad():
        # Tokenize input
        input_ids = tokenize(text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # Generate translation
        logits, hidden_states, attention_maps = model(input_tensor, return_hidden=True, return_attn=True)
        
        # Get predicted tokens (greedy decoding)
        predicted_ids = torch.argmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
        
        # Handle both single token and multiple tokens cases
        if isinstance(predicted_ids, int):
            predicted_ids = [predicted_ids]
        elif not isinstance(predicted_ids, list):
            predicted_ids = list(predicted_ids)
        
        # Detokenize
        translated_text = detokenize(predicted_ids)
    
    inference_time = time.time() - start_time
    return translated_text, inference_time


def evaluate_translation_quality(source: str, predicted: str, reference: str) -> Dict[str, Any]:
    """Evaluate translation quality with multiple metrics."""
    
    # Basic metrics
    predicted_words = predicted.split()
    reference_words = reference.split()
    
    # Word-level accuracy (simple)
    common_words = set(predicted_words) & set(reference_words)
    word_accuracy = len(common_words) / max(len(set(reference_words)), 1)
    
    # Character-level similarity
    pred_chars = set(predicted.lower())
    ref_chars = set(reference.lower())
    char_overlap = len(pred_chars & ref_chars) / max(len(ref_chars), 1)
    
    # Length ratio
    length_ratio = len(predicted) / max(len(reference), 1)
    
    # Exact match check
    exact_match = predicted.strip().lower() == reference.strip().lower()
    
    return {
        "source": source,
        "predicted": predicted,
        "reference": reference,
        "word_accuracy": word_accuracy,
        "char_overlap": char_overlap,
        "length_ratio": length_ratio,
        "exact_match": exact_match,
        "predicted_length": len(predicted),
        "reference_length": len(reference)
    }


def main():
    # Test data with diverse Korean sentences
    test_data = [
        ("안녕하세요", "Hello"),
        ("감사합니다", "Thank you"),
        ("오늘 날씨 좋네요", "Today weather is nice"),
        ("실시간 번역", "real-time translation"),
        ("한국어 영어", "Korean English"),
        ("좋은 하루 되세요", "Have a good day"),
        ("무엇을 도와드릴까요?", "How can I help you?"),
        ("이것은 테스트입니다", "This is a test"),
        ("빠른 번역 서비스", "Fast translation service"),
        ("정확한 결과", "Accurate results"),
        ("안녕히 가세요", "Goodbye"),
        ("잘 지내세요?", "How are you?"),
        ("만나서 반갑습니다", "Nice to meet you"),
        ("죄송합니다", "Sorry"),
        ("괜찮아요", "It's okay"),
        ("사랑해요", "I love you"),
        ("축하합니다", "Congratulations"),
        ("행운을 빌어요", "Good luck"),
        ("건강하세요", "Stay healthy"),
        ("즐거운 시간 되세요", "Have a good time")
    ]
    
    # Model configurations
    model_configs = [
        {
            "name": "AdamW Baseline",
            "checkpoint": "checkpoints/student_final.pt",
            "vocab_size": 32000,
            "d_model": 768,
            "n_layers": 6,
            "core_capacity": 64
        },
        {
            "name": "Lion Optimizer",
            "checkpoint": "checkpoints_lion/student_final.pt", 
            "vocab_size": 32000,
            "d_model": 768,
            "n_layers": 6,
            "core_capacity": 64
        }
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    all_results = []
    
    for model_config in model_configs:
        print(f"\n{'='*60}")
        print(f"Testing {model_config['name']}")
        print(f"{'='*60}")
        
        if not Path(model_config["checkpoint"]).exists():
            print(f"Checkpoint not found: {model_config['checkpoint']}")
            continue
        
        try:
            # Load model
            model = load_model(
                model_config["checkpoint"],
                model_config["vocab_size"],
                model_config["d_model"],
                model_config["n_layers"],
                model_config["core_capacity"],
                device
            )
            
            # Create tokenizer
            tokenize, detokenize = create_tokenizer(model_config["vocab_size"])
            
            # Test translations
            model_results = []
            total_inference_time = 0
            
            print(f"Testing {len(test_data)} translation pairs...")
            
            for i, (korean_text, english_reference) in enumerate(test_data, 1):
                # Translate
                predicted_translation, inference_time = translate_text(
                    model, korean_text, tokenize, detokenize, device
                )
                
                total_inference_time += inference_time
                
                # Evaluate quality
                quality_metrics = evaluate_translation_quality(
                    korean_text, predicted_translation, english_reference
                )
                
                model_results.append({
                    "test_id": i,
                    "translation": quality_metrics,
                    "inference_time": inference_time
                })
                
                # Print detailed results for first 5 examples
                if i <= 5:
                    print(f"\n--- Test {i} ---")
                    print(f"Korean:    {korean_text}")
                    print(f"Predicted: {predicted_translation}")
                    print(f"Reference: {english_reference}")
                    print(f"Word Acc:  {quality_metrics['word_accuracy']:.3f}")
                    print(f"Char Overlap: {quality_metrics['char_overlap']:.3f}")
                    print(f"Inference: {inference_time*1000:.2f}ms")
            
            # Calculate overall statistics
            avg_inference_time = total_inference_time / len(test_data)
            avg_word_accuracy = sum([r["translation"]["word_accuracy"] for r in model_results]) / len(model_results)
            avg_char_overlap = sum([r["translation"]["char_overlap"] for r in model_results]) / len(model_results)
            exact_matches = sum([r["translation"]["exact_match"] for r in model_results])
            
            model_summary = {
                "model_name": model_config["name"],
                "total_tests": len(test_data),
                "avg_inference_time_ms": avg_inference_time * 1000,
                "avg_word_accuracy": avg_word_accuracy,
                "avg_char_overlap": avg_char_overlap,
                "exact_matches": exact_matches,
                "exact_match_rate": exact_matches / len(test_data),
                "detailed_results": model_results
            }
            
            all_results.append(model_summary)
            
            print(f"\n{'='*40}")
            print(f"SUMMARY FOR {model_config['name'].upper()}")
            print(f"{'='*40}")
            print(f"Average inference time: {avg_inference_time*1000:.2f}ms")
            print(f"Average word accuracy: {avg_word_accuracy:.3f}")
            print(f"Average char overlap: {avg_char_overlap:.3f}")
            print(f"Exact matches: {exact_matches}/{len(test_data)} ({exact_matches/len(test_data)*100:.1f}%)")
            
        except Exception as e:
            print(f"Error testing {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comprehensive report
    report = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "total_test_pairs": len(test_data),
        "models_tested": len(all_results),
        "test_data": test_data,
        "model_results": all_results,
        "comparison": {
            "best_inference_time": min([r["avg_inference_time_ms"] for r in all_results]) if all_results else None,
            "best_word_accuracy": max([r["avg_word_accuracy"] for r in all_results]) if all_results else None,
            "best_char_overlap": max([r["avg_char_overlap"] for r in all_results]) if all_results else None
        }
    }
    
    # Save detailed report
    report_path = "reports/inference_quality_report.json"
    Path("reports").mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE INFERENCE QUALITY REPORT")
    print(f"{'='*60}")
    print(f"Report saved to: {report_path}")
    print(f"Total models tested: {len(all_results)}")
    print(f"Total test pairs: {len(test_data)}")
    
    if all_results:
        print(f"\nCOMPARISON SUMMARY:")
        print(f"Best inference time: {report['comparison']['best_inference_time']:.2f}ms")
        print(f"Best word accuracy: {report['comparison']['best_word_accuracy']:.3f}")
        print(f"Best char overlap: {report['comparison']['best_char_overlap']:.3f}")
    
    # Quality analysis and recommendations
    print(f"\n{'='*60}")
    print(f"QUALITY ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*60}")
    
    for model_result in all_results:
        print(f"\n{model_result['model_name']}:")
        
        if model_result['avg_word_accuracy'] < 0.3:
            print("  ⚠️  Low word accuracy - Consider:")
            print("     • Increasing model capacity (layers/dimensions)")
            print("     • Longer training with more diverse data")
            print("     • Better tokenization strategy")
        
        if model_result['avg_char_overlap'] < 0.5:
            print("  ⚠️  Low character overlap - Consider:")
            print("     • Improving vocabulary mapping")
            print("     • Better alignment between Korean/English")
            print("     • Attention mechanism optimization")
        
        if model_result['exact_match_rate'] < 0.1:
            print("  ⚠️  Few exact matches - Consider:")
            print("     • More training data")
            print("     • Knowledge distillation improvements")
            print("     • Hyperparameter tuning")
        
        if model_result['avg_inference_time_ms'] > 50:
            print("  ⚠️  Slow inference - Consider:")
            print("     • Model quantization")
            print("     • FlashAttention integration")
            print("     • Edge optimization")
        
        if all(metric >= 0.3 for metric in [model_result['avg_word_accuracy'], model_result['avg_char_overlap']]):
            print("  ✅ Good overall performance")


if __name__ == "__main__":
    main()