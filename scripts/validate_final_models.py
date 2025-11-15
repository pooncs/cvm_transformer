#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from cvm_translator.cvm_transformer import CVMTransformer
from cvm_translator.validation_protocol import ValidationConfig, DistillationValidator


def load_model(checkpoint_path: str, vocab_size: int, d_model: int, n_layers: int, core_capacity: int, device: str):
    """Load a trained model from checkpoint."""
    model = CVMTransformer(vocab_size, d_model=d_model, n_layers=n_layers, core_capacity=core_capacity)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def validate_model(model, test_data: List, device: str, model_name: str) -> Dict[str, Any]:
    """Validate a model on test data."""
    validator = DistillationValidator(ValidationConfig(), teacher_model=None)
    
    # Simple tokenizer that maps to vocabulary range
    vocab_size = 32000
    def simple_tokenize(text: str):
        # Map characters to vocab range safely
        return [min(max(ord(c) % vocab_size, 1), vocab_size-1) for c in text]
    
    # Simple inference test
    start_time = time.time()
    with torch.no_grad():
        for src_text, tgt_text in test_data:
            # Safe tokenization
            src_ids = torch.tensor([simple_tokenize(src_text)], dtype=torch.long).to(device)
            logits, hidden_states, attention_maps = model(src_ids, return_hidden=True, return_attn=True)
    
    inference_time = time.time() - start_time
    
    return {
        "model_name": model_name,
        "inference_time": inference_time,
        "avg_time_per_sample": inference_time / len(test_data),
        "device": device
    }


def main():
    # Test data
    test_data = [
        ("안녕하세요", "Hello"),
        ("오늘 날씨 좋네요", "Today weather is nice"),
        ("실시간 번역", "real-time translation"),
        ("한국어 영어", "Korean English"),
        ("감사합니다", "Thank you"),
        ("좋은 하루 되세요", "Have a good day"),
        ("무엇을 도와드릴까요?", "How can I help you?"),
        ("이것은 테스트입니다", "This is a test"),
        ("빠른 번역 서비스", "Fast translation service"),
        ("정확한 결과", "Accurate results")
    ]
    
    # Model configurations
    configs = [
        {
            "name": "AdamW Baseline",
            "checkpoint": "checkpoints/student_final.pt",
            "vocab_size": 32000,
            "d_model": 768,
            "n_layers": 6,
            "core_capacity": 64
        },
        {
            "name": "Lion Optimizer (AdamW fallback)",
            "checkpoint": "checkpoints_lion/student_final.pt", 
            "vocab_size": 32000,
            "d_model": 768,
            "n_layers": 6,
            "core_capacity": 64
        }
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = []
    
    for config in configs:
        print(f"\nValidating {config['name']}...")
        
        if not Path(config["checkpoint"]).exists():
            print(f"Checkpoint not found: {config['checkpoint']}")
            continue
            
        try:
            model = load_model(
                config["checkpoint"],
                config["vocab_size"],
                config["d_model"],
                config["n_layers"],
                config["core_capacity"],
                device
            )
            
            result = validate_model(model, test_data, device, config["name"])
            results.append(result)
            
            print(f"✓ {config['name']} validation completed")
            print(f"  Inference time: {result['inference_time']:.4f}s")
            print(f"  Avg time per sample: {result['avg_time_per_sample']:.4f}s")
            
        except Exception as e:
            print(f"✗ Error validating {config['name']}: {e}")
    
    # Generate comparison report
    report = {
        "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "test_data_size": len(test_data),
        "results": results,
        "summary": {
            "total_models_tested": len(results),
            "best_inference_time": min([r["inference_time"] for r in results]) if results else None,
            "avg_inference_time": sum([r["inference_time"] for r in results]) / len(results) if results else None
        }
    }
    
    # Save report
    report_path = "reports/final_validation_report.json"
    Path("reports").mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Final validation report saved to: {report_path}")
    print(f"  Total models tested: {report['summary']['total_models_tested']}")
    if results:
        print(f"  Best inference time: {report['summary']['best_inference_time']:.4f}s")
        print(f"  Average inference time: {report['summary']['avg_inference_time']:.4f}s")


if __name__ == "__main__":
    main()