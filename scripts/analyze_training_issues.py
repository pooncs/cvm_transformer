#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any

from cvm_translator.cvm_transformer import CVMTransformer


def analyze_model_outputs(checkpoint_path: str, vocab_size: int, d_model: int, n_layers: int, core_capacity: int, device: str):
    """Analyze model outputs to understand the training issues."""
    
    print(f"Analyzing model: {checkpoint_path}")
    
    # Load model
    model = CVMTransformer(vocab_size, d_model=d_model, n_layers=n_layers, core_capacity=core_capacity)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    # Test inputs
    test_korean = ["안녕하세요", "감사합니다", "오늘 날씨 좋네요"]
    
    print(f"\nModel Architecture Analysis:")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Model dimensions: {d_model}")
    print(f"- Number of layers: {n_layers}")
    print(f"- Core capacity: {core_capacity}")
    
    print(f"\nTraining Analysis:")
    print(f"- Training iterations: {checkpoint.get('iters', 'Unknown')}")
    
    analysis_results = []
    
    for korean_text in test_korean:
        print(f"\n{'='*50}")
        print(f"Testing: '{korean_text}'")
        print(f"{'='*50}")
        
        # Simple tokenization
        input_ids = [min(max(ord(c) % (vocab_size - 1) + 1, 1), vocab_size - 1) for c in korean_text]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits, hidden_states, attention_maps = model(input_tensor, return_hidden=True, return_attn=True)
            
            # Analyze logits
            probs = torch.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
            
            print(f"Input shape: {input_tensor.shape}")
            print(f"Output shape: {logits.shape}")
            print(f"Hidden states shape: {hidden_states.shape}")
            print(f"Attention maps shape: {attention_maps.shape}")
            
            # Logit analysis
            print(f"\nLogit Statistics:")
            print(f"- Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            print(f"- Logits mean: {logits.mean().item():.3f}")
            print(f"- Logits std: {logits.std().item():.3f}")
            
            # Probability analysis
            print(f"\nProbability Statistics:")
            print(f"- Max prob per position: {probs.max(dim=-1)[0].mean().item():.3f}")
            print(f"- Entropy per position: {-torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item():.3f}")
            
            # Vocabulary usage analysis
            vocab_usage = len(np.unique(predicted_ids))
            print(f"\nVocabulary Usage:")
            print(f"- Unique tokens used: {vocab_usage}/{vocab_size}")
            print(f"- Vocabulary coverage: {vocab_usage/vocab_size*100:.2f}%")
            
            # Predicted token analysis
            print(f"\nPredicted Tokens:")
            print(f"- Predicted IDs: {predicted_ids}")
            print(f"- Most frequent predicted ID: {np.bincount(predicted_ids).argmax()}")
            print(f"- Frequency of most common token: {np.bincount(predicted_ids).max()}/{len(predicted_ids)}")
            
            # Check for repetitive patterns
            if len(predicted_ids) > 1:
                # Check if all tokens are the same
                all_same = np.all(predicted_ids == predicted_ids[0])
                print(f"- All tokens identical: {all_same}")
                
                # Check for repetitive sequences
                unique_tokens = len(np.unique(predicted_ids))
                print(f"- Token diversity: {unique_tokens}/{len(predicted_ids)}")
            
            # Attention analysis
            print(f"\nAttention Analysis:")
            print(f"- Attention weights range: [{attention_maps.min().item():.3f}, {attention_maps.max().item():.3f}]")
            print(f"- Attention entropy: {-torch.sum(attention_maps * torch.log(attention_maps + 1e-8), dim=-1).mean().item():.3f}")
            
            analysis_results.append({
                "input": korean_text,
                "input_ids": input_ids,
                "predicted_ids": predicted_ids.tolist(),
                "logits_stats": {
                    "min": logits.min().item(),
                    "max": logits.max().item(),
                    "mean": logits.mean().item(),
                    "std": logits.std().item()
                },
                "vocab_usage": vocab_usage,
                "all_tokens_identical": bool(all_same) if len(predicted_ids) > 1 else False,
                "token_diversity": unique_tokens if len(predicted_ids) > 1 else 1
            })
    
    return analysis_results


def identify_issues(analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify specific training issues from analysis."""
    
    issues = {
        "repetitive_outputs": False,
        "vocab_underutilization": False,
        "logit_saturation": False,
        "attention_issues": False,
        "tokenization_problems": False
    }
    
    recommendations = []
    
    # Check for repetitive outputs
    identical_rates = [r["all_tokens_identical"] for r in analysis_results if "all_tokens_identical" in r]
    if identical_rates and sum(identical_rates) / len(identical_rates) > 0.5:
        issues["repetitive_outputs"] = True
        recommendations.append("🔄 Model is producing repetitive outputs - consider increasing model diversity")
    
    # Check vocabulary usage
    vocab_usages = [r["vocab_usage"] for r in analysis_results]
    avg_vocab_usage = sum(vocab_usages) / len(vocab_usages)
    vocab_utilization_rate = avg_vocab_usage / 32000  # Assuming vocab_size=32000
    
    if vocab_utilization_rate < 0.01:  # Less than 1% vocabulary used
        issues["vocab_underutilization"] = True
        recommendations.append("📚 Very low vocabulary utilization - improve tokenization or training data")
    
    # Check logit statistics
    logit_means = [r["logits_stats"]["mean"] for r in analysis_results]
    avg_logit_mean = sum(logit_means) / len(logit_means)
    
    if abs(avg_logit_mean) > 10:  # Very high logit values
        issues["logit_saturation"] = True
        recommendations.append("📊 Logit saturation detected - consider gradient clipping or learning rate adjustment")
    
    # Check token diversity
    diversities = [r["token_diversity"] for r in analysis_results]
    avg_diversity = sum(diversities) / len(diversities)
    
    if avg_diversity < 2:  # Very low token diversity
        issues["tokenization_problems"] = True
        recommendations.append("🔤 Low token diversity - current tokenization is too simplistic")
    
    return {
        "issues": issues,
        "recommendations": recommendations,
        "statistics": {
            "avg_vocab_usage": avg_vocab_usage,
            "vocab_utilization_rate": vocab_utilization_rate,
            "avg_logit_mean": avg_logit_mean,
            "avg_token_diversity": avg_diversity
        }
    }


def main():
    print("🔍 CVM TRANSFORMER TRAINING ISSUE ANALYSIS")
    print("="*60)
    
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
    
    all_analyses = []
    
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {config['name']}")
        print(f"{'='*60}")
        
        if not Path(config["checkpoint"]).exists():
            print(f"Checkpoint not found: {config['checkpoint']}")
            continue
        
        try:
            analysis = analyze_model_outputs(
                config["checkpoint"],
                config["vocab_size"],
                config["d_model"],
                config["n_layers"],
                config["core_capacity"],
                device
            )
            
            issues = identify_issues(analysis)
            
            print(f"\n{'='*40}")
            print(f"ISSUE ANALYSIS FOR {config['name']}")
            print(f"{'='*40}")
            
            for issue, present in issues["issues"].items():
                status = "❌ DETECTED" if present else "✅ OK"
                print(f"{issue.replace('_', ' ').title()}: {status}")
            
            print(f"\n📊 Key Statistics:")
            print(f"- Average vocabulary usage: {issues['statistics']['avg_vocab_usage']:.0f} tokens")
            print(f"- Vocabulary utilization: {issues['statistics']['vocab_utilization_rate']*100:.3f}%")
            print(f"- Average logit mean: {issues['statistics']['avg_logit_mean']:.3f}")
            print(f"- Average token diversity: {issues['statistics']['avg_token_diversity']:.1f}")
            
            if issues["recommendations"]:
                print(f"\n💡 Recommendations:")
                for rec in issues["recommendations"]:
                    print(f"  {rec}")
            else:
                print(f"\n✅ No major issues detected!")
            
            all_analyses.append({
                "model_name": config["name"],
                "analysis": analysis,
                "issues": issues
            })
            
        except Exception as e:
            print(f"Error analyzing {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comprehensive analysis report
    report = {
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "models_analyzed": len(all_analyses),
        "analyses": all_analyses,
        "overall_recommendations": []
    }
    
    # Overall analysis
    if all_analyses:
        print(f"\n{'='*60}")
        print(f"OVERALL ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Common issues across models
        common_issues = {}
        for analysis in all_analyses:
            for issue, present in analysis["issues"]["issues"].items():
                if present:
                    common_issues[issue] = common_issues.get(issue, 0) + 1
        
        if common_issues:
            print(f"\n🚨 Common Issues Across Models:")
            for issue, count in common_issues.items():
                percentage = count / len(all_analyses) * 100
                print(f"  {issue.replace('_', ' ').title()}: {count}/{len(all_analyses)} models ({percentage:.0f}%)")
        
        # Generate strategic recommendations
        print(f"\n🎯 Strategic Improvement Recommendations:")
        
        recommendations = [
            "1. 🔄 TOKENIZATION OVERHAUL",
            "   • Implement proper Korean-English subword tokenization (BPE/SentencePiece)",
            "   • Create aligned vocabulary mappings between languages",
            "   • Add special tokens for language identification",
            "",
            "2. 📊 TRAINING DATA ENHANCEMENT",
            "   • Expand training dataset with diverse Korean-English pairs",
            "   • Include domain-specific vocabulary (technical, casual, formal)",
            "   • Add data augmentation techniques for better generalization",
            "",
            "3. 🧠 MODEL ARCHITECTURE IMPROVEMENTS",
            "   • Increase model capacity (12+ layers, 1024+ dimensions)",
            "   • Implement proper cross-attention mechanisms",
            "   • Add language-specific encoding layers",
            "",
            "4. ⚡ TRAINING STRATEGY OPTIMIZATION",
            "   • Implement curriculum learning (easy→hard examples)",
            "   • Use advanced optimizers (Lion, Adafactor) with proper hyperparameters",
            "   • Apply gradient clipping and learning rate scheduling",
            "",
            "5. 🎯 KNOWLEDGE DISTILLATION ENHANCEMENT",
            "   • Improve teacher-student alignment with attention transfer",
            "   • Implement progressive distillation (gradual capacity reduction)",
            "   • Add consistency regularization techniques",
            "",
            "6. 🔍 VALIDATION & MONITORING",
            "   • Implement BLEU/ROUGE/BERTScore metrics during training",
            "   • Add attention visualization for quality monitoring",
            "   • Create human evaluation protocols"
        ]
        
        for rec in recommendations:
            print(rec)
        
        report["overall_recommendations"] = recommendations
    
    # Save analysis report
    report_path = "reports/training_analysis_report.json"
    Path("reports").mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed analysis report saved to: {report_path}")
    print(f"   Total models analyzed: {len(all_analyses)}")
    print(f"   Analysis completed at: {report['analysis_timestamp']}")


if __name__ == "__main__":
    main()