#!/usr/bin/env python3
"""
Final comprehensive validation for Korean-English translation system
Tests accuracy and generates detailed report
"""

import torch
from pathlib import Path
import json
import argparse
import numpy as np
from typing import List, Dict
import sentencepiece as spm
import re
import time
import os
import sys


def check_gpu_status():
    """Check GPU availability and status"""
    try:
        import subprocess

        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else "NVIDIA GPU not available"
    except:
        return "nvidia-smi not found"


def load_model_and_tokenizer(model_path: str, tokenizer_model: str, device: str):
    """Load model and tokenizer"""
    print(f"Loading model from {model_path}")
    print(f"Loading tokenizer from {tokenizer_model}")

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)
    vocab_size = tokenizer.get_piece_size()
    print(f"Vocabulary size: {vocab_size}")

    # Load model - try beam search decoder first
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
        from src.training.beam_search_decoder import TranslationInference

        inference = TranslationInference(model_path, tokenizer_model, device)
        print("Using beam search decoder for inference")
        return inference, tokenizer, True
    except Exception as e:
        print(f"Beam search decoder failed: {e}")
        print("Falling back to basic model loading...")

        # Basic model loading
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        from src.training.train_extended_nmt import TransformerNMT

        args = checkpoint.get("args", None)
        if args is not None:
            model = TransformerNMT(
                vocab_size=vocab_size,
                d_model=getattr(args, "d_model", 512),
                nhead=getattr(args, "nhead", 8),
                num_encoder_layers=getattr(args, "num_layers", 6),
                num_decoder_layers=getattr(args, "num_layers", 6),
                dim_feedforward=getattr(args, "dim_feedforward", 2048),
                dropout=getattr(args, "dropout", 0.1),
                max_len=getattr(args, "max_length", 128),
            ).to(device)
        else:
            model = TransformerNMT(vocab_size=vocab_size).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print("Using basic model for inference")
        return model, tokenizer, False


def translate_text(
    model_or_inference, text: str, tokenizer, use_beam_search: bool
) -> str:
    """Translate Korean text to English"""
    if use_beam_search:
        return model_or_inference.translate(text)
    else:
        # Basic translation
        model = model_or_inference
        device = next(model.parameters()).device

        # Tokenize input
        bos_id = tokenizer.piece_to_id("<s>")
        eos_id = tokenizer.piece_to_id("</s>")
        pad_id = tokenizer.piece_to_id("<pad>")

        tokens = [bos_id] + tokenizer.encode(text) + [eos_id]
        tokens = tokens[:128]  # Truncate
        length = len(tokens)
        padded = tokens + [pad_id] * (128 - length)

        # Convert to tensor
        input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)
        length_tensor = torch.tensor(length, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            # Generate translation (simplified)
            output_tokens = [bos_id]

            for _ in range(128):
                tgt_tensor = (
                    torch.tensor(output_tokens, dtype=torch.long)
                    .unsqueeze(0)
                    .to(device)
                )

                # Create masks
                src_mask = model.create_padding_mask(
                    length_tensor, input_tensor.size(1)
                )
                tgt_length = (
                    torch.tensor(len(output_tokens), dtype=torch.long)
                    .unsqueeze(0)
                    .to(device)
                )
                tgt_mask = model.create_padding_mask(tgt_length, tgt_tensor.size(1))

                # Forward pass
                output = model(
                    input_tensor,
                    tgt_tensor,
                    src_key_padding_mask=src_mask,
                    tgt_key_padding_mask=tgt_mask,
                    memory_key_padding_mask=src_mask,
                )

                # Get next token
                next_token = output[0, -1].argmax().item()
                output_tokens.append(next_token)

                if next_token == eos_id:
                    break

            # Decode
            output_tokens = [t for t in output_tokens if t not in [bos_id, eos_id]]
            return tokenizer.decode(output_tokens)


def calculate_bleu_score(reference: str, hypothesis: str) -> float:
    """Calculate simple BLEU score"""
    # Basic tokenization
    ref_words = re.findall(r"\w+", reference.lower())
    hyp_words = re.findall(r"\w+", hypothesis.lower())

    if not ref_words or not hyp_words:
        return 0.0

    # Calculate precision
    ref_set = set(ref_words)
    hyp_set = set(hyp_words)

    if not hyp_set:
        return 0.0

    precision = len(ref_set.intersection(hyp_set)) / len(hyp_set)

    # Brevity penalty
    bp = min(1.0, len(hyp_words) / len(ref_words)) if ref_words else 0.0

    return bp * precision


def calculate_character_accuracy(reference: str, hypothesis: str) -> float:
    """Calculate character-level accuracy"""
    ref_chars = list(reference.lower().strip())
    hyp_chars = list(hypothesis.lower().strip())

    if not ref_chars:
        return 1.0 if not hyp_chars else 0.0

    # Simple character overlap
    matches = 0
    total = len(ref_chars)

    for i, ref_char in enumerate(ref_chars):
        if i < len(hyp_chars) and ref_char == hyp_chars[i]:
            matches += 1

    return matches / total if total > 0 else 1.0


def run_final_validation(
    model_path: str,
    tokenizer_model: str,
    test_data: str,
    accuracy_threshold: float = 99.0,
    max_samples: int = None,
    search_strategy: str = "beam",
    beam_size: int = 5,
    length_penalty: float = 0.6,
):
    """Run final comprehensive validation"""

    print("=" * 60)
    print("Korean-English Translation Final Validation")
    print("=" * 60)

    # Check GPU status
    print(f"GPU Status:\n{check_gpu_status()}")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_or_inference, tokenizer, use_beam_search = load_model_and_tokenizer(
        model_path, tokenizer_model, device
    )

    # Load test data
    print(f"Loading test data from {test_data}")
    with open(test_data, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)

    if max_samples:
        test_pairs = test_pairs[:max_samples]

    print(f"Testing on {len(test_pairs)} samples")

    # Run validation
    results = []
    total_time = 0

    print("\nRunning validation...")
    for i, pair in enumerate(test_pairs):
        if i % 50 == 0:
            print(f"Processing sample {i+1}/{len(test_pairs)}")

        korean = pair["korean"]
        reference = pair["english"]

        # Translate
        start_time = time.time()
        if use_beam_search and hasattr(model_or_inference, "decoder"):
            model_or_inference.decoder.beam_size = beam_size
            model_or_inference.decoder.length_penalty = length_penalty
            if hasattr(model_or_inference.decoder, "strategy"):
                model_or_inference.decoder.strategy = search_strategy

        hypothesis = translate_text(
            model_or_inference, korean, tokenizer, use_beam_search
        )
        inference_time = time.time() - start_time
        total_time += inference_time

        # Calculate metrics
        bleu_score = calculate_bleu_score(reference, hypothesis)
        char_accuracy = calculate_character_accuracy(reference, hypothesis)
        exact_match = (
            1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0
        )

        results.append(
            {
                "korean": korean,
                "reference": reference,
                "hypothesis": hypothesis,
                "bleu_score": bleu_score,
                "character_accuracy": char_accuracy,
                "exact_match": exact_match,
                "inference_time": inference_time,
            }
        )

    # Calculate aggregate metrics
    bleu_scores = [r["bleu_score"] for r in results]
    char_accuracies = [r["character_accuracy"] for r in results]
    exact_matches = [r["exact_match"] for r in results]
    try:
        from src.utils.metrics import ROUGELScore, SimpleMETEOR

        rouge = ROUGELScore()(
            [r["reference"] for r in results], [r["hypothesis"] for r in results]
        )
        meteor = SimpleMETEOR()(
            [r["reference"] for r in results], [r["hypothesis"] for r in results]
        )
    except Exception:
        rouge = 0.0
        meteor = 0.0

    avg_bleu = np.mean(bleu_scores)
    avg_char_acc = np.mean(char_accuracies)
    exact_match_rate = np.mean(exact_matches)
    avg_inference_time = total_time / len(results)

    # Calculate overall translation accuracy (weighted combination)
    translation_accuracy = (
        0.6 * avg_char_acc + 0.3 * avg_bleu + 0.1 * exact_match_rate
    ) * 100

    # Check if meets threshold
    meets_threshold = translation_accuracy >= accuracy_threshold

    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total Samples: {len(results)}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average Character Accuracy: {avg_char_acc:.4f}")
    print(f"Exact Match Rate: {exact_match_rate:.4f}")
    print(f"Average ROUGE-L: {float(rouge):.4f}")
    print(f"Average METEOR: {float(meteor):.4f}")
    print(f"Average Inference Time: {avg_inference_time:.4f}s")
    print(f"Translation Accuracy: {translation_accuracy:.2f}%")
    print(f"Target Threshold: {accuracy_threshold}%")
    print(f"Meets Threshold: {'✓ YES' if meets_threshold else '✗ NO'}")

    # Show some examples
    print("\nSample Translations:")
    print("-" * 60)
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"Korean:    {r['korean']}")
        print(f"Reference: {r['reference']}")
        print(f"Hypothesis:{r['hypothesis']}")
        print(f"BLEU: {r['bleu_score']:.4f}, Char Acc: {r['character_accuracy']:.4f}")
        print("-" * 40)

    # Save detailed results
    output_file = "final_validation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        payload = {
            "summary": {
                "total_samples": int(len(results)),
                "translation_accuracy": float(translation_accuracy),
                "avg_bleu_score": float(avg_bleu),
                "avg_character_accuracy": float(avg_char_acc),
                "exact_match_rate": float(exact_match_rate),
                "avg_inference_time": float(avg_inference_time),
                "avg_rouge_l": float(rouge),
                "avg_meteor": float(meteor),
                "meets_threshold": bool(meets_threshold),
                "threshold": float(accuracy_threshold),
            },
            "detailed_results": results,
            "model_info": {
                "model_path": model_path,
                "tokenizer_model": tokenizer_model,
                "use_beam_search": use_beam_search,
            },
        }
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log_dir = Path("experiment_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload["summary"], f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to {output_file}")

    # Return success/failure
    if meets_threshold:
        print(
            f"\n🎉 SUCCESS: Translation system achieves {translation_accuracy:.2f}% accuracy!"
        )
        print(f"✓ Meets the {accuracy_threshold}% threshold requirement")
        return 0
    else:
        print(
            f"\n❌ FAILED: Translation accuracy {translation_accuracy:.2f}% is below {accuracy_threshold}% threshold"
        )
        print("Recommendations:")
        print("- Extend training with more epochs")
        print("- Increase model size or architecture complexity")
        print("- Improve data quality and diversity")
        print("- Implement ensemble methods or knowledge distillation")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Final Korean-English Translation Validation"
    )
    parser.add_argument("--model-path", default="models/extended/best_model.pt")
    parser.add_argument(
        "--tokenizer-model",
        default="data/processed_large_simple/sentencepiece_large.model",
    )
    parser.add_argument(
        "--test-data", default="data/processed_large_simple/test_data.json"
    )
    parser.add_argument("--accuracy-threshold", type=float, default=99.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--search-strategy",
        type=str,
        choices=["greedy", "beam", "diverse_beam"],
        default="beam",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at {args.model_path}")
        print("Please run extended training first:")
        print(
            "python src/training/train_extended_nmt.py --epochs 50 --quick-validation"
        )
        return 1

    # Run validation
    return run_final_validation(
        model_path=args.model_path,
        tokenizer_model=args.tokenizer_model,
        test_data=args.test_data,
        accuracy_threshold=args.accuracy_threshold,
        max_samples=args.max_samples,
        search_strategy=args.search_strategy,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
    )


if __name__ == "__main__":
    exit(main())
