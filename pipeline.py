#!/usr/bin/env python3
"""
Korean-English Translation Pipeline Main Entry Point
Orchestrates the entire translation pipeline from data preparation to validation
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print("=" * 60)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print(result.stdout)
    return True


def main():
    parser = argparse.ArgumentParser(description="Korean-English Translation Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "data", "train", "validate", "test", "multimodal"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "image", "audio", "multimodal"],
        default="text",
        help="Translation mode (text-only, image-to-text, audio-to-text, or multimodal)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file path for image/audio/multimodal translation",
    )
    parser.add_argument("--input-text", type=str, help="Text input for translation")
    parser.add_argument(
        "--data-size", type=int, default=50000, help="Number of training samples"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--accuracy-threshold", type=float, default=99.0, help="Accuracy threshold"
    )
    parser.add_argument("--quick", action="store_true", help="Quick validation mode")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    print("Korean-English Translation Pipeline")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Stage: {args.stage}")
    print(f"Data Size: {args.data_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("=" * 60)

    success = True

    if args.stage in ["all", "data"]:
        # Data preparation
        print("\n1. Data Preparation Stage")
        print("-" * 40)

        # Generate large corpus if needed
        if not os.path.exists("data/processed_large_simple/train.tsv"):
            success &= run_command(
                f"python src/data/generate_large_corpus.py --num-samples {args.data_size}",
                "Generating large corpus",
            )

        # Prepare corpus
        success &= run_command(
            "python src/data/prepare_large_corpus_simple.py",
            "Preparing large corpus with tokenization",
        )

        # Convert to JSON
        success &= run_command(
            "python convert_tsv_to_json.py", "Converting TSV to JSON format"
        )

        if not success:
            print("❌ Data preparation failed!")
            return 1

    if args.stage in ["all", "train"]:
        # Training
        print("\n2. Training Stage")
        print("-" * 40)

        train_cmd = f"python src/training/train_extended_nmt.py"
        train_cmd += f" --epochs {args.epochs}"
        train_cmd += f" --batch-size {args.batch_size}"
        train_cmd += f" --learning-rate {args.learning_rate}"
        train_cmd += f" --d-model 768 --nhead 12 --num-layers 8 --dim-feedforward 3072 --dropout 0.15"

        if args.quick:
            train_cmd += " --quick-validation"

        success &= run_command(
            train_cmd, "Extended training with advanced configuration"
        )

        if not success:
            print("❌ Training failed!")
            return 1

    if args.stage in ["all", "validate"]:
        # Validation
        print("\n3. Validation Stage")
        print("-" * 40)

        # Check if model exists
        model_path = "models/extended/best_model.pt"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Please run training first")
            return 1

        # Run comprehensive validation
        val_cmd = f"python src/evaluation/final_validation.py"
        val_cmd += f" --model-path {model_path}"
        val_cmd += f" --accuracy-threshold {args.accuracy_threshold}"

        if args.quick:
            val_cmd += " --max-samples 100"

        success &= run_command(val_cmd, "Final validation with accuracy testing")

        if not success:
            print(f"❌ Validation failed - accuracy below {args.accuracy_threshold}%")
            return 1

    if args.stage in ["all", "test"]:
        # Test inference
        print("\n4. Inference Testing")
        print("-" * 40)

        # Test some sample translations
        test_samples = [
            "안녕하세요",
            "오늘 날씨가 좋네요",
            "한국어를 영어로 번역합니다",
            "이것은 테스트 문장입니다",
        ]

        print("Testing sample translations:")
        for korean in test_samples:
            print(f"Korean: {korean}")
            # This would need a proper inference script
            print("Translation: [Run inference script to see results]")
            print("-" * 30)

    if args.stage == "multimodal":
        # Multimodal translation
        print("\n5. Multimodal Translation")
        print("-" * 40)
        print(f"Mode: {args.mode}")

        # Import multimodal processor
        try:
            from src.multimodal.multimodal_processor import (
                MultimodalTranslationProcessor,
            )

            processor = MultimodalTranslationProcessor(device=args.device)

            if args.mode == "text" and args.input_text:
                results = processor.multimodal_translate(text_input=args.input_text)
            elif args.mode == "image" and args.input_file:
                results = processor.multimodal_translate(image_path=args.input_file)
            elif args.mode == "audio" and args.input_file:
                results = processor.multimodal_translate(audio_path=args.input_file)
            elif args.mode == "multimodal":
                results = processor.multimodal_translate(
                    text_input=args.input_text,
                    image_path=args.input_file if args.input_file else None,
                )
            else:
                print("❌ Please provide appropriate input for the selected mode")
                return 1

            # Display results
            print(f"Status: {results['status']}")
            print(f"Input Modalities: {results['input_modalities']}")
            print(f"Extracted Text: {results['extracted_text']}")
            print(f"Translation: {results['translation']}")
            print(f"Confidence: {results['confidence']:.2f}")

            if "error" in results:
                print(f"Error: {results['error']}")
                return 1

        except ImportError as e:
            print(f"❌ Multimodal processing not available: {e}")
            return 1

    if success:
        print("\n🎉 Pipeline completed successfully!")
        print(
            f"✓ Translation system ready with {args.accuracy_threshold}% accuracy target"
        )
        return 0
    else:
        print("\n❌ Pipeline failed!")
        return 1


if __name__ == "__main__":
    exit(main())
