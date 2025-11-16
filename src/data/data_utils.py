#!/usr/bin/env python3
"""
Data preparation utilities for Korean-English translation
"""

import pandas as pd
import json
import argparse
import os
from typing import List, Dict, Tuple


def convert_tsv_to_json(tsv_file: str, json_file: str, max_samples: int = None) -> int:
    """Convert TSV file to JSON format"""
    print(f"Converting {tsv_file} to {json_file}")

    # Read TSV file
    df = pd.read_csv(tsv_file, sep="\t", header=None, names=["korean", "english"])

    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)

    # Convert to list of dictionaries
    data = []
    for _, row in df.iterrows():
        data.append(
            {
                "korean": str(row["korean"]).strip(),
                "english": str(row["english"]).strip(),
            }
        )

    # Save as JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(data)} samples to JSON format")
    return len(data)


def split_data(
    json_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """Split JSON data into train/validation/test sets"""
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Shuffle data
    np.random.seed(seed)
    np.random.shuffle(data)

    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    # Save splits
    base_dir = os.path.dirname(json_file)
    base_name = os.path.basename(json_file).replace(".json", "")

    train_file = os.path.join(base_dir, f"{base_name}_train.json")
    val_file = os.path.join(base_dir, f"{base_name}_val.json")
    test_file = os.path.join(base_dir, f"{base_name}_test.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Data split completed:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    return len(train_data), len(val_data), len(test_data)


def validate_data_quality(json_file: str) -> Dict[str, any]:
    """Validate data quality and statistics"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    korean_lengths = [len(item["korean"]) for item in data]
    english_lengths = [len(item["english"]) for item in data]

    stats = {
        "total_samples": len(data),
        "korean_avg_length": np.mean(korean_lengths),
        "korean_max_length": max(korean_lengths),
        "korean_min_length": min(korean_lengths),
        "english_avg_length": np.mean(english_lengths),
        "english_max_length": max(english_lengths),
        "english_min_length": min(english_lengths),
        "empty_korean": sum(1 for item in data if not item["korean"].strip()),
        "empty_english": sum(1 for item in data if not item["english"].strip()),
        "ratio_kor_eng": (
            np.mean(korean_lengths) / np.mean(english_lengths) if english_lengths else 0
        ),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Data preparation utilities")
    parser.add_argument("--tsv-file", help="Input TSV file")
    parser.add_argument("--json-file", help="Output JSON file")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to process")
    parser.add_argument(
        "--split", action="store_true", help="Split data into train/val/test"
    )
    parser.add_argument("--validate", action="store_true", help="Validate data quality")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")

    args = parser.parse_args()

    if args.tsv_file and args.json_file:
        # Convert TSV to JSON
        convert_tsv_to_json(args.tsv_file, args.json_file, args.max_samples)

        if args.split:
            split_data(
                args.json_file, args.train_ratio, args.val_ratio, args.test_ratio
            )

        if args.validate:
            stats = validate_data_quality(args.json_file)
            print("\nData Quality Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    elif args.validate and args.json_file:
        # Just validate
        stats = validate_data_quality(args.json_file)
        print("\nData Quality Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    else:
        print(
            "Please specify --tsv-file and --json-file for conversion, or --json-file for validation"
        )


if __name__ == "__main__":
    import numpy as np

    main()
