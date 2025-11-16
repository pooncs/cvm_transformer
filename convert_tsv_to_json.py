#!/usr/bin/env python3
"""
Convert TSV data to JSON format for training
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_utils import convert_tsv_to_json
import argparse

# Script functionality is now in src/data/data_utils.py

def main():
    parser = argparse.ArgumentParser(description='Convert TSV to JSON')
    parser.add_argument('--train-tsv', default='data/processed_large_simple/train.tsv')
    parser.add_argument('--val-tsv', default='data/processed_large_simple/val.tsv')
    parser.add_argument('--test-tsv', default='data/processed_large_simple/test.tsv')
    parser.add_argument('--train-json', default='data/processed_large_simple/train_data.json')
    parser.add_argument('--val-json', default='data/processed_large_simple/val_data.json')
    parser.add_argument('--test-json', default='data/processed_large_simple/test_data.json')
    parser.add_argument('--max-train-samples', type=int, default=None)
    args = parser.parse_args()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(args.train_json), exist_ok=True)
    
    # Convert files
    train_count = convert_tsv_to_json(args.train_tsv, args.train_json, args.max_train_samples)
    val_count = convert_tsv_to_json(args.val_tsv, args.val_json)
    test_count = convert_tsv_to_json(args.test_tsv, args.test_json)
    
    print(f"\nConversion completed:")
    print(f"Train: {train_count} samples")
    print(f"Validation: {val_count} samples")
    print(f"Test: {test_count} samples")

if __name__ == '__main__':
    main()