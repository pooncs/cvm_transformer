#!/usr/bin/env python3
"""
Convert TSV data to JSON format for training
"""
import pandas as pd
import json

def convert_tsv_to_json():
    """Convert TSV files to JSON format"""
    
    # Load train and validation data
    train_df = pd.read_csv('data/processed/train.tsv', sep='\t')
    val_df = pd.read_csv('data/processed/val.tsv', sep='\t')
    
    # Convert to JSON format
    train_data = []
    for _, row in train_df.iterrows():
        train_data.append({
            'korean': str(row['korean']),
            'english': str(row['english'])
        })
    
    val_data = []
    for _, row in val_df.iterrows():
        val_data.append({
            'korean': str(row['korean']),
            'english': str(row['english'])
        })
    
    # Save JSON files
    with open('data/processed/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open('data/processed/val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(train_data)} training samples and {len(val_data)} validation samples to JSON")
    print(f"Train JSON saved to: data/processed/train.json")
    print(f"Val JSON saved to: data/processed/val.json")

if __name__ == "__main__":
    convert_tsv_to_json()