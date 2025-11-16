#!/usr/bin/env python3
"""
Simple data preparation script for Korean-English translation
"""
import pandas as pd
import sentencepiece as spm
import os
import json
from pathlib import Path

def prepare_data_simple():
    """Prepare data with basic tokenization"""
    
    # Load parallel corpus
    df = pd.read_csv('data/raw/korean_english_parallel.tsv', sep='\t')
    print(f"Loaded {len(df)} sentence pairs")
    
    # Save train/validation splits
    train_size = int(len(df) * 0.9)
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    # Save splits
    train_df.to_csv('data/processed/train.tsv', sep='\t', index=False)
    val_df.to_csv('data/processed/val.tsv', sep='\t', index=False)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Create combined text for tokenizer training
    combined_text = []
    for _, row in df.iterrows():
        combined_text.append(str(row['korean']))
        combined_text.append(str(row['english']))
    
    # Save combined text
    with open('data/processed/combined.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_text))
    
    # Train SentencePiece tokenizer
    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        input='data/processed/combined.txt',
        model_prefix='data/tokenizers/kr_en_simple',
        vocab_size=600,  # Reduced for small dataset
        character_coverage=0.995,
        model_type='bpe'
    )
    
    # Load tokenizer and test
    sp = spm.SentencePieceProcessor(model_file='data/tokenizers/kr_en_simple.model')
    
    # Test tokenization
    test_sentences = [
        "안녕하세요",
        "Hello world",
        "감사합니다",
        "Thank you"
    ]
    
    print("\nTokenizer test:")
    for sent in test_sentences:
        pieces = sp.encode_as_pieces(sent)
        ids = sp.encode_as_ids(sent)
        print(f"'{sent}' -> {pieces} -> {ids}")
    
    print(f"\nVocab size: {sp.get_piece_size()}")
    
    # Create vocabulary mapping
    vocab = {}
    for i in range(sp.get_piece_size()):
        vocab[sp.id_to_piece(i)] = i
    
    with open('data/tokenizers/kr_en_simple_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print("Data preparation completed!")
    return {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'vocab_size': sp.get_piece_size(),
        'tokenizer_path': 'data/tokenizers/kr_en_simple.model'
    }

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/tokenizers', exist_ok=True)
    
    stats = prepare_data_simple()
    print(f"\nPreparation stats: {stats}")