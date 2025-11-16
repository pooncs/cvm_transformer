"""
Simple corpus preparation for large Korean-English dataset.
Focuses on tokenization without complex alignment validation.
"""

import json
import pandas as pd
from pathlib import Path
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_large_corpus_simple(
    input_file: str = "data/raw/korean_english_large.tsv",
    output_dir: str = "data/processed_large_simple",
    vocab_size: int = 3000,  # Reduced to match dataset size
    test_size: float = 0.02,
    val_size: float = 0.02
):
    """Prepare large corpus with simple processing."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the large corpus
    logger.info(f"Loading corpus from {input_file}")
    df = pd.read_csv(input_file, sep='\t', encoding='utf-8')
    logger.info(f"Loaded {len(df)} sentence pairs")
    
    # Basic cleaning - remove empty rows
    df = df.dropna()
    df = df[(df['korean'].str.len() > 2) & (df['english'].str.len() > 2)]
    logger.info(f"After basic cleaning: {len(df)} sentence pairs")
    
    # Add domain column if not present
    if 'domain' not in df.columns:
        df['domain'] = 'general'
    
    # Split into train/val/test
    logger.info("Splitting datasets")
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=42, shuffle=True)
    
    val_ratio = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_ratio, random_state=42, shuffle=True)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save datasets
    for split, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        # Save as TSV
        tsv_file = output_path / f"{split}.tsv"
        split_df.to_csv(tsv_file, sep='\t', index=False, encoding='utf-8')
        
        # Save as JSON
        json_file = output_path / f"{split}.json"
        json_data = []
        for _, row in split_df.iterrows():
            json_data.append({
                'src': row['korean'],
                'tgt': row['english'],
                'domain': row['domain']
            })
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(split_df)} pairs to {tsv_file} and {json_file}")
    
    # Train SentencePiece tokenizer
    logger.info("Training SentencePiece tokenizer")
    
    # Combine all text for tokenizer training
    all_text = []
    for _, row in df.iterrows():
        all_text.append(row['korean'])
        all_text.append(row['english'])
    
    # Save combined text for training
    train_file = output_path / "tokenizer_train.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        for text in all_text:
            f.write(text + '\n')
    
    # Train SentencePiece with corrected configuration
    model_prefix = str(output_path / "sentencepiece_large")
    
    spm.SentencePieceTrainer.train(
        input=str(train_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe',
        normalization_rule_name='nmt_nfkc_cf',
        remove_extra_whitespaces=True,
        max_sentence_length=128,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        control_symbols=['[BT]', '[IMG]', '[AUDIO]', '[MASK]', '[SEP]'],
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        num_threads=8
    )
    
    tokenizer_model = f"{model_prefix}.model"
    logger.info(f"Trained SentencePiece tokenizer: {tokenizer_model}")
    
    # Save tokenizer config
    tokenizer_config = {
        'vocab_size': vocab_size,
        'character_coverage': 0.9995,
        'max_sentence_length': 128,
        'special_tokens': {
            'pad_token': '<pad>',
            'unk_token': '<unk>',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'additional_special_tokens': ['[BT]', '[IMG]', '[AUDIO]', '[MASK]', '[SEP]']
        }
    }
    
    config_file = output_path / "tokenizer_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Corpus preparation completed!")
    logger.info(f"Tokenizer model: {tokenizer_model}")
    logger.info(f"Output directory: {output_dir}")
    
    return tokenizer_model

if __name__ == "__main__":
    prepare_large_corpus_simple()