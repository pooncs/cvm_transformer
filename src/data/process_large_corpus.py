"""
Process the large Korean-English corpus using the sophisticated preparation pipeline.
This will apply advanced cleaning, alignment validation, and professional tokenization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.prepare_corpus import prepare_corpus, CorpusConfig

def main():
    # Configure corpus preparation with more lenient settings for our generated corpus
    config = CorpusConfig(
        src_lang="ko",
        tgt_lang="en", 
        vocab_size=32000,  # Professional vocabulary size
        character_coverage=0.9995,
        max_sentence_length=128,
        min_sentence_length=2,
        deduplication_threshold=0.95,
        alignment_threshold=0.1,  # Much more lenient for our generated corpus
        test_size=0.02,
        val_size=0.02,
        num_workers=8,
        output_dir="data/processed_large",
        tokenizer_model_prefix="sentencepiece_large"
    )
    
    # Input files - our newly generated large corpus
    input_files = [
        "data/raw/korean_english_large.tsv"
    ]
    
    print("Starting large corpus preparation...")
    print(f"Input file: {input_files[0]}")
    print(f"Target vocabulary size: {config.vocab_size}")
    print(f"Output directory: {config.output_dir}")
    print(f"Alignment threshold (lenient): {config.alignment_threshold}")
    
    # Process the corpus
    tokenizer_model = prepare_corpus(input_files, config)
    
    print(f"\nLarge corpus preparation completed!")
    print(f"Tokenizer model: {tokenizer_model}")
    print(f"Check output directory: {config.output_dir}")

if __name__ == "__main__":
    main()