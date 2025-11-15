"""
Scalable Korean-English corpus preparation and tokenizer training.
Handles data ingestion, cleaning, filtering, alignment validation, and domain tagging.
"""

import os
import json
import gzip
import bz2
import lzma
import re
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterator
from dataclasses import dataclass
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CorpusConfig:
    """Configuration for corpus preparation."""
    src_lang: str = "ko"
    tgt_lang: str = "en"
    vocab_size: int = 32000
    character_coverage: float = 0.9995
    max_sentence_length: int = 128
    min_sentence_length: int = 2
    deduplication_threshold: float = 0.95
    alignment_threshold: float = 0.7
    test_size: float = 0.02
    val_size: float = 0.02
    num_workers: int = mp.cpu_count()
    cache_dir: str = "data/cache"
    output_dir: str = "data/processed"
    tokenizer_model_prefix: str = "sentencepiece"

class DataCleaner:
    """Text cleaning and normalization utilities."""
    
    @staticmethod
    def normalize_korean(text: str) -> str:
        """Normalize Korean text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize punctuation
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        text = text.replace('—', '-').replace('–', '-')
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_english(text: str) -> str:
        """Normalize English text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize punctuation
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        text = text.replace('—', '-').replace('–', '-')
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    @staticmethod
    def is_valid_sentence(text: str, min_len: int = 2, max_len: int = 128) -> bool:
        """Check if sentence is valid."""
        if not text or len(text) < min_len or len(text) > max_len:
            return False
        
        # Check for excessive punctuation or numbers
        if len(re.findall(r'[!?.]{3,}', text)) > 0:
            return False
        
        # Check for excessive uppercase (likely acronym spam)
        words = text.split()
        if len(words) > 0:
            uppercase_ratio = sum(1 for word in words if word.isupper()) / len(words)
            if uppercase_ratio > 0.5:
                return False
        
        return True
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection."""
        # Count Korean characters
        korean_count = len(re.findall(r'[\uac00-\ud7af]', text))
        
        # Count English characters
        english_count = len(re.findall(r'[a-zA-Z]', text))
        
        # Count other characters
        total_count = len(text.replace(' ', ''))
        
        if korean_count > english_count and korean_count > total_count * 0.3:
            return "ko"
        elif english_count > korean_count and english_count > total_count * 0.3:
            return "en"
        else:
            return "unknown"

class AlignmentValidator:
    """Sentence alignment validation using bilingual embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/LaBSE"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None  # Will be loaded on demand
        self._load_model()
    
    def _load_model(self):
        """Load the alignment model."""
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
            self.model.eval()
            logger.info("Loaded LaBSE alignment model")
        except Exception as e:
            logger.warning(f"Could not load LaBSE model: {e}. Using fallback alignment.")
            self.model = None
    
    def compute_similarity(self, src_text: str, tgt_text: str) -> float:
        """Compute semantic similarity between source and target text."""
        if self.model is None:
            # Fallback: simple length ratio and character overlap
            src_len = len(src_text.split())
            tgt_len = len(tgt_text.split())
            length_ratio = min(src_len, tgt_len) / max(src_len, tgt_len) if max(src_len, tgt_len) > 0 else 0
            
            # Character overlap
            src_chars = set(src_text.lower())
            tgt_chars = set(tgt_text.lower())
            overlap = len(src_chars.intersection(tgt_chars)) / len(src_chars.union(tgt_chars)) if src_chars.union(tgt_chars) else 0
            
            return (length_ratio + overlap) / 2
        
        # Use LaBSE for semantic similarity
        try:
            with torch.no_grad():
                src_inputs = self.tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                tgt_inputs = self.tokenizer(tgt_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                
                src_embeddings = self.model(**src_inputs).pooler_output
                tgt_embeddings = self.model(**tgt_inputs).pooler_output
                
                similarity = torch.cosine_similarity(src_embeddings, tgt_embeddings).item()
                return similarity
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

class DomainClassifier:
    """Domain classification for sentence pairs."""
    
    def __init__(self):
        self.domains = {
            'news': ['news', 'report', 'article', 'journalist', 'breaking', 'headline'],
            'conversational': ['hello', 'hi', 'how are you', 'thank you', 'please', 'sorry'],
            'technical': ['technology', 'software', 'hardware', 'algorithm', 'data', 'code'],
            'subtitles': ['subtitle', 'caption', 'dialogue', 'scene', 'character'],
            'wiki': ['wikipedia', 'encyclopedia', 'article', 'reference', 'citation'],
            'general': []  # Default domain
        }
    
    def classify_domain(self, src_text: str, tgt_text: str) -> str:
        """Classify the domain of a sentence pair."""
        combined_text = (src_text + " " + tgt_text).lower()
        
        domain_scores = {}
        for domain, keywords in self.domains.items():
            if domain == 'general':
                continue
            
            score = sum(1 for keyword in keywords if keyword in combined_text)
            domain_scores[domain] = score
        
        # Return domain with highest score, or 'general' if no clear match
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            return best_domain if domain_scores[best_domain] > 0 else 'general'
        else:
            return 'general'

class ParallelCorpusProcessor:
    """Process parallel Korean-English corpus."""
    
    def __init__(self, config: CorpusConfig):
        self.config = config
        self.cleaner = DataCleaner()
        self.aligner = AlignmentValidator()
        self.domain_classifier = DomainClassifier()
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load_parallel_corpus(self, file_path: str) -> Iterator[Tuple[str, str]]:
        """Load parallel corpus from various formats."""
        path = Path(file_path)
        
        if path.suffix == '.gz':
            opener = gzip.open
        elif path.suffix == '.bz2':
            opener = bz2.open
        elif path.suffix == '.xz':
            opener = lzma.open
        else:
            opener = open
        
        with opener(file_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try tab-separated format first
                parts = line.split('\t')
                if len(parts) >= 2:
                    src_text, tgt_text = parts[0], parts[1]
                else:
                    # Try ||| separated format
                    parts = line.split(' ||| ')
                    if len(parts) >= 2:
                        src_text, tgt_text = parts[0], parts[1]
                    else:
                        logger.warning(f"Line {line_num}: Invalid format in {file_path}")
                        continue
                
                yield src_text.strip(), tgt_text.strip()
    
    def clean_sentence_pair(self, src_text: str, tgt_text: str) -> Optional[Tuple[str, str]]:
        """Clean and validate a sentence pair."""
        # Clean texts
        src_clean = self.cleaner.normalize_korean(src_text)
        tgt_clean = self.cleaner.normalize_english(tgt_text)
        
        # Check validity
        if not (self.cleaner.is_valid_sentence(src_clean, self.config.min_sentence_length, 
                                               self.config.max_sentence_length) and
                self.cleaner.is_valid_sentence(tgt_clean, self.config.min_sentence_length,
                                              self.config.max_sentence_length)):
            return None
        
        # Check language detection
        if (self.cleaner.detect_language(src_clean) != self.config.src_lang or
            self.cleaner.detect_language(tgt_clean) != self.config.tgt_lang):
            return None
        
        return src_clean, tgt_clean
    
    def filter_by_alignment(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Filter sentence pairs by alignment quality."""
        filtered_pairs = []
        
        for src_text, tgt_text in tqdm(pairs, desc="Alignment filtering"):
            similarity = self.aligner.compute_similarity(src_text, tgt_text)
            if similarity >= self.config.alignment_threshold:
                filtered_pairs.append((src_text, tgt_text))
        
        logger.info(f"Alignment filtering: {len(filtered_pairs)}/{len(pairs)} pairs retained")
        return filtered_pairs
    
    def deduplicate_pairs(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Remove duplicate and near-duplicate pairs."""
        seen = set()
        unique_pairs = []
        
        for src_text, tgt_text in tqdm(pairs, desc="Deduplication"):
            # Create a normalized key
            key = (src_text.lower().strip(), tgt_text.lower().strip())
            
            # Check for exact duplicates
            if key in seen:
                continue
            
            # Check for near-duplicates using simple similarity
            is_near_duplicate = False
            for seen_src, seen_tgt in seen:
                src_similarity = self._text_similarity(src_text.lower(), seen_src.lower())
                tgt_similarity = self._text_similarity(tgt_text.lower(), seen_tgt.lower())
                
                if src_similarity > self.config.deduplication_threshold and \
                   tgt_similarity > self.config.deduplication_threshold:
                    is_near_duplicate = True
                    break
            
            if not is_near_duplicate:
                seen.add(key)
                unique_pairs.append((src_text, tgt_text))
        
        logger.info(f"Deduplication: {len(unique_pairs)}/{len(pairs)} pairs retained")
        return unique_pairs
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity."""
        if not text1 or not text2:
            return 0.0
        
        # Jaccard similarity on character n-grams (n=2)
        def get_ngrams(text: str, n: int = 2) -> set:
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def add_domain_tags(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """Add domain tags to sentence pairs."""
        tagged_pairs = []
        
        for src_text, tgt_text in tqdm(pairs, desc="Domain tagging"):
            domain = self.domain_classifier.classify_domain(src_text, tgt_text)
            tagged_pairs.append((src_text, tgt_text, domain))
        
        return tagged_pairs
    
    def process_corpus(self, input_files: List[str]) -> Dict[str, List[Tuple[str, str, str]]]:
        """Process multiple corpus files."""
        all_pairs = []
        
        # Load all pairs
        for file_path in input_files:
            logger.info(f"Loading {file_path}")
            pairs = list(self.load_parallel_corpus(file_path))
            logger.info(f"Loaded {len(pairs)} pairs from {file_path}")
            all_pairs.extend(pairs)
        
        logger.info(f"Total pairs loaded: {len(all_pairs)}")
        
        # Clean pairs
        logger.info("Cleaning sentence pairs")
        cleaned_pairs = []
        for src_text, tgt_text in tqdm(all_pairs, desc="Cleaning"):
            cleaned = self.clean_sentence_pair(src_text, tgt_text)
            if cleaned:
                cleaned_pairs.append(cleaned)
        
        logger.info(f"Cleaned pairs: {len(cleaned_pairs)}")
        
        # Filter by alignment
        logger.info("Filtering by alignment")
        aligned_pairs = self.filter_by_alignment(cleaned_pairs)
        
        # Deduplicate
        logger.info("Deduplicating pairs")
        unique_pairs = self.deduplicate_pairs(aligned_pairs)
        
        # Add domain tags
        logger.info("Adding domain tags")
        tagged_pairs = self.add_domain_tags(unique_pairs)
        
        # Split into train/val/test
        logger.info("Splitting datasets")
        train_pairs, temp_pairs = train_test_split(
            tagged_pairs, test_size=self.config.test_size + self.config.val_size, 
            random_state=42, shuffle=True
        )
        
        val_size = self.config.val_size / (self.config.test_size + self.config.val_size)
        val_pairs, test_pairs = train_test_split(
            temp_pairs, test_size=1 - val_size, random_state=42, shuffle=True
        )
        
        logger.info(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
        
        return {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
    
    def save_datasets(self, datasets: Dict[str, List[Tuple[str, str, str]]], 
                     output_dir: str):
        """Save processed datasets."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split, pairs in datasets.items():
            # Save as TSV
            tsv_file = output_path / f"{split}.tsv"
            with open(tsv_file, 'w', encoding='utf-8') as f:
                for src, tgt, domain in pairs:
                    f.write(f"{src}\t{tgt}\t{domain}\n")
            
            # Save as JSON
            json_file = output_path / f"{split}.json"
            data = [
                {'src': src, 'tgt': tgt, 'domain': domain}
                for src, tgt, domain in pairs
            ]
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(pairs)} pairs to {tsv_file} and {json_file}")
    
    def train_tokenizer(self, datasets: Dict[str, List[Tuple[str, str, str]]], 
                       output_dir: str):
        """Train SentencePiece tokenizer on the corpus."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Combine all text for tokenizer training
        all_text = []
        for split, pairs in datasets.items():
            for src, tgt, _ in pairs:
                all_text.append(src)
                all_text.append(tgt)
        
        # Save combined text for training
        train_file = output_path / "tokenizer_train.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in all_text:
                f.write(text + '\n')
        
        # Train SentencePiece
        model_prefix = str(output_path / self.config.tokenizer_model_prefix)
        
        spm.SentencePieceTrainer.train(
            input=str(train_file),
            model_prefix=model_prefix,
            vocab_size=self.config.vocab_size,
            character_coverage=self.config.character_coverage,
            model_type='bpe',
            normalization_rule_name='nmt_nfkc_cf',
            remove_extra_whitespaces=True,
            max_sentence_length=self.config.max_sentence_length,
            split_digits=True,
            allow_whitespace_only_pieces=True,
            control_symbols=['[BT]', '[IMG]', '[AUDIO]', '[MASK]', '[SEP]'],
            user_defined_symbols=['<bos>', '<eos>', '<pad>', '<unk>'],
            shuffle_input_sentence=True,
            train_extremely_large_corpus=True,
            num_threads=self.config.num_workers
        )
        
        logger.info(f"Trained SentencePiece tokenizer: {model_prefix}.model")
        
        # Save tokenizer config
        tokenizer_config = {
            'vocab_size': self.config.vocab_size,
            'character_coverage': self.config.character_coverage,
            'max_sentence_length': self.config.max_sentence_length,
            'special_tokens': {
                'pad_token': '<pad>',
                'unk_token': '<unk>',
                'bos_token': '<bos>',
                'eos_token': '<eos>',
                'additional_special_tokens': ['[BT]', '[IMG]', '[AUDIO]', '[MASK]', '[SEP]']
            }
        }
        
        config_file = output_path / "tokenizer_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
        return f"{model_prefix}.model"

def prepare_corpus(input_files: List[str], config: Optional[CorpusConfig] = None) -> str:
    """Main function to prepare corpus."""
    if config is None:
        config = CorpusConfig()
    
    processor = ParallelCorpusProcessor(config)
    
    # Process corpus
    logger.info("Starting corpus preparation")
    datasets = processor.process_corpus(input_files)
    
    # Save datasets
    processor.save_datasets(datasets, config.output_dir)
    
    # Train tokenizer
    tokenizer_model = processor.train_tokenizer(datasets, config.output_dir)
    
    logger.info(f"Corpus preparation completed. Tokenizer model: {tokenizer_model}")
    logger.info(f"Output directory: {config.output_dir}")
    
    return tokenizer_model

if __name__ == "__main__":
    # Example usage
    input_files = [
        "data/raw/korean_english_parallel.tsv",
        "data/raw/ai_hub_ko_en.tsv.gz",
        "data/raw/paracrawl_ko_en.tsv.bz2"
    ]
    
    config = CorpusConfig(
        vocab_size=32000,
        max_sentence_length=128,
        num_workers=8
    )
    
    tokenizer_model = prepare_corpus(input_files, config)
    print(f"Tokenizer model saved to: {tokenizer_model}")