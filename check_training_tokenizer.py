"""
Check the actual tokenizer used in training.
"""

import sentencepiece as spm

# Load the tokenizer that the training script is using
tokenizer = spm.SentencePieceProcessor(
    model_file="data/processed_large_simple/sentencepiece_large.model"
)

print(f"Training tokenizer vocabulary size: {tokenizer.get_piece_size()}")
print(f"BOS ID: {tokenizer.bos_id()}")
print(f"EOS ID: {tokenizer.eos_id()}")
print(f"PAD ID: {tokenizer.pad_id()}")
print(f"UNK ID: {tokenizer.unk_id()}")

# Compare with the simple tokenizer
simple_tokenizer = spm.SentencePieceProcessor(
    model_file="data/tokenizers/kr_en_simple.model"
)
print(f"\nSimple tokenizer vocabulary size: {simple_tokenizer.get_piece_size()}")

# Check a sample from the training data
import json

with open("data/processed_large_simple/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\nFirst training sample:")
print(f"Source: {data[0]['src'][:50]}...")
print(f"Target: {data[0]['tgt'][:50]}...")

# Tokenize with both tokenizers
src_tokens_large = tokenizer.encode(data[0]["src"], out_type=int)
tgt_tokens_large = tokenizer.encode(data[0]["tgt"], out_type=int)

src_tokens_simple = simple_tokenizer.encode(data[0]["src"], out_type=int)
tgt_tokens_simple = simple_tokenizer.encode(data[0]["tgt"], out_type=int)

print(f"\nLarge tokenizer tokens:")
print(f"Source: {src_tokens_large[:10]}...")
print(f"Target: {tgt_tokens_large[:10]}...")
print(f"Max source token: {max(src_tokens_large)}")
print(f"Max target token: {max(tgt_tokens_large)}")

print(f"\nSimple tokenizer tokens:")
print(f"Source: {src_tokens_simple[:10]}...")
print(f"Target: {tgt_tokens_simple[:10]}...")
print(f"Max source token: {max(src_tokens_simple)}")
print(f"Max target token: {max(tgt_tokens_simple)}")
