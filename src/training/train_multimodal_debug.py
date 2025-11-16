import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
import json
import time
import logging
from pathlib import Path
from tqdm import tqdm

# Import our models
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.nmt_transformer import NMTTransformer
from src.models.image_encoder import EnhancedMultimodalNMT
from src.models.audio_encoder import MultimodalAudioNMT


class SimpleTokenizer:
    """Simple tokenizer for testing purposes."""

    def __init__(self):
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        # Build vocabulary from data
        self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

        # Add some common Korean and English characters/words
        korean_chars = list(
            "안녕하세요감사합니다죄송네아니요오늘날씨가좋네요밥먹었어요어디가세요"
        )
        english_words = [
            "hello",
            "thank",
            "you",
            "sorry",
            "yes",
            "no",
            "today",
            "weather",
            "nice",
            "eat",
            "go",
            "where",
        ]

        for char in korean_chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        for word in english_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """Simple encoding - character level."""
        tokens = [self.bos_id]
        for char in text.lower():
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                # For unknown characters, try to add them
                self.vocab[char] = len(self.vocab)
                tokens.append(self.vocab[char])
        tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Simple decoding."""
        # Reverse vocab
        reverse_vocab = {v: k for k, v in self.vocab.items()}

        result = []
        for token in tokens:
            if token in [self.pad_id, self.bos_id, self.eos_id]:
                continue
            if token in reverse_vocab:
                result.append(reverse_vocab[token])
            else:
                result.append("<unk>")

        return "".join(result)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class SimpleDataset(Dataset):
    """Simple dataset for multimodal training."""

    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize text
        src_tokens = self.tokenizer.encode(item["korean"])[: self.max_length - 2]
        tgt_tokens = self.tokenizer.encode(item["english"])[: self.max_length - 2]

        # Pad sequences
        src_tokens = [self.tokenizer.bos_id] + src_tokens + [self.tokenizer.eos_id]
        tgt_tokens = [self.tokenizer.bos_id] + tgt_tokens + [self.tokenizer.eos_id]

        src_tokens += [self.tokenizer.pad_id] * (self.max_length - len(src_tokens))
        tgt_tokens += [self.tokenizer.pad_id] * (self.max_length - len(tgt_tokens))

        return {
            "src_tokens": torch.tensor(src_tokens, dtype=torch.long),
            "tgt_tokens": torch.tensor(tgt_tokens, dtype=torch.long),
            "src_length": torch.tensor(
                len([t for t in src_tokens if t != self.tokenizer.pad_id])
            ),
            "tgt_length": torch.tensor(
                len([t for t in tgt_tokens if t != self.tokenizer.pad_id])
            ),
            "image": torch.randn(3, 64, 64),  # Synthetic image
            "audio": torch.randn(8000),  # Synthetic audio (1 second)
            "korean_text": item["korean"],
            "english_text": item["english"],
        }


def create_training_data() -> Tuple[List[Dict], List[Dict]]:
    """Create training and validation data."""
    training_pairs = [
        {"korean": "안녕하세요", "english": "hello"},
        {"korean": "감사합니다", "english": "thank you"},
        {"korean": "죄송합니다", "english": "sorry"},
        {"korean": "네", "english": "yes"},
        {"korean": "아니요", "english": "no"},
        {"korean": "오늘 날씨가 좋네요", "english": "weather nice today"},
        {"korean": "밥 먹었어요?", "english": "did you eat"},
        {"korean": "어디 가세요?", "english": "where are you going"},
        {"korean": "잘 지내셨어요?", "english": "have you been well"},
        {"korean": "저는 한국어를 배우고 있어요", "english": "i am learning korean"},
    ]

    validation_pairs = [
        {
            "korean": "이 책은 정말 흥미로워요",
            "english": "this book really interesting",
        },
        {
            "korean": "내일 학교에 가야 해요",
            "english": "i have to go to school tomorrow",
        },
        {"korean": "커피 마시고 싶어요", "english": "i want to drink coffee"},
        {"korean": "회의는 오후 3시에 시작됩니다", "english": "meeting starts at 3 pm"},
        {
            "korean": "프로젝트 일정을 확인해 주세요",
            "english": "please check project schedule",
        },
    ]

    return training_pairs, validation_pairs


def test_tokenizer():
    """Test the tokenizer to ensure it's working correctly."""
    print("Testing tokenizer...")
    tokenizer = SimpleTokenizer()

    test_text = "안녕하세요"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    return tokenizer


def test_model_creation():
    """Test model creation to ensure dimensions are correct."""
    print("\nTesting model creation...")

    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    device = "cpu"  # Use CPU for debugging

    # Test text model
    print("Creating text model...")
    text_model = NMTTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=128,
        max_len=32,
        dropout=0.1,
        pad_id=tokenizer.pad_id,
        use_flash=False,
    ).to(device)

    # Test with dummy input
    batch_size = 2
    seq_len = 16
    src_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"Input shape: {src_tokens.shape}")
    print(f"Target shape: {tgt_tokens.shape}")

    try:
        with torch.no_grad():
            output = text_model(src_tokens, tgt_tokens)
        print(f"Text model output shape: {output.shape}")
        print("✓ Text model works!")
    except Exception as e:
        print(f"✗ Text model failed: {e}")
        return None

    # Test image model
    print("Creating image model...")
    image_model = EnhancedMultimodalNMT(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=128,
        max_len=32,
        dropout=0.1,
        pad_id=tokenizer.pad_id,
        use_flash=False,
        img_size=32,
        patch_size=4,
        img_embed_dim=32,
        img_num_heads=2,
        img_num_layers=1,
        fusion_dim=64,
        fusion_heads=2,
    ).to(device)

    try:
        image_model.set_mode("image")
        test_image = torch.randn(batch_size, 3, 32, 32)
        with torch.no_grad():
            output = image_model(src_tokens, tgt_tokens, src_images=test_image)
        print(f"Image model output shape: {output.shape}")
        print("✓ Image model works!")
    except Exception as e:
        print(f"✗ Image model failed: {e}")
        return None

    # Test audio model
    print("Creating audio model...")
    audio_model = MultimodalAudioNMT(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=128,
        max_len=32,
        dropout=0.1,
        pad_id=tokenizer.pad_id,
        use_flash=False,
        audio_encoder_type="cnn",
        audio_output_dim=64,
        fusion_dim=64,
        fusion_heads=2,
    ).to(device)

    try:
        audio_model.set_mode("audio")
        test_audio = torch.randn(batch_size, 8000)
        with torch.no_grad():
            output = audio_model(src_tokens, tgt_tokens, src_audio=test_audio)
        print(f"Audio model output shape: {output.shape}")
        print("✓ Audio model works!")
    except Exception as e:
        print(f"✗ Audio model failed: {e}")
        return None

    return tokenizer, text_model, image_model, audio_model


def main():
    """Main function for debugging."""
    print("=" * 60)
    print("MULTIMODAL MODEL DEBUGGING")
    print("=" * 60)

    # Test tokenizer
    tokenizer = test_tokenizer()

    # Test model creation
    models = test_model_creation()

    if models is None:
        print("\nModel creation failed. Stopping.")
        return

    tokenizer, text_model, image_model, audio_model = models

    print("\n" + "=" * 60)
    print("ALL MODELS CREATED SUCCESSFULLY!")
    print("=" * 60)

    # Test dataset creation
    print("\nTesting dataset creation...")
    train_data, val_data = create_training_data()

    train_dataset = SimpleDataset(train_data, tokenizer)
    val_dataset = SimpleDataset(val_data, tokenizer)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Test a single batch
    sample = train_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Source tokens shape: {sample['src_tokens'].shape}")
    print(f"Target tokens shape: {sample['tgt_tokens'].shape}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Audio shape: {sample['audio'].shape}")

    print("\n✓ Dataset creation works!")

    print("\n" + "=" * 60)
    print("DEBUGGING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Ready for training!")


if __name__ == "__main__":
    main()
