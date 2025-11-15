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
        self.vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        
        # Add some common Korean and English characters/words
        korean_chars = list('안녕하세요감사합니다죄송네아니요오늘날씨가좋네요밥먹었어요어디가세요')
        english_words = ['hello', 'thank', 'you', 'sorry', 'yes', 'no', 'today', 'weather', 'nice', 'eat', 'go', 'where']
        
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
                result.append('<unk>')
        
        return ''.join(result)
    
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
        src_tokens = self.tokenizer.encode(item['korean'])[:self.max_length-2]
        tgt_tokens = self.tokenizer.encode(item['english'])[:self.max_length-2]
        
        # Pad sequences
        src_tokens = [self.tokenizer.bos_id] + src_tokens + [self.tokenizer.eos_id]
        tgt_tokens = [self.tokenizer.bos_id] + tgt_tokens + [self.tokenizer.eos_id]
        
        src_tokens += [self.tokenizer.pad_id] * (self.max_length - len(src_tokens))
        tgt_tokens += [self.tokenizer.pad_id] * (self.max_length - len(tgt_tokens))
        
        return {
            'src_tokens': torch.tensor(src_tokens, dtype=torch.long),
            'tgt_tokens': torch.tensor(tgt_tokens, dtype=torch.long),
            'src_length': torch.tensor(len([t for t in src_tokens if t != self.tokenizer.pad_id])),
            'tgt_length': torch.tensor(len([t for t in tgt_tokens if t != self.tokenizer.pad_id])),
            'image': torch.randn(3, 64, 64),  # Synthetic image
            'audio': torch.randn(8000),  # Synthetic audio (1 second)
            'korean_text': item['korean'],
            'english_text': item['english']
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
        {"korean": "이 책은 정말 흥미로워요", "english": "this book really interesting"},
        {"korean": "내일 학교에 가야 해요", "english": "i have to go to school tomorrow"},
        {"korean": "커피 마시고 싶어요", "english": "i want to drink coffee"},
        {"korean": "회의는 오후 3시에 시작됩니다", "english": "meeting starts at 3 pm"},
        {"korean": "프로젝트 일정을 확인해 주세요", "english": "please check project schedule"},
    ]
    
    validation_pairs = [
        {"korean": "보고서를 제출해야 합니다", "english": "i need to submit report"},
        {"korean": "많이 드세요", "english": "please eat a lot"},
        {"korean": "수고하셨습니다", "english": "thank you for your hard work"},
        {"korean": "들어오세요", "english": "please come in"},
        {"korean": "이것은 무엇입니까?", "english": "what is this"},
    ]
    
    return training_pairs, validation_pairs


def train_model(model, dataloader, optimizer, criterion, device, modality):
    """Train a single model."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    if hasattr(model, 'set_mode'):
        model.set_mode(modality)
    
    progress_bar = tqdm(dataloader, desc=f"Training {modality}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move data to device
        src_tokens = batch['src_tokens'].to(device)
        tgt_tokens = batch['tgt_tokens'].to(device)
        
        # Forward pass based on modality
        if modality == 'text':
            outputs = model(src_tokens, tgt_tokens[:, :-1])
        elif modality == 'image':
            images = batch['image'].to(device)
            outputs = model(src_tokens, tgt_tokens[:, :-1], src_images=images)
        elif modality == 'audio':
            audio = batch['audio'].to(device)
            outputs = model(src_tokens, tgt_tokens[:, :-1], src_audio=audio)
        
        # Calculate loss
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_tokens[:, 1:].reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Calculate accuracy
        pred_tokens = outputs.argmax(dim=-1)
        mask = (tgt_tokens[:, 1:] != 0)  # pad_id = 0
        correct = (pred_tokens == tgt_tokens[:, 1:]) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return {'loss': avg_loss, 'accuracy': accuracy}


def validate_model(model, dataloader, criterion, device, modality):
    """Validate a model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    if hasattr(model, 'set_mode'):
        model.set_mode(modality)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validating {modality}"):
            # Move data to device
            src_tokens = batch['src_tokens'].to(device)
            tgt_tokens = batch['tgt_tokens'].to(device)
            
            # Forward pass based on modality
            if modality == 'text':
                outputs = model(src_tokens, tgt_tokens[:, :-1])
            elif modality == 'image':
                images = batch['image'].to(device)
                outputs = model(src_tokens, tgt_tokens[:, :-1], src_images=images)
            elif modality == 'audio':
                audio = batch['audio'].to(device)
                outputs = model(src_tokens, tgt_tokens[:, :-1], src_audio=audio)
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_tokens[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            pred_tokens = outputs.argmax(dim=-1)
            mask = (tgt_tokens[:, 1:] != 0)  # pad_id = 0
            correct = (pred_tokens == tgt_tokens[:, 1:]) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return {'loss': avg_loss, 'accuracy': accuracy}


def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create training data
    print("Creating training and validation data...")
    train_data, val_data = create_training_data()
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create datasets
    train_dataset = SimpleDataset(train_data, tokenizer)
    val_dataset = SimpleDataset(val_data, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create models
    print("Creating models...")
    
    # Text model
    text_model = NMTTransformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        max_len=32,
        dropout=0.1,
        pad_id=tokenizer.pad_id,
        use_flash=False
    ).to(device)
    
    # Image model
    image_model = EnhancedMultimodalNMT(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        max_len=32,
        dropout=0.1,
        pad_id=tokenizer.pad_id,
        use_flash=False,
        img_size=64,
        patch_size=8,
        img_embed_dim=64,
        img_num_heads=4,
        img_num_layers=2,
        fusion_dim=128,
        fusion_heads=4
    ).to(device)
    
    # Audio model
    audio_model = MultimodalAudioNMT(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        max_len=32,
        dropout=0.1,
        pad_id=tokenizer.pad_id,
        use_flash=False,
        audio_encoder_type='cnn',
        audio_output_dim=128,
        fusion_dim=128,
        fusion_heads=4
    ).to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    # Optimizers
    text_optimizer = optim.AdamW(text_model.parameters(), lr=1e-3, weight_decay=0.01)
    image_optimizer = optim.AdamW(image_model.parameters(), lr=1e-3, weight_decay=0.01)
    audio_optimizer = optim.AdamW(audio_model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Training loop
    num_epochs = 3
    print(f"\nStarting training for {num_epochs} epochs...")
    
    results = {
        'text': {'train': [], 'val': []},
        'image': {'train': [], 'val': []},
        'audio': {'train': [], 'val': []}
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train text model
        print("Training text model...")
        text_train_metrics = train_model(text_model, train_loader, text_optimizer, criterion, device, 'text')
        text_val_metrics = validate_model(text_model, val_loader, criterion, device, 'text')
        results['text']['train'].append(text_train_metrics)
        results['text']['val'].append(text_val_metrics)
        
        # Train image model
        print("Training image model...")
        image_train_metrics = train_model(image_model, train_loader, image_optimizer, criterion, device, 'image')
        image_val_metrics = validate_model(image_model, val_loader, criterion, device, 'image')
        results['image']['train'].append(image_train_metrics)
        results['image']['val'].append(image_val_metrics)
        
        # Train audio model
        print("Training audio model...")
        audio_train_metrics = train_model(audio_model, train_loader, audio_optimizer, criterion, device, 'audio')
        audio_val_metrics = validate_model(audio_model, val_loader, criterion, device, 'audio')
        results['audio']['train'].append(audio_train_metrics)
        results['audio']['val'].append(audio_val_metrics)
        
        # Print epoch results
        print(f"Text - Train Loss: {text_train_metrics['loss']:.4f}, Train Acc: {text_train_metrics['accuracy']:.4f}, "
              f"Val Loss: {text_val_metrics['loss']:.4f}, Val Acc: {text_val_metrics['accuracy']:.4f}")
        print(f"Image - Train Loss: {image_train_metrics['loss']:.4f}, Train Acc: {image_train_metrics['accuracy']:.4f}, "
              f"Val Loss: {image_val_metrics['loss']:.4f}, Val Acc: {image_val_metrics['accuracy']:.4f}")
        print(f"Audio - Train Loss: {audio_train_metrics['loss']:.4f}, Train Acc: {audio_train_metrics['accuracy']:.4f}, "
              f"Val Loss: {audio_val_metrics['loss']:.4f}, Val Acc: {audio_val_metrics['accuracy']:.4f}")
    
    # Save models
    print("\nSaving models...")
    save_dir = Path("models/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(text_model.state_dict(), save_dir / "text_model.pt")
    torch.save(image_model.state_dict(), save_dir / "image_model.pt")
    torch.save(audio_model.state_dict(), save_dir / "audio_model.pt")
    
    # Save tokenizer vocab
    with open(save_dir / "tokenizer_vocab.json", 'w', encoding='utf-8') as f:
        json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=2)
    
    print("Training completed!")
    print("Models saved to models/checkpoints/")
    
    return results


if __name__ == "__main__":
    main()