import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time
import logging
from pathlib import Path
from tqdm import tqdm
import random

# Import our models
from src.models.nmt_transformer import NMTTransformer
from src.models.image_encoder import EnhancedMultimodalNMT, KoreanTextImageEncoder
from src.models.audio_encoder import MultimodalAudioNMT, KoreanSpeechEncoder


class MultimodalDataset(Dataset):
    """Dataset for multimodal Korean-English translation."""
    
    def __init__(self, 
                 text_data: List[Dict[str, str]],
                 tokenizer,
                 max_length: int = 128,
                 include_images: bool = True,
                 include_audio: bool = True):
        
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_images = include_images
        self.include_audio = include_audio
        
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        item = self.text_data[idx]
        
        # Tokenize text
        src_tokens = self.tokenizer.encode(item['korean'])[:self.max_length-2]
        tgt_tokens = self.tokenizer.encode(item['english'])[:self.max_length-2]
        
        # Pad sequences
        src_tokens = [self.tokenizer.bos_id] + src_tokens + [self.tokenizer.eos_id]
        tgt_tokens = [self.tokenizer.bos_id] + tgt_tokens + [self.tokenizer.eos_id]
        
        src_tokens += [self.tokenizer.pad_id] * (self.max_length - len(src_tokens))
        tgt_tokens += [self.tokenizer.pad_id] * (self.max_length - len(tgt_tokens))
        
        result = {
            'src_tokens': torch.tensor(src_tokens, dtype=torch.long),
            'tgt_tokens': torch.tensor(tgt_tokens, dtype=torch.long),
            'src_length': torch.tensor(len([t for t in src_tokens if t != self.tokenizer.pad_id])),
            'tgt_length': torch.tensor(len([t for t in tgt_tokens if t != self.tokenizer.pad_id])),
            'korean_text': item['korean'],
            'english_text': item['english']
        }
        
        # Add synthetic image data
        if self.include_images:
            result['image'] = torch.randn(3, 224, 224)  # Synthetic image
            
        # Add synthetic audio data
        if self.include_audio:
            result['audio'] = torch.randn(16000 * 3)  # 3 seconds of synthetic audio
            result['audio_length'] = torch.tensor(16000 * 3)
            
        return result


class MultimodalTrainer:
    """Trainer for multimodal Korean-English translation models."""
    
    def __init__(self,
                 text_model: NMTTransformer,
                 image_model: EnhancedMultimodalNMT,
                 audio_model: MultimodalAudioNMT,
                 tokenizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.text_model = text_model
        self.image_model = image_model
        self.audio_model = audio_model
        self.tokenizer = tokenizer
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
        
        # Optimizers for each model
        self.text_optimizer = optim.AdamW(text_model.parameters(), lr=1e-4, weight_decay=0.01)
        self.image_optimizer = optim.AdamW(image_model.parameters(), lr=1e-4, weight_decay=0.01)
        self.audio_optimizer = optim.AdamW(audio_model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Schedulers
        self.text_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.text_optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        self.image_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.image_optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        self.audio_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.audio_optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
    def train_epoch(self, dataloader: DataLoader, model, optimizer, scheduler, modality: str) -> Dict[str, float]:
        """Train one epoch for a specific model."""
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        if modality == 'text':
            self.text_model.set_mode('text')
        elif modality == 'image':
            self.image_model.set_mode('image')
        elif modality == 'audio':
            self.audio_model.set_mode('audio')
        elif modality == 'multimodal':
            self.image_model.set_mode('multimodal')
            self.audio_model.set_mode('multimodal')
        
        progress_bar = tqdm(dataloader, desc=f"Training {modality}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move data to device
            src_tokens = batch['src_tokens'].to(self.device)
            tgt_tokens = batch['tgt_tokens'].to(self.device)
            src_length = batch['src_length'].to(self.device)
            tgt_length = batch['tgt_length'].to(self.device)
            
            # Forward pass based on modality
            if modality == 'text':
                outputs = self.text_model(src_tokens, tgt_tokens[:, :-1], src_length, tgt_length-1)
                
            elif modality == 'image':
                images = batch['image'].to(self.device)
                outputs = self.image_model(src_tokens, tgt_tokens[:, :-1], src_images=images)
                
            elif modality == 'audio':
                audio = batch['audio'].to(self.device)
                outputs = self.audio_model(src_tokens, tgt_tokens[:, :-1], src_audio=audio)
                
            elif modality == 'multimodal':
                # For multimodal, we'll use a combination approach
                images = batch['image'].to(self.device)
                audio = batch['audio'].to(self.device)
                
                # Get outputs from both models and combine
                img_outputs = self.image_model(src_tokens, tgt_tokens[:, :-1], src_images=images)
                aud_outputs = self.audio_model(src_tokens, tgt_tokens[:, :-1], src_audio=audio)
                
                # Simple averaging for now (could be more sophisticated)
                outputs = (img_outputs + aud_outputs) / 2
            
            # Calculate loss
            loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt_tokens[:, 1:].reshape(-1))
            
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
            mask = (tgt_tokens[:, 1:] != self.tokenizer.pad_id)
            correct = (pred_tokens == tgt_tokens[:, 1:]) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0
            })
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, dataloader: DataLoader, model, modality: str) -> Dict[str, float]:
        """Validate a model."""
        model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        if modality == 'text':
            self.text_model.set_mode('text')
        elif modality == 'image':
            self.image_model.set_mode('image')
        elif modality == 'audio':
            self.audio_model.set_mode('audio')
        elif modality == 'multimodal':
            self.image_model.set_mode('multimodal')
            self.audio_model.set_mode('multimodal')
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validating {modality}"):
                # Move data to device
                src_tokens = batch['src_tokens'].to(self.device)
                tgt_tokens = batch['tgt_tokens'].to(self.device)
                src_length = batch['src_length'].to(self.device)
                tgt_length = batch['tgt_length'].to(self.device)
                
                # Forward pass based on modality
                if modality == 'text':
                    outputs = self.text_model(src_tokens, tgt_tokens[:, :-1], src_length, tgt_length-1)
                    
                elif modality == 'image':
                    images = batch['image'].to(self.device)
                    outputs = self.image_model(src_tokens, tgt_tokens[:, :-1], src_images=images)
                    
                elif modality == 'audio':
                    audio = batch['audio'].to(self.device)
                    outputs = self.audio_model(src_tokens, tgt_tokens[:, :-1], src_audio=audio)
                    
                elif modality == 'multimodal':
                    images = batch['image'].to(self.device)
                    audio = batch['audio'].to(self.device)
                    
                    img_outputs = self.image_model(src_tokens, tgt_tokens[:, :-1], src_images=images)
                    aud_outputs = self.audio_model(src_tokens, tgt_tokens[:, :-1], src_audio=audio)
                    outputs = (img_outputs + aud_outputs) / 2
                
                # Calculate loss
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt_tokens[:, 1:].reshape(-1))
                total_loss += loss.item()
                
                # Calculate accuracy
                pred_tokens = outputs.argmax(dim=-1)
                mask = (tgt_tokens[:, 1:] != self.tokenizer.pad_id)
                correct = (pred_tokens == tgt_tokens[:, 1:]) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train_multimodal_system(self, 
                                 train_data: List[Dict[str, str]], 
                                 val_data: List[Dict[str, str]],
                                 num_epochs: int = 10,
                                 batch_size: int = 32,
                                 save_dir: str = "models/checkpoints") -> Dict[str, List[float]]:
        """Train all multimodal models."""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create datasets
        train_dataset = MultimodalDataset(train_data, self.tokenizer, include_images=True, include_audio=True)
        val_dataset = MultimodalDataset(val_data, self.tokenizer, include_images=True, include_audio=True)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Training history
        history = {
            'text_train_loss': [], 'text_train_acc': [], 'text_val_loss': [], 'text_val_acc': [],
            'image_train_loss': [], 'image_train_acc': [], 'image_val_loss': [], 'image_val_acc': [],
            'audio_train_loss': [], 'audio_train_acc': [], 'audio_val_loss': [], 'audio_val_acc': []
        }
        
        self.logger.info(f"Starting multimodal training for {num_epochs} epochs")
        self.logger.info(f"Training data: {len(train_data)} samples")
        self.logger.info(f"Validation data: {len(val_data)} samples")
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train text model
            self.logger.info("Training text model...")
            text_train_metrics = self.train_epoch(train_loader, self.text_model, self.text_optimizer, self.text_scheduler, 'text')
            text_val_metrics = self.validate(val_loader, self.text_model, 'text')
            
            # Train image model
            self.logger.info("Training image model...")
            image_train_metrics = self.train_epoch(train_loader, self.image_model, self.image_optimizer, self.image_scheduler, 'image')
            image_val_metrics = self.validate(val_loader, self.image_model, 'image')
            
            # Train audio model
            self.logger.info("Training audio model...")
            audio_train_metrics = self.train_epoch(train_loader, self.audio_model, self.audio_optimizer, self.audio_scheduler, 'audio')
            audio_val_metrics = self.validate(val_loader, self.audio_model, 'audio')
            
            # Update history
            history['text_train_loss'].append(text_train_metrics['loss'])
            history['text_train_acc'].append(text_train_metrics['accuracy'])
            history['text_val_loss'].append(text_val_metrics['loss'])
            history['text_val_acc'].append(text_val_metrics['accuracy'])
            
            history['image_train_loss'].append(image_train_metrics['loss'])
            history['image_train_acc'].append(image_train_metrics['accuracy'])
            history['image_val_loss'].append(image_val_metrics['loss'])
            history['image_val_acc'].append(image_val_metrics['accuracy'])
            
            history['audio_train_loss'].append(audio_train_metrics['loss'])
            history['audio_train_acc'].append(audio_train_metrics['accuracy'])
            history['audio_val_loss'].append(audio_val_metrics['loss'])
            history['audio_val_acc'].append(audio_val_metrics['accuracy'])
            
            # Log epoch results
            self.logger.info(f"Text - Train Loss: {text_train_metrics['loss']:.4f}, Train Acc: {text_train_metrics['accuracy']:.4f}, "
                           f"Val Loss: {text_val_metrics['loss']:.4f}, Val Acc: {text_val_metrics['accuracy']:.4f}")
            self.logger.info(f"Image - Train Loss: {image_train_metrics['loss']:.4f}, Train Acc: {image_train_metrics['accuracy']:.4f}, "
                           f"Val Loss: {image_val_metrics['loss']:.4f}, Val Acc: {image_val_metrics['accuracy']:.4f}")
            self.logger.info(f"Audio - Train Loss: {audio_train_metrics['loss']:.4f}, Train Acc: {audio_train_metrics['accuracy']:.4f}, "
                           f"Val Loss: {audio_val_metrics['loss']:.4f}, Val Acc: {audio_val_metrics['accuracy']:.4f}")
            
            # Save checkpoints
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1, save_dir)
        
        # Save final models
        self.save_final_models(save_dir)
        
        return history
    
    def save_checkpoint(self, epoch: int, save_dir: Path):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'text_model_state_dict': self.text_model.state_dict(),
            'image_model_state_dict': self.image_model.state_dict(),
            'audio_model_state_dict': self.audio_model.state_dict(),
            'text_optimizer_state_dict': self.text_optimizer.state_dict(),
            'image_optimizer_state_dict': self.image_optimizer.state_dict(),
            'audio_optimizer_state_dict': self.audio_optimizer.state_dict(),
            'tokenizer_config': {
                'vocab_size': self.tokenizer.vocab_size,
                'pad_id': self.tokenizer.pad_id,
                'bos_id': self.tokenizer.bos_id,
                'eos_id': self.tokenizer.eos_id,
                'unk_id': self.tokenizer.unk_id
            }
        }
        
        checkpoint_path = save_dir / f"multimodal_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def save_final_models(self, save_dir: Path):
        """Save final trained models."""
        # Save text model
        text_model_path = save_dir / "nmt_transformer_best.pt"
        torch.save({
            'model_state_dict': self.text_model.state_dict(),
            'config': {
                'src_vocab_size': self.text_model.src_vocab_size,
                'tgt_vocab_size': self.text_model.tgt_vocab_size,
                'd_model': self.text_model.d_model,
                'n_heads': self.text_model.n_heads,
                'n_encoder_layers': self.text_model.n_encoder_layers,
                'n_decoder_layers': self.text_model.n_decoder_layers,
                'd_ff': self.text_model.d_ff,
                'max_len': self.text_model.max_len,
                'dropout': self.text_model.dropout,
                'pad_id': self.text_model.pad_id,
                'use_flash': self.text_model.use_flash
            }
        }, text_model_path)
        
        # Save image model
        image_model_path = save_dir / "multimodal_image_best.pt"
        torch.save({
            'model_state_dict': self.image_model.state_dict(),
            'config': {
                'src_vocab_size': self.image_model.text_nmt.src_vocab_size,
                'tgt_vocab_size': self.image_model.text_nmt.tgt_vocab_size,
                'd_model': self.image_model.text_nmt.d_model,
                'n_heads': self.image_model.text_nmt.n_heads,
                'n_encoder_layers': self.image_model.text_nmt.n_encoder_layers,
                'n_decoder_layers': self.image_model.text_nmt.n_decoder_layers,
                'd_ff': self.image_model.text_nmt.d_ff,
                'max_len': self.image_model.text_nmt.max_len,
                'dropout': self.image_model.text_nmt.dropout,
                'pad_id': self.image_model.text_nmt.pad_id,
                'use_flash': self.image_model.text_nmt.use_flash
            }
        }, image_model_path)
        
        # Save audio model
        audio_model_path = save_dir / "multimodal_audio_best.pt"
        torch.save({
            'model_state_dict': self.audio_model.state_dict(),
            'config': {
                'src_vocab_size': self.audio_model.text_nmt.src_vocab_size,
                'tgt_vocab_size': self.audio_model.text_nmt.tgt_vocab_size,
                'd_model': self.audio_model.text_nmt.d_model,
                'n_heads': self.audio_model.text_nmt.n_heads,
                'n_encoder_layers': self.audio_model.text_nmt.n_encoder_layers,
                'n_decoder_layers': self.audio_model.text_nmt.n_decoder_layers,
                'd_ff': self.audio_model.text_nmt.d_ff,
                'max_len': self.audio_model.text_nmt.max_len,
                'dropout': self.audio_model.text_nmt.dropout,
                'pad_id': self.audio_model.text_nmt.pad_id,
                'use_flash': self.audio_model.text_nmt.use_flash
            }
        }, audio_model_path)
        
        self.logger.info(f"Final models saved to {save_dir}")


def create_training_data() -> Tuple[List[Dict], List[Dict]]:
    """Create training and validation data."""
    # Korean-English sentence pairs for training
    training_pairs = [
        {"korean": "안녕하세요", "english": "Hello"},
        {"korean": "감사합니다", "english": "Thank you"},
        {"korean": "죄송합니다", "english": "Sorry"},
        {"korean": "네", "english": "Yes"},
        {"korean": "아니요", "english": "No"},
        {"korean": "오늘 날씨가 좋네요", "english": "The weather is nice today"},
        {"korean": "밥 먹었어요?", "english": "Did you eat?"},
        {"korean": "어디 가세요?", "english": "Where are you going?"},
        {"korean": "잘 지내셨어요?", "english": "Have you been well?"},
        {"korean": "저는 한국어를 배우고 있어요", "english": "I am learning Korean"},
        {"korean": "이 책은 정말 흥미로워요", "english": "This book is really interesting"},
        {"korean": "내일 학교에 가야 해요", "english": "I have to go to school tomorrow"},
        {"korean": "커피 마시고 싶어요", "english": "I want to drink coffee"},
        {"korean": "회의는 오후 3시에 시작됩니다", "english": "The meeting starts at 3 PM"},
        {"korean": "프로젝트 일정을 확인해 주세요", "english": "Please check the project schedule"},
        {"korean": "보고서를 제출해야 합니다", "english": "I need to submit the report"},
        {"korean": "많이 드세요", "english": "Please eat a lot"},
        {"korean": "수고하셨습니다", "english": "Thank you for your hard work"},
        {"korean": "들어오세요", "english": "Please come in"},
        {"korean": "이것은 무엇입니까?", "english": "What is this?"},
    ]
    
    # Validation pairs
    validation_pairs = [
        {"korean": "언제 도착했어요?", "english": "When did you arrive?"},
        {"korean": "어떻게 가요?", "english": "How do I get there?"},
        {"korean": "누구세요?", "english": "Who are you?"},
        {"korean": "어제 영화를 봤어요", "english": "I watched a movie yesterday"},
        {"korean": "지금 공부하고 있어요", "english": "I am studying now"},
        {"korean": "내일 친구를 만날 거예요", "english": "I will meet my friend tomorrow"},
        {"korean": "선생님, 질문이 있어요", "english": "Teacher, I have a question"},
        {"korean": "부모님께 감사드립니다", "english": "I thank my parents"},
        {"korean": "할아버지께 안부를 전해 주세요", "english": "Please give my regards to grandfather"},
        {"korean": "사과 두 개 주세요", "english": "Please give me two apples"},
    ]
    
    return training_pairs, validation_pairs


def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create training data
    print("Creating training and validation data...")
    train_data, val_data = create_training_data()
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create simple tokenizer for testing
    class SimpleTokenizer:
        def __init__(self):
            self.pad_id = 0
            self.unk_id = 1
            self.bos_id = 2
            self.eos_id = 3
            
            # Build vocabulary from data
            self.vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
            
            for item in train_data + val_data:
                for char in item['korean'] + item['english']:
                    if char not in self.vocab:
                        self.vocab[char] = len(self.vocab)
        
        def encode(self, text):
            tokens = [self.bos_id]
            for char in text:
                tokens.append(self.vocab.get(char, self.unk_id))
            tokens.append(self.eos_id)
            return tokens
        
        def decode(self, tokens):
            reverse_vocab = {v: k for k, v in self.vocab.items()}
            result = []
            for token in tokens:
                if token in [self.pad_id, self.bos_id, self.eos_id]:
                    continue
                result.append(reverse_vocab.get(token, '<unk>'))
            return ''.join(result)
        
        @property
        def vocab_size(self):
            return len(self.vocab)
    
    tokenizer = SimpleTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Text model
    text_model = NMTTransformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=512,
        max_len=64,
        dropout=0.1,
        pad_id=tokenizer.pad_id,
        use_flash=False
    ).to(device)
    
    # Image model
    image_model = EnhancedMultimodalNMT(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=512,
        max_len=64,
        dropout=0.1,
        pad_id=tokenizer.pad_id,
        use_flash=False,
        img_size=64,
        patch_size=8,
        img_embed_dim=128,
        img_num_heads=4,
        img_num_layers=2,
        fusion_dim=256,
        fusion_heads=4
    ).to(device)
    
    # Audio model
    audio_model = MultimodalAudioNMT(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=512,
        max_len=64,
        dropout=0.1,
        pad_id=tokenizer.pad_id,
        use_flash=False,
        audio_encoder_type='cnn',
        audio_output_dim=256,
        fusion_dim=256,
        fusion_heads=4
    ).to(device)
    
    # Create trainer
    trainer = MultimodalTrainer(
        text_model=text_model,
        image_model=image_model,
        audio_model=audio_model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Train models
    print("\nStarting multimodal training...")
    history = trainer.train_multimodal_system(
        train_data=train_data,
        val_data=val_data,
        num_epochs=5,
        batch_size=4,
        save_dir="models/checkpoints"
    )
    
    print("\nTraining completed!")
    print("Models saved to models/checkpoints/")
    
    return history


if __name__ == "__main__":
    main()