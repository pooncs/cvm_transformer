import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import json
import os
from pathlib import Path
import logging
from tqdm import tqdm
import wandb

from ..models.multimodal_nmt import create_multimodal_nmt_model
from ..data.prepare_corpus import ParallelCorpusProcessor
from ..utils.metrics import BLEUScore, ExactMatchScore, CharacterErrorRate
from ..utils.checkpointing import save_checkpoint, load_checkpoint


class MultimodalTranslationDataset(Dataset):
    """Dataset for multimodal translation with text, images, and audio."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, 
                 max_src_len: int = 512, max_tgt_len: int = 512,
                 image_size: int = 224, audio_max_len: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.image_size = image_size
        self.audio_max_len = audio_max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        src_tokens = self.tokenizer.encode(
            item['source'], out_type=int, add_bos=True, add_eos=True
        )[:self.max_src_len]
        
        tgt_tokens = self.tokenizer.encode(
            item['target'], out_type=int, add_bos=True, add_eos=True
        )[:self.max_tgt_len]
        
        # Pad sequences
        src_tokens = self._pad_sequence(src_tokens, self.max_src_len)
        tgt_tokens = self._pad_sequence(tgt_tokens, self.max_tgt_len)
        
        result = {
            'src_tokens': torch.tensor(src_tokens, dtype=torch.long),
            'tgt_tokens': torch.tensor(tgt_tokens, dtype=torch.long),
            'src_length': len([t for t in src_tokens if t != 0]),
            'tgt_length': len([t for t in tgt_tokens if t != 0]),
            'domain': item.get('domain', 'general'),
            'difficulty': item.get('difficulty', 1.0)
        }
        
        # Add image data if available
        if 'image_path' in item:
            result['image'] = self._load_image(item['image_path'])
            result['has_image'] = True
        else:
            result['has_image'] = False
            
        # Add audio data if available
        if 'audio_path' in item:
            result['audio'] = self._load_audio(item['audio_path'])
            result['audio_length'] = item.get('audio_length', self.audio_max_len)
            result['has_audio'] = True
        else:
            result['has_audio'] = False
            
        return result
    
    def _pad_sequence(self, sequence, max_len, pad_id=0):
        if len(sequence) < max_len:
            sequence = sequence + [pad_id] * (max_len - len(sequence))
        return sequence[:max_len]
    
    def _load_image(self, image_path):
        """Load and preprocess image."""
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(image_path).convert('RGB')
            return transform(image)
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio."""
        try:
            import librosa
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Truncate or pad to max length
            if len(audio) > self.audio_max_len:
                audio = audio[:self.audio_max_len]
            else:
                audio = np.pad(audio, (0, self.audio_max_len - len(audio)))
            
            return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            logging.warning(f"Failed to load audio {audio_path}: {e}")
            return torch.zeros(1, self.audio_max_len)


class MultimodalNMTTrainer:
    """Trainer for multimodal neural machine translation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizers
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        self.translation_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.char_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.phoneme_loss = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Metrics
        self.bleu_metric = BLEUScore()
        self.exact_match_metric = ExactMatchScore()
        self.cer_metric = CharacterErrorRate()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_bleu = 0.0
        self.best_exact_match = 0.0
        
        # Logging
        self.setup_logging()
        
        # Initialize wandb if available
        if config.get('use_wandb', True):
            wandb.init(project="multimodal-nmt", config=config)
    
    def _create_model(self):
        """Create multimodal NMT model."""
        model = create_multimodal_nmt_model(
            src_vocab_size=self.config['src_vocab_size'],
            tgt_vocab_size=self.config['tgt_vocab_size'],
            d_model=self.config.get('d_model', 1024),
            n_heads=self.config.get('n_heads', 16),
            n_encoder_layers=self.config.get('n_encoder_layers', 12),
            n_decoder_layers=self.config.get('n_decoder_layers', 12),
            d_ff=self.config.get('d_ff', 4096),
            max_len=self.config.get('max_len', 512),
            dropout=self.config.get('dropout', 0.1),
            use_images=self.config.get('use_images', True),
            use_audio=self.config.get('use_audio', True),
            fusion_strategy=self.config.get('fusion_strategy', 'cross_attention')
        )
        
        return model.to(self.device)
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for different components."""
        param_groups = [
            {'params': self.model.nmt_model.parameters(), 'lr': self.config.get('lr', 1e-4)},
            {'params': self.model.multimodal_modules.parameters(), 'lr': self.config.get('multimodal_lr', 5e-5)},
            {'params': self.model.multimodal_encoder.parameters(), 'lr': self.config.get('encoder_lr', 1e-4)},
            {'params': self.model.decoder_fusion.parameters(), 'lr': self.config.get('fusion_lr', 1e-4)},
        ]
        
        if self.config.get('optimizer', 'adamw') == 'adamw':
            return optim.AdamW(
                param_groups,
                weight_decay=self.config.get('weight_decay', 0.01),
                betas=(0.9, 0.98)
            )
        else:
            return optim.Adam(param_groups)
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = self.config.get('max_epochs', 100) * self.config.get('steps_per_epoch', 1000)
        warmup_steps = self.config.get('warmup_steps', 4000)
        
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('T_0', 10),
            T_mult=self.config.get('T_mult', 2),
            eta_min=self.config.get('min_lr', 1e-6)
        )
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with curriculum learning."""
        self.model.train()
        epoch_metrics = {'train_loss': 0.0, 'train_bleu': 0.0, 'train_em': 0.0}
        
        # Curriculum learning: sort by difficulty
        if self.config.get('curriculum_learning', True):
            train_loader.dataset.data.sort(key=lambda x: x.get('difficulty', 1.0))
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            outputs = self.model(
                src_tokens=batch['src_tokens'],
                tgt_tokens=batch['tgt_tokens'],
                images=batch.get('image'),
                audio=batch.get('audio'),
                src_lengths=batch.get('src_length'),
                audio_lengths=batch.get('audio_length')
            )
            
            # Compute losses
            loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            metrics = self._compute_metrics(outputs, batch)
            
            epoch_metrics['train_loss'] += loss.item()
            epoch_metrics['train_bleu'] += metrics['bleu']
            epoch_metrics['train_em'] += metrics['exact_match']
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bleu': f'{metrics["bleu"]:.4f}',
                'em': f'{metrics["exact_match"]:.4f}'
            })
            
            self.global_step += 1
            
            # Validation at intervals
            if (batch_idx + 1) % self.config.get('val_interval', 100) == 0:
                val_metrics = self.validate(val_loader)
                self.model.train()
                
                # Log to wandb
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/bleu': metrics['bleu'],
                        'train/exact_match': metrics['exact_match'],
                        'val/loss': val_metrics['val_loss'],
                        'val/bleu': val_metrics['val_bleu'],
                        'val/exact_match': val_metrics['val_em'],
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
        
        # Average metrics
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_metrics = {'val_loss': 0.0, 'val_bleu': 0.0, 'val_em': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                batch = self._move_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    src_tokens=batch['src_tokens'],
                    tgt_tokens=batch['tgt_tokens'],
                    images=batch.get('image'),
                    audio=batch.get('audio'),
                    src_lengths=batch.get('src_length'),
                    audio_lengths=batch.get('audio_length')
                )
                
                # Compute losses and metrics
                loss = self._compute_loss(outputs, batch)
                metrics = self._compute_metrics(outputs, batch)
                
                val_metrics['val_loss'] += loss.item()
                val_metrics['val_bleu'] += metrics['bleu']
                val_metrics['val_em'] += metrics['exact_match']
        
        # Average metrics
        num_batches = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
            
        return val_metrics
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multimodal loss."""
        # Translation loss
        logits = outputs['logits']
        tgt_tokens = batch['tgt_tokens'][:, 1:]  # Remove BOS
        
        # Reshape for loss computation
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt_tokens.reshape(-1)
        
        translation_loss = self.translation_loss(logits_flat, tgt_flat)
        
        # Character detection loss (if available)
        char_loss = 0.0
        if 'char_logits' in outputs and batch.get('has_image', False):
            char_logits = outputs['char_logits']
            # Create dummy targets for character detection (this should be replaced with real labels)
            char_targets = torch.zeros_like(char_logits[..., 0], dtype=torch.long)
            char_loss = self.char_loss(
                char_logits.reshape(-1, char_logits.size(-1)),
                char_targets.reshape(-1)
            ) * 0.1  # Weight for auxiliary loss
        
        # Phoneme classification loss (if available)
        phoneme_loss = 0.0
        if 'phoneme_logits' in outputs and batch.get('has_audio', False):
            phoneme_logits = outputs['phoneme_logits']
            # Create dummy targets for phoneme classification
            phoneme_targets = torch.zeros_like(phoneme_logits[..., 0], dtype=torch.long)
            phoneme_loss = self.phoneme_loss(
                phoneme_logits.reshape(-1, phoneme_logits.size(-1)),
                phoneme_targets.reshape(-1)
            ) * 0.1  # Weight for auxiliary loss
        
        total_loss = translation_loss + char_loss + phoneme_loss
        
        return total_loss
    
    def _compute_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Get predictions
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=-1)
        
        # Get targets
        tgt_tokens = batch['tgt_tokens'][:, 1:]  # Remove BOS
        
        # Convert to text for BLEU calculation
        pred_texts = self._tokens_to_text(predictions)
        tgt_texts = self._tokens_to_text(tgt_tokens)
        
        # Compute BLEU
        bleu = self.bleu_metric(pred_texts, tgt_texts)
        
        # Compute exact match
        exact_match = self.exact_match_metric(pred_texts, tgt_texts)
        
        return {
            'bleu': bleu,
            'exact_match': exact_match
        }
    
    def _tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        """Convert token tensors to text."""
        texts = []
        for seq in tokens:
            # Remove padding and special tokens
            seq = seq[seq != 0]  # Remove padding
            seq = seq[seq != 2]  # Remove BOS
            seq = seq[seq != 3]  # Remove EOS
            
            if len(seq) > 0:
                text = self.tokenizer.decode(seq.tolist())
                texts.append(text)
            else:
                texts.append("")
        return texts
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def train(self, train_data: List[Dict[str, str]], val_data: List[Dict[str, str]]) -> Dict[str, float]:
        """Main training loop."""
        self.logger.info("Starting multimodal NMT training...")
        
        # Create datasets
        train_dataset = MultimodalTranslationDataset(
            train_data, self.tokenizer, 
            max_src_len=self.config.get('max_src_len', 512),
            max_tgt_len=self.config.get('max_tgt_len', 512)
        )
        val_dataset = MultimodalTranslationDataset(
            val_data, self.tokenizer,
            max_src_len=self.config.get('max_src_len', 512),
            max_tgt_len=self.config.get('max_tgt_len', 512)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('val_batch_size', 64),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Training loop
        best_val_bleu = 0.0
        patience = self.config.get('patience', 10)
        patience_counter = 0
        
        for epoch in range(self.config.get('max_epochs', 100)):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, val_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch + 1}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train BLEU: {train_metrics['train_bleu']:.4f}, "
                f"Train EM: {train_metrics['train_em']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val BLEU: {val_metrics['val_bleu']:.4f}, "
                f"Val EM: {val_metrics['val_em']:.4f}"
            )
            
            # Save checkpoint if best
            if val_metrics['val_bleu'] > best_val_bleu:
                best_val_bleu = val_metrics['val_bleu']
                self.best_bleu = best_val_bleu
                self.best_exact_match = val_metrics['val_em']
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_bleu': best_val_bleu,
                    'best_exact_match': self.best_exact_match,
                    'config': self.config
                }, f"checkpoint_best.pth")
                
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        self.logger.info(f"Training completed. Best BLEU: {best_val_bleu:.4f}")
        
        return {
            'best_bleu': best_val_bleu,
            'best_exact_match': self.best_exact_match
        }


def main():
    """Main function for training multimodal NMT."""
    # Configuration
    config = {
        'src_vocab_size': 32000,
        'tgt_vocab_size': 32000,
        'd_model': 1024,
        'n_heads': 16,
        'n_encoder_layers': 12,
        'n_decoder_layers': 12,
        'd_ff': 4096,
        'max_len': 512,
        'dropout': 0.1,
        'batch_size': 32,
        'val_batch_size': 64,
        'max_epochs': 100,
        'lr': 1e-4,
        'multimodal_lr': 5e-5,
        'encoder_lr': 1e-4,
        'fusion_lr': 1e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'warmup_steps': 4000,
        'T_0': 10,
        'T_mult': 2,
        'min_lr': 1e-6,
        'patience': 10,
        'val_interval': 100,
        'use_images': True,
        'use_audio': True,
        'fusion_strategy': 'cross_attention',
        'curriculum_learning': True,
        'use_wandb': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'log_dir': 'logs',
        'max_src_len': 512,
        'max_tgt_len': 512
    }
    
    # Load data (this should be replaced with actual data loading)
    # For now, create dummy data for demonstration
    train_data = create_dummy_multimodal_data(1000)
    val_data = create_dummy_multimodal_data(200)
    
    # Create trainer
    trainer = MultimodalNMTTrainer(config)
    
    # Train model
    results = trainer.train(train_data, val_data)
    
    print(f"Training completed!")
    print(f"Best BLEU score: {results['best_bleu']:.4f}")
    print(f"Best Exact Match: {results['best_exact_match']:.4f}")


def create_dummy_multimodal_data(n_samples: int) -> List[Dict[str, Any]]:
    """Create dummy multimodal data for testing."""
    data = []
    
    korean_sentences = [
        "안녕하세요",
        "감사합니다",
        "죄송합니다",
        "네, 알겠습니다",
        "아니요, 괜찮습니다",
        "도와주세요",
        "어디에 있나요?",
        "얼마예요?",
        "맛있어요",
        "추워요"
    ]
    
    english_translations = [
        "Hello",
        "Thank you",
        "I'm sorry",
        "Yes, I understand",
        "No, it's okay",
        "Help me",
        "Where is it?",
        "How much is it?",
        "It's delicious",
        "It's cold"
    ]
    
    for i in range(n_samples):
        idx = i % len(korean_sentences)
        
        item = {
            'source': korean_sentences[idx],
            'target': english_translations[idx],
            'domain': 'daily_conversation',
            'difficulty': np.random.uniform(0.1, 1.0)
        }
        
        # Add image path for 70% of samples
        if np.random.random() < 0.7:
            item['image_path'] = f'data/images/sample_{i % 100}.jpg'
        
        # Add audio path for 60% of samples
        if np.random.random() < 0.6:
            item['audio_path'] = f'data/audio/sample_{i % 100}.wav'
            item['audio_length'] = np.random.randint(1000, 5000)
        
        data.append(item)
    
    return data


if __name__ == '__main__':
    main()