import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import random
from tqdm import tqdm

# Add project root to Python path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our models
from src.models.nmt_transformer import NMTTransformer, create_nmt_model
from src.models.image_encoder import create_korean_image_encoder
from src.models.audio_encoder import create_korean_audio_encoder
from src.models.multimodal_fusion import create_multimodal_korean_encoder
from src.utils.metrics import BLEUScore, ExactMatchScore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalTranslationDataset(Dataset):
    """Dataset for multimodal Korean-English translation"""
    
    def __init__(self, 
                 text_data: List[Dict[str, str]],
                 image_data: Optional[Dict[str, torch.Tensor]] = None,
                 audio_data: Optional[Dict[str, torch.Tensor]] = None,
                 max_length: int = 128,
                 tokenizer=None):
        self.text_data = text_data
        self.image_data = image_data or {}
        self.audio_data = audio_data or {}
        self.max_length = max_length
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        item = self.text_data[idx]
        korean_text = item['korean']
        english_text = item['english']
        
        # Tokenize text (simplified - use proper tokenizer in production)
        korean_tokens = self.tokenize_text(korean_text)
        english_tokens = self.tokenize_text(english_text)
        
        # Get multimodal data if available
        image_tensor = self.image_data.get(korean_text, torch.zeros(1, 3, 224, 224))
        audio_tensor = self.audio_data.get(korean_text, torch.zeros(1, 48000))  # 3 seconds at 16kHz
        
        return {
            'korean_tokens': torch.tensor(korean_tokens, dtype=torch.long),
            'english_tokens': torch.tensor(english_tokens, dtype=torch.long),
            'korean_text': korean_text,
            'english_text': english_text,
            'image_tensor': image_tensor,
            'audio_tensor': audio_tensor,
            'has_image': korean_text in self.image_data,
            'has_audio': korean_text in self.audio_data
        }
    
    def tokenize_text(self, text: str) -> List[int]:
        """Simple tokenization (replace with proper tokenizer)"""
        # This is a placeholder - use proper SentencePiece tokenizer
        tokens = [ord(c) % 10000 for c in text[:self.max_length]]
        return tokens + [0] * (self.max_length - len(tokens))  # Pad

class MultimodalNMTTrainer:
    """Trainer for multimodal Korean-English NMT"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_models()
        self.setup_training_components()
        self.setup_metrics()
        
    def setup_models(self):
        """Initialize all models"""
        logger.info("Setting up multimodal models...")
        
        # Text-only NMT model
        self.text_model = create_nmt_model(
            src_vocab_size=self.config['model']['vocab_size'],
            tgt_vocab_size=self.config['model']['vocab_size'],
            d_model=self.config['model']['d_model'],
            n_heads=self.config['model']['n_heads'],
            n_encoder_layers=self.config['model']['n_encoder_layers'],
            n_decoder_layers=self.config['model']['n_decoder_layers'],
            d_ff=self.config['model']['d_ff'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # Multimodal encoders
        self.image_encoder = create_korean_image_encoder(
            self.config['multimodal']['image_encoder']
        ).to(self.device)
        
        self.audio_encoder = create_korean_audio_encoder(
            self.config['multimodal']['audio_encoder']
        ).to(self.device)
        
        self.multimodal_encoder = create_multimodal_korean_encoder({
            'text_vocab_size': self.config['model']['vocab_size'],
            'embed_dim': self.config['multimodal']['fusion']['embed_dim'],
            'num_heads': self.config['multimodal']['fusion']['num_heads'],
            'num_layers': self.config['multimodal']['fusion']['num_layers']
        }).to(self.device)
        
        # Multimodal NMT model (decoder-only, uses multimodal encoder)
        self.multimodal_model = NMTTransformer(
            src_vocab_size=self.config['model']['vocab_size'],
            tgt_vocab_size=self.config['model']['vocab_size'],
            d_model=self.config['model']['d_model'],
            n_heads=self.config['model']['n_heads'],
            n_encoder_layers=0,  # We use multimodal encoder instead
            n_decoder_layers=self.config['model']['n_decoder_layers'],
            d_ff=self.config['model']['d_ff'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        logger.info("Models initialized successfully")
    
    def setup_training_components(self):
        """Setup optimizers, schedulers, and loss functions"""
        logger.info("Setting up training components...")
        
        # Optimizers
        self.text_optimizer = optim.AdamW(
            self.text_model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.multimodal_optimizer = optim.AdamW(
            list(self.multimodal_model.parameters()) +
            list(self.image_encoder.parameters()) +
            list(self.audio_encoder.parameters()) +
            list(self.multimodal_encoder.parameters()),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Schedulers
        self.text_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.text_optimizer,
            T_0=self.config['training']['scheduler_t0'],
            T_mult=self.config['training']['scheduler_t_mult']
        )
        
        self.multimodal_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.multimodal_optimizer,
            T_0=self.config['training']['scheduler_t0'],
            T_mult=self.config['training']['scheduler_t_mult']
        )
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # pad token
            label_smoothing=self.config['training']['label_smoothing']
        )
        
        logger.info("Training components initialized")
    
    def setup_metrics(self):
        """Setup evaluation metrics"""
        self.bleu_score = BLEUScore()
        self.exact_match = ExactMatchScore()
    
    def create_enhanced_training_data(self) -> List[Dict[str, str]]:
        """Create enhanced training dataset for 99% accuracy target"""
        logger.info("Creating enhanced training dataset...")
        
        # Base dataset with perfect translations
        base_data = [
            # Greetings and basic phrases
            {"korean": "안녕하세요", "english": "Hello"},
            {"korean": "감사합니다", "english": "Thank you"},
            {"korean": "사랑해요", "english": "I love you"},
            {"korean": "미안합니다", "english": "I'm sorry"},
            {"korean": "괜찮아요", "english": "It's okay"},
            
            # Daily life
            {"korean": "학교에 가요", "english": "I go to school"},
            {"korean": "밥 먹었어요?", "english": "Did you eat?"},
            {"korean": "날씨가 좋네요", "english": "The weather is nice"},
            {"korean": "오늘은 바빠요", "english": "I'm busy today"},
            {"korean": "내일 봐요", "english": "See you tomorrow"},
            
            # Requests and questions
            {"korean": "도와주세요", "english": "Please help me"},
            {"korean": "얼마예요?", "english": "How much is it?"},
            {"korean": "어디에 있어요?", "english": "Where is it?"},
            {"korean": "언제예요?", "english": "When is it?"},
            {"korean": "누구예요?", "english": "Who is it?"},
            
            # Comprehension and responses
            {"korean": "이해했어요", "english": "I understand"},
            {"korean": "몰라요", "english": "I don't know"},
            {"korean": "알겠어요", "english": "I got it"},
            {"korean": "맞아요", "english": "That's right"},
            {"korean": "틀려요", "english": "That's wrong"}
        ]
        
        # Domain-specific data
        business_data = [
            {"korean": "회의가 몇 시에 있나요?", "english": "What time is the meeting?"},
            {"korean": "보고서를 제출했습니다", "english": "I submitted the report"},
            {"korean": "프로젝트 일정을 논의하자", "english": "Let's discuss the project schedule"},
            {"korean": "예산을 검토해야 합니다", "english": "We need to review the budget"},
            {"korean": "고객 요구사항을 분석하세요", "english": "Analyze the customer requirements"},
            {"korean": "마감일이 언제인가요?", "english": "When is the deadline?"},
            {"korean": "업무 진행 상황을 알려주세요", "english": "Please update me on the progress"},
            {"korean": "팀 미팅을 잡읍시다", "english": "Let's schedule a team meeting"},
            {"korean": "계약서를 준비하세요", "english": "Prepare the contract"},
            {"korean": "협상이 잘 되었습니다", "english": "The negotiation went well"}
        ]
        
        medical_data = [
            {"korean": "어디가 아프세요?", "english": "Where does it hurt?"},
            {"korean": "약을 복용하세요", "english": "Take the medicine"},
            {"korean": "건강검진을 받으세요", "english": "Get a health checkup"},
            {"korean": "증상이 호전되었어요", "english": "The symptoms have improved"},
            {"korean": "의사 선생님께 상담받으세요", "english": "Consult with the doctor"},
            {"korean": "처방전을 작성해 드리겠습니다", "english": "I'll write you a prescription"},
            {"korean": "검사 결과를 확인해야 합니다", "english": "We need to check the test results"},
            {"korean": "수술이 필요할 수 있습니다", "english": "You may need surgery"},
            {"korean": "휴식이 중요합니다", "english": "Rest is important"},
            {"korean": "회복이 빠르시네요", "english": "You're recovering quickly"}
        ]
        
        technology_data = [
            {"korean": "컴퓨터가 고장났어요", "english": "The computer is broken"},
            {"korean": "소프트웨어를 업데이트하세요", "english": "Update the software"},
            {"korean": "인터넷 연결이 끊겼어요", "english": "The internet connection is lost"},
            {"korean": "데이터를 백업하세요", "english": "Back up the data"},
            {"korean": "보안 설정을 확인하세요", "english": "Check the security settings"},
            {"korean": "시스템을 재시작하세요", "english": "Restart the system"},
            {"korean": "바이러스를 제거했습니다", "english": "I removed the virus"},
            {"korean": "네트워크 속도가 느려요", "english": "The network speed is slow"},
            {"korean": "프로그램을 설치하세요", "english": "Install the program"},
            {"korean": "하드웨어를 교체해야 합니다", "english": "We need to replace the hardware"}
        ]
        
        education_data = [
            {"korean": "숙제를 냈나요?", "english": "Did you submit your homework?"},
            {"korean": "시험을 준비하세요", "english": "Prepare for the exam"},
            {"korean": "수업에 참여하세요", "english": "Participate in class"},
            {"korean": "질문이 있나요?", "english": "Do you have any questions?"},
            {"korean": "성적이 향상되었어요", "english": "The grades have improved"},
            {"korean": "과제를 설명해 드리겠습니다", "english": "I'll explain the assignment"},
            {"korean": "도서관에서 공부하세요", "english": "Study in the library"},
            {"korean": "논문을 작성해야 합니다", "english": "I need to write a thesis"},
            {"korean": "장학금을 받았습니다", "english": "I received a scholarship"},
            {"korean": "졸업을 축하합니다", "english": "Congratulations on your graduation"}
        ]
        
        travel_data = [
            {"korean": "여행 일정을 계획하세요", "english": "Plan your travel itinerary"},
            {"korean": "호텔을 예약했어요", "english": "I booked a hotel"},
            {"korean": "비행기표를 확인하세요", "english": "Check your flight ticket"},
            {"korean": "여권을 가져오세요", "english": "Bring your passport"},
            {"korean": "관광지를 방문하세요", "english": "Visit the tourist spots"},
            {"korean": "공항에 도착했습니다", "english": "I arrived at the airport"},
            {"korean": "체크인을 하세요", "english": "Check in"},
            {"korean": "보험에 가입하세요", "english": "Get travel insurance"},
            {"korean": "지도를 확인하세요", "english": "Check the map"},
            {"korean": "현지 음식을 맛보세요", "english": "Try the local food"}
        ]
        
        # Combine all data
        all_data = (base_data + business_data + medical_data + 
                   technology_data + education_data + travel_data)
        
        # Data augmentation for better coverage
        augmented_data = self.augment_training_data(all_data)
        
        logger.info(f"Created {len(augmented_data)} training samples")
        return augmented_data
    
    def augment_training_data(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Augment training data with variations"""
        augmented = data.copy()
        
        # Add variations with different politeness levels
        politeness_variations = {
            "안녕하세요": ["안녕하십니까", "안녕", "여보세요"],
            "감사합니다": ["고맙습니다", "고마워요", "감사해요"],
            "미안합니다": ["죄송합니다", "미안해요", "죄송해요"],
            "이해했어요": ["이해했습니다", "알겠어요", "알겠습니다"]
        }
        
        for item in data:
            korean = item['korean']
            english = item['english']
            
            # Add original
            augmented.append({"korean": korean, "english": english})
            
            # Add politeness variations
            if korean in politeness_variations:
                for variant in politeness_variations[korean]:
                    augmented.append({"korean": variant, "english": english})
        
        # Remove duplicates
        unique_data = []
        seen = set()
        for item in augmented:
            key = (item['korean'], item['english'])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        return unique_data
    
    def create_synthetic_multimodal_data(self, text_data: List[Dict[str, str]]) -> Tuple[Dict, Dict]:
        """Create synthetic image and audio data for training"""
        logger.info("Creating synthetic multimodal data...")
        
        image_data = {}
        audio_data = {}
        
        for item in text_data:
            korean_text = item['korean']
            
            # Create synthetic image (simulating Korean text in image)
            # Different characteristics based on text content
            text_length = len(korean_text)
            complexity = len(set(korean_text)) / text_length if text_length > 0 else 0
            
            image_tensor = torch.randn(1, 3, 224, 224)
            # Add some structure based on text characteristics
            image_tensor = image_tensor * (0.5 + 0.5 * complexity)
            image_tensor = torch.clamp(image_tensor, -1, 1)
            
            image_data[korean_text] = image_tensor
            
            # Create synthetic audio (simulating Korean speech)
            # Varying length based on text length (2-5 seconds at 16kHz)
            audio_length = 16000 * (2 + text_length // 8)
            audio_tensor = torch.randn(1, audio_length)
            
            # Add speech-like structure
            t = torch.linspace(0, 1, audio_length)
            # Add formant frequencies typical for Korean speech
            formant1 = 0.3 * torch.sin(2 * np.pi * 500 * t)   # First formant
            formant2 = 0.2 * torch.sin(2 * np.pi * 1500 * t)  # Second formant  
            formant3 = 0.1 * torch.sin(2 * np.pi * 2500 * t)  # Third formant
            
            audio_tensor = audio_tensor * 0.4 + (formant1 + formant2 + formant3).unsqueeze(0)
            audio_tensor = torch.clamp(audio_tensor, -1, 1)
            
            audio_data[korean_text] = audio_tensor
        
        logger.info(f"Created {len(image_data)} synthetic images and {len(audio_data)} synthetic audio samples")
        return image_data, audio_data
    
    def train_epoch(self, dataloader: DataLoader, model_type: str = "multimodal") -> Dict[str, float]:
        """Train for one epoch"""
        if model_type == "text":
            model = self.text_model
            optimizer = self.text_optimizer
            scheduler = self.text_scheduler
        else:
            model = self.multimodal_model
            optimizer = self.multimodal_optimizer
            scheduler = self.multimodal_scheduler
            
        model.train()
        if model_type == "multimodal":
            self.image_encoder.train()
            self.audio_encoder.train()
            self.multimodal_encoder.train()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training {model_type}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move data to device
            korean_tokens = batch['korean_tokens'].to(self.device)
            english_tokens = batch['english_tokens'].to(self.device)
            image_tensors = batch['image_tensor'].to(self.device)
            audio_tensors = batch['audio_tensor'].to(self.device)
            
            if model_type == "text":
                # Text-only training
                output = model(korean_tokens, english_tokens[:, :-1])
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    english_tokens[:, 1:].reshape(-1)
                )
            else:
                # Multimodal training
                with torch.cuda.amp.autocast(enabled=self.config['training']['use_amp']):
                    # Encode multimodal inputs
                    batch_size = korean_tokens.size(0)
                    encoded_features_list = []
                    
                    for i in range(batch_size):
                        # Process each sample individually
                        korean_token = korean_tokens[i:i+1]
                        image_tensor = image_tensors[i:i+1]
                        audio_tensor = audio_tensors[i:i+1]
                        
                        # Encode multimodal input
                        multimodal_output = self.multimodal_encoder(
                            text_input=korean_token,
                            image_input=image_tensor,
                            audio_input=audio_tensor,
                            training=True
                        )
                        
                        encoded_features = multimodal_output['encoded_features']
                        encoded_features_list.append(encoded_features)
                    
                    # Combine encoded features
                    encoded_features = torch.cat(encoded_features_list, dim=0)
                    
                    # Generate translation using multimodal features
                    output = self.multimodal_model.generate_with_features(
                        encoded_features, english_tokens[:, :-1]
                    )
                    
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)),
                        english_tokens[:, 1:].reshape(-1)
                    )
            
            # Backward pass
            if self.config['training']['use_amp']:
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Step scheduler
            if scheduler:
                scheduler.step()
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}
    
    def evaluate(self, dataloader: DataLoader, model_type: str = "multimodal") -> Dict[str, float]:
        """Evaluate model performance"""
        if model_type == "text":
            model = self.text_model
        else:
            model = self.multimodal_model
        
        model.eval()
        if model_type == "multimodal":
            self.image_encoder.eval()
            self.audio_encoder.eval()
            self.multimodal_encoder.eval()
        
        all_predictions = []
        all_references = []
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {model_type}"):
                # Move data to device
                korean_tokens = batch['korean_tokens'].to(self.device)
                english_tokens = batch['english_tokens'].to(self.device)
                image_tensors = batch['image_tensor'].to(self.device)
                audio_tensors = batch['audio_tensor'].to(self.device)
                
                if model_type == "text":
                    # Text-only evaluation
                    output = model(korean_tokens, english_tokens[:, :-1])
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)),
                        english_tokens[:, 1:].reshape(-1)
                    )
                    
                    # Generate predictions
                    predictions = torch.argmax(output, dim=-1)
                    
                else:
                    # Multimodal evaluation
                    batch_size = korean_tokens.size(0)
                    encoded_features_list = []
                    
                    for i in range(batch_size):
                        korean_token = korean_tokens[i:i+1]
                        image_tensor = image_tensors[i:i+1]
                        audio_tensor = audio_tensors[i:i+1]
                        
                        multimodal_output = self.multimodal_encoder(
                            text_input=korean_token,
                            image_input=image_tensor,
                            audio_input=audio_tensor,
                            training=False
                        )
                        
                        encoded_features = multimodal_output['encoded_features']
                        encoded_features_list.append(encoded_features)
                    
                    encoded_features = torch.cat(encoded_features_list, dim=0)
                    
                    output = self.multimodal_model.generate_with_features(
                        encoded_features, english_tokens[:, :-1]
                    )
                    
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)),
                        english_tokens[:, 1:].reshape(-1)
                    )
                    
                    predictions = torch.argmax(output, dim=-1)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Convert to text for BLEU calculation
                for i in range(predictions.size(0)):
                    pred_tokens = predictions[i].cpu().numpy()
                    ref_tokens = english_tokens[i, 1:].cpu().numpy()
                    
                    pred_text = self.tokens_to_text(pred_tokens)
                    ref_text = self.tokens_to_text(ref_tokens)
                    
                    all_predictions.append(pred_text)
                    all_references.append(ref_text)
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        bleu_score = self.bleu_score(all_predictions, all_references)
        exact_match_rate = np.mean([
            self.exact_match(pred, ref) 
            for pred, ref in zip(all_predictions, all_references)
        ])
        
        perfect_translations = sum([
            1 for pred, ref in zip(all_predictions, all_references) 
            if self.exact_match(pred, ref) == 1.0
        ])
        
        perfect_rate = perfect_translations / len(all_predictions)
        
        return {
            "loss": avg_loss,
            "bleu_score": bleu_score,
            "exact_match_rate": exact_match_rate,
            "perfect_translation_rate": perfect_rate,
            "total_samples": len(all_predictions)
        }
    
    def tokens_to_text(self, tokens: np.ndarray) -> str:
        """Convert tokens to text (simplified)"""
        # This is a placeholder - use proper detokenizer
        tokens = tokens[tokens != 0]  # Remove padding
        if len(tokens) == 0:
            return ""
        
        # Simple character mapping for demonstration
        text = ""
        for token in tokens:
            if token < 128:
                text += chr(token % 26 + ord('a'))
            else:
                text += " "
        
        return text.strip()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], model_type: str = "multimodal"):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"{model_type}_checkpoint_epoch_{epoch}_{timestamp}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'timestamp': timestamp,
            'config': self.config,
            'metrics': metrics,
            'model_state_dict': (self.text_model.state_dict() if model_type == "text" 
                               else self.multimodal_model.state_dict()),
            'optimizer_state_dict': (self.text_optimizer.state_dict() if model_type == "text"
                                   else self.multimodal_optimizer.state_dict()),
            'scheduler_state_dict': (self.text_scheduler.state_dict() if model_type == "text"
                                   else self.multimodal_scheduler.state_dict())
        }
        
        if model_type == "multimodal":
            checkpoint['image_encoder_state_dict'] = self.image_encoder.state_dict()
            checkpoint['audio_encoder_state_dict'] = self.audio_encoder.state_dict()
            checkpoint['multimodal_encoder_state_dict'] = self.multimodal_encoder.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save as best model if this is the best so far
        if metrics.get('perfect_translation_rate', 0) > getattr(self, 'best_perfect_rate', 0):
            self.best_perfect_rate = metrics['perfect_translation_rate']
            best_path = checkpoint_dir / f"{model_type}_best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def train(self, train_data: List[Dict[str, str]], val_data: List[Dict[str, str]]):
        """Main training loop"""
        logger.info("Starting multimodal NMT training...")
        
        # Create synthetic multimodal data
        train_image_data, train_audio_data = self.create_synthetic_multimodal_data(train_data)
        val_image_data, val_audio_data = self.create_synthetic_multimodal_data(val_data)
        
        # Create datasets
        train_dataset = MultimodalTranslationDataset(
            train_data, train_image_data, train_audio_data,
            max_length=self.config['data']['max_length']
        )
        
        val_dataset = MultimodalTranslationDataset(
            val_data, val_image_data, val_audio_data,
            max_length=self.config['data']['max_length']
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Training loop
        best_val_metrics = {"perfect_translation_rate": 0.0}
        patience_counter = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            # Train text model first (if specified)
            if self.config['training']['train_text_first'] and epoch < self.config['training']['text_epochs']:
                logger.info("Training text model...")
                train_metrics = self.train_epoch(train_dataloader, "text")
                val_metrics = self.evaluate(val_dataloader, "text")
                
                logger.info(f"Text Model - Train Loss: {train_metrics['loss']:.4f}")
                logger.info(f"Text Model - Val Loss: {val_metrics['loss']:.4f}")
                logger.info(f"Text Model - BLEU: {val_metrics['bleu_score']:.4f}")
                logger.info(f"Text Model - Perfect Rate: {val_metrics['perfect_translation_rate']:.1%}")
                
                self.save_checkpoint(epoch, val_metrics, "text")
            
            # Train multimodal model
            logger.info("Training multimodal model...")
            train_metrics = self.train_epoch(train_dataloader, "multimodal")
            val_metrics = self.evaluate(val_dataloader, "multimodal")
            
            logger.info(f"Multimodal Model - Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Multimodal Model - Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Multimodal Model - BLEU: {val_metrics['bleu_score']:.4f}")
            logger.info(f"Multimodal Model - Perfect Rate: {val_metrics['perfect_translation_rate']:.1%}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, "multimodal")
            
            # Early stopping check
            if val_metrics['perfect_translation_rate'] > best_val_metrics['perfect_translation_rate']:
                best_val_metrics = val_metrics.copy()
                patience_counter = 0
                logger.info(f"New best perfect translation rate: {val_metrics['perfect_translation_rate']:.1%}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} epochs")
            
            # Check if we reached the target
            if val_metrics['perfect_translation_rate'] >= 0.99:
                logger.info("🎉 TARGET ACHIEVED: 99% perfect translation rate reached!")
                break
            
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation metrics: {best_val_metrics}")
        
        return best_val_metrics

def create_default_config() -> Dict[str, Any]:
    """Create default training configuration"""
    return {
        'model': {
            'vocab_size': 32000,
            'd_model': 1024,
            'n_heads': 16,
            'n_encoder_layers': 12,
            'n_decoder_layers': 12,
            'd_ff': 4096,
            'dropout': 0.1,
            'max_len': 512
        },
        'multimodal': {
            'image_encoder': {
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 1024,
                'num_heads': 16,
                'num_layers': 12
            },
            'audio_encoder': {
                'sample_rate': 16000,
                'n_fft': 400,
                'hop_length': 160,
                'n_mels': 80,
                'embed_dim': 1024,
                'num_heads': 16,
                'num_layers': 12
            },
            'fusion': {
                'embed_dim': 1024,
                'num_heads': 16,
                'num_layers': 12
            }
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'label_smoothing': 0.1,
            'scheduler_t0': 10,
            'scheduler_t_mult': 2,
            'early_stopping_patience': 15,
            'use_amp': True,
            'train_text_first': True,
            'text_epochs': 5
        },
        'data': {
            'max_length': 128,
            'train_split': 0.8,
            'val_split': 0.2
        },
        'paths': {
            'checkpoint_dir': 'models/checkpoints',
            'log_dir': 'logs'
        }
    }

def main():
    """Main function to train multimodal NMT model"""
    logger.info("Starting Multimodal Korean-English NMT Training")
    
    # Create configuration
    config = create_default_config()
    
    # Create trainer
    trainer = MultimodalNMTTrainer(config)
    
    # Create training data
    logger.info("Creating training data...")
    train_data = trainer.create_enhanced_training_data()
    
    # Split into train and validation
    split_idx = int(len(train_data) * config['data']['train_split'])
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]
    
    logger.info(f"Train samples: {len(train_split)}")
    logger.info(f"Validation samples: {len(val_split)}")
    
    # Start training
    best_metrics = trainer.train(train_split, val_split)
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Best Perfect Translation Rate: {best_metrics['perfect_translation_rate']:.1%}")
    logger.info(f"Best BLEU Score: {best_metrics['bleu_score']:.4f}")
    logger.info(f"Best Exact Match Rate: {best_metrics['exact_match_rate']:.1%}")
    
    if best_metrics['perfect_translation_rate'] >= 0.99:
        logger.info("✅ SUCCESS: Target 99% perfect translation achieved!")
    else:
        gap = 0.99 - best_metrics['perfect_translation_rate']
        logger.info(f"❌ TARGET NOT REACHED: Gap of {gap:.1%} to 99% target")
        logger.info("Recommendations: Increase training data, adjust hyperparameters, or extend training time")

if __name__ == "__main__":
    main()