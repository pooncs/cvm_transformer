"""
Multimodal Translation Processor
Handles image-to-text and audio-to-text translation for Korean-English translation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from PIL import Image
import numpy as np
import os

try:
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False
    logging.warning("torchaudio not available, audio processing will be limited")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.warning("opencv-python not available, some image processing will be limited")

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False
    logging.warning("pytesseract not available, OCR will be limited")

# Import our multimodal models
from src.models.multimodal_encoders import ImageEncoder, AudioEncoder
from src.models.multimodal_fusion import GatedMultimodalFusion
from src.models.sp_tokenizer import SPTokenizer

# Import translation model
try:
    from src.models.translation_model import EnhancedTranslationModel
    HAS_TRANSLATION_MODEL = True
except ImportError:
    HAS_TRANSLATION_MODEL = False
    logging.warning("Translation model not available, will use placeholder translation")


class MultimodalTranslationProcessor:
    """Handles multimodal input processing for translation."""
    
    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # Initialize translation model
        self.translation_model = None
        self.has_translation_model = False
        
        if HAS_TRANSLATION_MODEL and model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                config = checkpoint.get('config', {})
                
                self.translation_model = EnhancedTranslationModel(
                    vocab_size=config.get('vocab_size', 32000),
                    d_model=config.get('d_model', 1024),
                    n_heads=config.get('nhead', 16),
                    n_layers=config.get('n_layers_student', 8),
                    ff_dim=config.get('dim_feedforward', 4096),
                    max_len=config.get('max_len', 128),
                    pad_id=0
                ).to(self.device)
                
                self.translation_model.load_state_dict(checkpoint['model_state_dict'])
                self.translation_model.eval()
                self.has_translation_model = True
                logging.info(f"Translation model loaded from {model_path}")
                
            except Exception as e:
                logging.error(f"Failed to load translation model: {e}")
                self.has_translation_model = False
        else:
            logging.warning("No translation model available, will use placeholder translation")
        # Initialize tokenizer if available, otherwise use simple text processing
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = SPTokenizer(tokenizer_path)
            self.has_tokenizer = True
        else:
            logging.warning(f"Tokenizer not found at {tokenizer_path}, using simple text processing")
            self.has_tokenizer = False
            self.tokenizer = None
        
        # Initialize multimodal encoders
        self.image_encoder = ImageEncoder(d_model=768).to(device)
        self.audio_encoder = AudioEncoder(d_model=768).to(device)
        self.fusion_module = GatedMultimodalFusion(embed_dim=768, num_modalities=3).to(device)
        
        logging.info(f"Multimodal processor initialized on {device}")

    def _simple_translate(self, text: str) -> str:
        mapping = {
            "안녕하세요": "Hello",
            "감사합니다": "Thank you",
            "미안합니다": "Sorry",
            "네": "Yes",
            "아니요": "No",
            "물": "Water",
            "밥": "Rice",
            "학교": "School",
            "집": "House",
            "차": "Car",
            "책": "Book",
            "컴퓨터": "Computer",
            "전화": "Phone",
            "시간": "Time",
            "날씨": "Weather",
            "돈": "Money",
            "사랑": "Love",
            "친구": "Friend",
            "가족": "Family",
            "일": "Work",
            "도와주세요": "Please help me",
            "어디 가세요?": "Where are you going?",
            "얼마예요?": "How much is it?",
            "몇 시예요?": "What time is it?",
            "오랜만이에요": "Long time no see",
        }
        for k, v in mapping.items():
            if k in text:
                return text.replace(k, v)
        return text
    
    def process_image(self, image_path: Union[str, Path]) -> Optional[str]:
        """
        Extract Korean text from image using OCR and computer vision.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted Korean text or None if extraction fails
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Use OCR to extract text
            if HAS_PYTESSERACT:
                # Configure for Korean text
                custom_config = r'--oem 3 --psm 6 -l kor'
                text = pytesseract.image_to_string(image, config=custom_config)
                
                if text.strip():
                    logging.info(f"OCR extracted text: {text[:100]}...")
                    return text.strip()
            
            # Fallback: Use vision encoder for text detection
            if HAS_CV2:
                # Convert to numpy array for CV2 processing
                img_array = np.array(image)
                
                # Preprocess for better text detection
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Apply threshold to get better contrast
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Try OCR again on preprocessed image
                if HAS_PYTESSERACT:
                    preprocessed_image = Image.fromarray(thresh)
                    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
                    
                    if text.strip():
                        logging.info(f"OCR on preprocessed image extracted text: {text[:100]}...")
                        return text.strip()
            
            logging.warning("No text extracted from image")
            return None
            
        except Exception as e:
            logging.error(f"Image processing failed: {e}")
            return None
    
    def process_audio(self, audio_path: Union[str, Path]) -> Optional[str]:
        """
        Convert Korean speech to text using audio processing.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed Korean text or None if transcription fails
        """
        if not HAS_TORCHAUDIO:
            logging.error("torchaudio not available for audio processing")
            return None
        
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = T.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Normalize audio
            waveform = waveform / torch.max(torch.abs(waveform))
            
            # Use audio encoder to extract features
            audio_features = self.audio_encoder(waveform.to(self.device))
            
            # For now, return placeholder text
            # In a full implementation, this would connect to a speech recognition model
            logging.info(f"Audio processed successfully, features shape: {audio_features.shape}")
            
            # Placeholder: In production, connect to Korean ASR model
            placeholder_text = "오디오 파일이 성공적으로 처리되었습니다"  # "Audio file processed successfully"
            return placeholder_text
            
        except Exception as e:
            logging.error(f"Audio processing failed: {e}")
            return None
    
    def multimodal_translate(
        self,
        text_input: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        audio_path: Optional[Union[str, Path]] = None,
        target_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Perform multimodal translation combining text, image, and audio inputs.
        
        Args:
            text_input: Optional text input
            image_path: Optional image file path
            audio_path: Optional audio file path
            target_language: Target language code ("en" for English)
            
        Returns:
            Dictionary containing translation results and metadata
        """
        results = {
            "input_modalities": [],
            "extracted_text": {},
            "translation": None,
            "confidence": 0.0,
            "status": "started"
        }
        
        try:
            # Process each modality
            if text_input:
                results["input_modalities"].append("text")
                results["extracted_text"]["text"] = text_input
            
            if image_path:
                results["input_modalities"].append("image")
                extracted_text = self.process_image(image_path)
                if extracted_text:
                    results["extracted_text"]["image"] = extracted_text
                else:
                    logging.warning("Failed to extract text from image")
            
            if audio_path:
                results["input_modalities"].append("audio")
                extracted_text = self.process_audio(audio_path)
                if extracted_text:
                    results["extracted_text"]["audio"] = extracted_text
                else:
                    logging.warning("Failed to transcribe audio")
            
            # Combine extracted texts
            if not results["extracted_text"]:
                results["status"] = "failed"
                results["error"] = "No text extracted from any modality"
                return results
            
            # Use text from the most confident source
            # In a full implementation, this would use the fusion module
            if "text" in results["extracted_text"]:
                combined_text = results["extracted_text"]["text"]
                results["confidence"] = 1.0
            elif "image" in results["extracted_text"]:
                combined_text = results["extracted_text"]["image"]
                results["confidence"] = 0.8
            elif "audio" in results["extracted_text"]:
                combined_text = results["extracted_text"]["audio"]
                results["confidence"] = 0.7
            else:
                results["status"] = "failed"
                results["error"] = "No valid text found"
                return results
            
            # For now, return the Korean text as placeholder
            # In production, this would connect to the NMT model
            # Translate Korean text to English
            if self.has_translation_model and self.translation_model and self.has_tokenizer and self.tokenizer:
                try:
                    # Tokenize input
                    src_tokens = self.tokenizer.encode(combined_text)
                    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
                    
                    # Generate translation
                    with torch.no_grad():
                        logits = self.translation_model(src_tensor)
                        predicted_tokens = logits.squeeze(0).tolist()
                        
                        # Remove BOS and EOS tokens
                        if predicted_tokens and predicted_tokens[0] == 2:  # Remove BOS
                            predicted_tokens = predicted_tokens[1:]
                        
                        # Remove everything after EOS token
                        if 3 in predicted_tokens:  # EOS token
                            eos_idx = predicted_tokens.index(3)
                            predicted_tokens = predicted_tokens[:eos_idx]
                        
                        # Decode output
                        english_text = self.tokenizer.decode(predicted_tokens)
                    
                    results["translation"] = english_text
                    logging.info(f"Translation completed: {combined_text[:30]}... -> {english_text[:30]}...")
                    
                except Exception as e:
                    logging.error(f"Translation failed: {e}")
                    # Fallback to placeholder
                    results["translation"] = self._simple_translate(combined_text)
                    results["status"] = "partial_success"
            else:
                # Fallback to placeholder translation
                results["translation"] = self._simple_translate(combined_text)
                results["status"] = "fallback"

            
            logging.info(f"Multimodal translation completed: {combined_text[:50]}...")
            return results
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logging.error(f"Multimodal translation failed: {e}")
            return results


def main():
    """Example usage of the multimodal processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal Translation Processor")
    parser.add_argument("--text", type=str, help="Text input for translation")
    parser.add_argument("--image", type=str, help="Image file path")
    parser.add_argument("--audio", type=str, help="Audio file path")
    parser.add_argument("--target-lang", type=str, default="en", help="Target language")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = MultimodalTranslationProcessor(device=args.device)
    
    # Perform multimodal translation
    results = processor.multimodal_translate(
        text_input=args.text,
        image_path=args.image,
        audio_path=args.audio,
        target_language=args.target_lang
    )
    
    # Print results
    print("\n=== Multimodal Translation Results ===")
    print(f"Status: {results['status']}")
    print(f"Input Modalities: {results['input_modalities']}")
    print(f"Extracted Text: {results['extracted_text']}")
    print(f"Translation: {results['translation']}")
    print(f"Confidence: {results['confidence']:.2f}")
    
    if "error" in results:
        print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()
