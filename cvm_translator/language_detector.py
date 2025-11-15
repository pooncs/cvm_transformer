import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import tempfile
import wave
import io
from dataclasses import dataclass
from enum import Enum

# Text language detection
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False

# Audio language detection
try:
    import speechbrain as sb
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Image language detection (OCR-based)
try:
    import cv2
    import pytesseract
    CV2_AVAILABLE = True
    TESSERACT_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    TESSERACT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# PaddleOCR for better multilingual OCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


class DetectionMode(Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    AUTO = "auto"


@dataclass
class DetectionResult:
    language: str
    confidence: float
    mode: DetectionMode
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LanguageDetector:
    """
    Multimodal language detection supporting text, audio, and image inputs.
    Optimized for edge deployment with lightweight models.
    """
    
    def __init__(self, 
                 models_dir: str = "models/language_detection",
                 device: str = "auto",
                 confidence_threshold: float = 0.7):
        """
        Initialize the multimodal language detector.
        
        Args:
            models_dir: Directory to store detection models
            device: Device for inference ('auto', 'cpu', 'cuda')
            confidence_threshold: Minimum confidence for language detection
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._init_text_models()
        self._init_audio_models()
        self._init_image_models()
        
        # Language mapping for common languages
        self.language_names = {
            'ko': 'Korean', 'en': 'English', 'ja': 'Japanese', 'zh': 'Chinese',
            'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ru': 'Russian',
            'ar': 'Arabic', 'hi': 'Hindi', 'pt': 'Portuguese', 'it': 'Italian'
        }
        
    def _init_text_models(self):
        """Initialize text-based language detection models."""
        self.text_models = {}
        
        # FastText model (lightweight, 176 languages)
        if FASTTEXT_AVAILABLE:
            try:
                fasttext_model_path = self.models_dir / "lid.176.bin"
                if not fasttext_model_path.exists():
                    self.logger.info("Downloading FastText language detection model...")
                    fasttext.util.download_model('176', str(fasttext_model_path))
                
                self.text_models['fasttext'] = fasttext.load_model(str(fasttext_model_path))
                self.logger.info("FastText model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load FastText model: {e}")
        
        # langid.py as fallback
        if LANGID_AVAILABLE:
            try:
                langid.set_languages(['ko', 'en', 'ja', 'zh', 'es', 'fr', 'de'])
                self.text_models['langid'] = langid
                self.logger.info("langid model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load langid model: {e}")
    
    def _init_audio_models(self):
        """Initialize audio-based language detection models."""
        self.audio_models = {}
        
        # SpeechBrain ECAPA-TDNN for language identification
        if SPEECHBRAIN_AVAILABLE and LIBROSA_AVAILABLE:
            try:
                # Load VoxLingua107 ECAPA-TDNN model
                self.audio_models['ecapa_tdnn'] = EncoderClassifier.from_hparams(
                    source="speechbrain/lang-id-voxlingua107-ecapa",
                    savedir=self.models_dir / "speechbrain_ecapa"
                )
                self.logger.info("SpeechBrain ECAPA-TDNN model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load SpeechBrain model: {e}")
    
    def _init_image_models(self):
        """Initialize image-based language detection models (OCR)."""
        self.image_models = {}
        
        # PaddleOCR for multilingual text detection
        if PADDLEOCR_AVAILABLE:
            try:
                # Initialize PaddleOCR with multilingual support
                self.image_models['paddleocr'] = PaddleOCR(
                    lang='multilingual',
                    use_angle_cls=True,
                    show_log=False,
                    det_model_dir=str(self.models_dir / "paddleocr_det"),
                    rec_model_dir=str(self.models_dir / "paddleocr_rec")
                )
                self.logger.info("PaddleOCR model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load PaddleOCR model: {e}")
        
        # Tesseract as fallback
        if TESSERACT_AVAILABLE:
            try:
                # Test if tesseract is available
                pytesseract.get_tesseract_version()
                self.image_models['tesseract'] = pytesseract
                self.logger.info("Tesseract OCR loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load Tesseract: {e}")
    
    def detect_text_language(self, text: str) -> Optional[DetectionResult]:
        """
        Detect language from text input.
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectionResult with language and confidence
        """
        if not text or not text.strip():
            return None
            
        # Try FastText first (most accurate)
        if 'fasttext' in self.text_models:
            try:
                predictions = self.text_models['fasttext'].predict(text, k=1)
                lang_code = predictions[0][0].replace('__label__', '')
                confidence = float(predictions[1][0])
                
                if confidence >= self.confidence_threshold:
                    return DetectionResult(
                        language=lang_code,
                        confidence=confidence,
                        mode=DetectionMode.TEXT,
                        metadata={'model': 'fasttext', 'text_length': len(text)}
                    )
            except Exception as e:
                self.logger.warning(f"FastText detection failed: {e}")
        
        # Fallback to langid
        if 'langid' in self.text_models:
            try:
                lang_code, confidence = self.text_models['langid'].classify(text)
                confidence = float(confidence)
                
                if confidence >= self.confidence_threshold:
                    return DetectionResult(
                        language=lang_code,
                        confidence=confidence,
                        mode=DetectionMode.TEXT,
                        metadata={'model': 'langid', 'text_length': len(text)}
                    )
            except Exception as e:
                self.logger.warning(f"langid detection failed: {e}")
                
        return None
    
    def detect_audio_language(self, audio_data: Union[str, bytes, np.ndarray], 
                            sample_rate: int = 16000) -> Optional[DetectionResult]:
        """
        Detect language from audio input.
        
        Args:
            audio_data: Audio file path, bytes, or numpy array
            sample_rate: Sample rate for audio data
            
        Returns:
            DetectionResult with language and confidence
        """
        if not LIBROSA_AVAILABLE:
            self.logger.warning("Librosa not available for audio processing")
            return None
            
        try:
            # Load audio data
            if isinstance(audio_data, str):
                # File path
                audio, sr = librosa.load(audio_data, sr=sample_rate)
            elif isinstance(audio_data, bytes):
                # Raw audio bytes
                audio, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate)
            elif isinstance(audio_data, np.ndarray):
                # Numpy array
                audio = audio_data
                sr = sample_rate
            else:
                self.logger.error("Unsupported audio data type")
                return None
                
            # Ensure minimum audio length (1 second)
            if len(audio) < sr:
                self.logger.warning("Audio too short for reliable detection")
                return None
            
            # Use SpeechBrain ECAPA-TDNN
            if 'ecapa_tdnn' in self.audio_models:
                # Prepare audio for model
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
                
                # Get prediction
                prediction = self.audio_models['ecapa_tdnn'].classify_batch(audio_tensor)
                lang_scores = prediction[0].squeeze()
                
                # Get top prediction
                top_confidence, top_idx = torch.max(lang_scores, dim=0)
                confidence = float(top_confidence)
                
                # Map index to language code (this is model-specific)
                # For VoxLingua107, we need to map the indices
                lang_code = self._map_voxlingua107_language(top_idx.item())
                
                if confidence >= self.confidence_threshold:
                    return DetectionResult(
                        language=lang_code,
                        confidence=confidence,
                        mode=DetectionMode.AUDIO,
                        metadata={
                            'model': 'ecapa_tdnn',
                            'duration': len(audio) / sr,
                            'sample_rate': sr
                        }
                    )
                    
        except Exception as e:
            self.logger.error(f"Audio language detection failed: {e}")
            
        return None
    
    def detect_image_language(self, image_data: Union[str, bytes, np.ndarray]) -> Optional[DetectionResult]:
        """
        Detect language from image input (OCR-based).
        
        Args:
            image_data: Image file path, bytes, or numpy array
            
        Returns:
            DetectionResult with language and confidence
        """
        try:
            # Load image
            if isinstance(image_data, str):
                image = cv2.imread(image_data) if CV2_AVAILABLE else Image.open(image_data)
            elif isinstance(image_data, bytes):
                if PIL_AVAILABLE:
                    image = Image.open(io.BytesIO(image_data))
                else:
                    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            elif isinstance(image_data, np.ndarray):
                image = image_data
            else:
                self.logger.error("Unsupported image data type")
                return None
            
            # Try PaddleOCR first
            if 'paddleocr' in self.image_models:
                try:
                    results = self.image_models['paddleocr'].ocr(image, cls=True)
                    
                    if results and results[0]:
                        # Extract text from OCR results
                        ocr_text = " ".join([line[1][0] for line in results[0]])
                        
                        if ocr_text.strip():
                            # Detect language of extracted text
                            text_result = self.detect_text_language(ocr_text)
                            if text_result:
                                text_result.mode = DetectionMode.IMAGE
                                text_result.metadata.update({
                                    'model': 'paddleocr+text',
                                    'ocr_confidence': np.mean([line[1][1] for line in results[0]]),
                                    'text_extracted': ocr_text[:100]  # First 100 chars
                                })
                                return text_result
                                
                except Exception as e:
                    self.logger.warning(f"PaddleOCR failed: {e}")
            
            # Fallback to Tesseract
            if 'tesseract' in self.image_models and PIL_AVAILABLE:
                try:
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    ocr_text = self.image_models['tesseract'].image_to_string(image)
                    
                    if ocr_text.strip():
                        text_result = self.detect_text_language(ocr_text)
                        if text_result:
                            text_result.mode = DetectionMode.IMAGE
                            text_result.metadata.update({
                                'model': 'tesseract+text',
                                'text_extracted': ocr_text[:100]
                            })
                            return text_result
                            
                except Exception as e:
                    self.logger.warning(f"Tesseract failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Image language detection failed: {e}")
            
        return None
    
    def detect(self, 
               input_data: Union[str, bytes, np.ndarray],
               mode: DetectionMode = DetectionMode.AUTO,
               **kwargs) -> Optional[DetectionResult]:
        """
        Unified language detection interface.
        
        Args:
            input_data: Input data (text string, audio bytes, or image)
            mode: Detection mode (auto-detect input type if AUTO)
            **kwargs: Additional parameters (sample_rate for audio, etc.)
            
        Returns:
            DetectionResult with language and confidence
        """
        if mode == DetectionMode.AUTO:
            # Auto-detect input type
            if isinstance(input_data, str):
                # Could be text or file path
                if len(input_data) < 1000 and Path(input_data).exists():
                    # Likely a file path
                    file_suffix = Path(input_data).suffix.lower()
                    if file_suffix in ['.wav', '.mp3', '.flac', '.m4a']:
                        mode = DetectionMode.AUDIO
                    elif file_suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        mode = DetectionMode.IMAGE
                    else:
                        mode = DetectionMode.TEXT
                else:
                    # Likely text content
                    mode = DetectionMode.TEXT
            elif isinstance(input_data, bytes):
                # Could be audio or image - assume audio for now
                mode = DetectionMode.AUDIO
            elif isinstance(input_data, np.ndarray):
                # Could be audio or image - need context
                mode = kwargs.get('force_mode', DetectionMode.AUDIO)
        
        # Route to appropriate detector
        if mode == DetectionMode.TEXT:
            return self.detect_text_language(input_data)
        elif mode == DetectionMode.AUDIO:
            return self.detect_audio_language(input_data, **kwargs)
        elif mode == DetectionMode.IMAGE:
            return self.detect_image_language(input_data)
        else:
            self.logger.error(f"Unsupported detection mode: {mode}")
            return None
    
    def _map_voxlingua107_language(self, idx: int) -> str:
        """
        Map VoxLingua107 model indices to language codes.
        This is a simplified mapping - full mapping would require model metadata.
        """
        # Common language mappings for VoxLingua107
        common_langs = {
            0: 'en', 1: 'de', 2: 'es', 3: 'fr', 4: 'it', 5: 'ja', 
            6: 'ko', 7: 'pt', 8: 'ru', 9: 'zh', 10: 'ar', 11: 'hi'
        }
        return common_langs.get(idx, 'unknown')
    
    def get_available_modes(self) -> List[DetectionMode]:
        """Get list of available detection modes based on loaded models."""
        available = []
        
        if self.text_models:
            available.append(DetectionMode.TEXT)
        if self.audio_models:
            available.append(DetectionMode.AUDIO)
        if self.image_models:
            available.append(DetectionMode.IMAGE)
            
        return available
    
    def get_supported_languages(self, mode: DetectionMode) -> List[str]:
        """Get list of supported languages for a specific mode."""
        if mode == DetectionMode.TEXT:
            if 'fasttext' in self.text_models:
                return list(self.language_names.keys())
            elif 'langid' in self.text_models:
                return ['ko', 'en', 'ja', 'zh', 'es', 'fr', 'de']
        elif mode == DetectionMode.AUDIO:
            # VoxLingua107 supports 107 languages
            return ['en', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'pt', 'ru', 'zh', 'ar', 'hi']
        elif mode == DetectionMode.IMAGE:
            # OCR-based, depends on text detection
            return list(self.language_names.keys())
            
        return []
    
    def batch_detect(self, 
                      inputs: List[Union[str, bytes, np.ndarray]],
                      modes: List[DetectionMode] = None,
                      **kwargs) -> List[Optional[DetectionResult]]:
        """
        Batch processing for multiple inputs.
        
        Args:
            inputs: List of input data
            modes: List of detection modes (one per input)
            **kwargs: Additional parameters
            
        Returns:
            List of DetectionResults
        """
        if modes is None:
            modes = [DetectionMode.AUTO] * len(inputs)
        
        results = []
        for input_data, mode in zip(inputs, modes):
            try:
                result = self.detect(input_data, mode, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch detection failed for input: {e}")
                results.append(None)
                
        return results


def create_language_detector(config: Dict = None) -> LanguageDetector:
    """
    Factory function to create a LanguageDetector instance with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured LanguageDetector instance
    """
    if config is None:
        config = {}
    
    return LanguageDetector(
        models_dir=config.get('models_dir', 'models/language_detection'),
        device=config.get('device', 'auto'),
        confidence_threshold=config.get('confidence_threshold', 0.7)
    )


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = LanguageDetector()
    
    # Test text detection
    test_texts = [
        "Hello, how are you?",
        "안녕하세요, 어떻게 지내세요?",
        "こんにちは、お元気ですか？",
        "你好，你好吗？"
    ]
    
    print("Text Language Detection:")
    for text in test_texts:
        result = detector.detect_text_language(text)
        if result:
            print(f"Text: '{text}' -> Language: {result.language} "
                  f"({detector.language_names.get(result.language, 'Unknown')}) "
                  f"(confidence: {result.confidence:.3f})")
        else:
            print(f"Text: '{text}' -> Detection failed")
    
    print(f"\nAvailable detection modes: {[mode.value for mode in detector.get_available_modes()]}")
    print(f"Supported languages (text): {detector.get_supported_languages(DetectionMode.TEXT)}")