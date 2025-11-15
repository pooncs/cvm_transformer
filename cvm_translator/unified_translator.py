import logging
import asyncio
from typing import Dict, Optional, Union, List, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np

from .language_detector import LanguageDetector, DetectionMode, DetectionResult
from .cvm_transformer import CVMTransformer
from .sp_tokenizer import SPTokenizer


@dataclass
class TranslationRequest:
    """Unified translation request with automatic language detection."""
    input_data: Union[str, bytes, np.ndarray]
    input_mode: DetectionMode = DetectionMode.AUTO
    source_language: Optional[str] = None  # Override auto-detection
    target_language: str = "en"
    audio_sample_rate: int = 16000
    enable_fallback: bool = True
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TranslationResponse:
    """Complete translation response with detection and translation results."""
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    detection_result: DetectionResult
    processing_time: float
    fallback_used: bool = False
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UnifiedTranslator:
    """
    Unified translator with automatic language detection and routing.
    Integrates multimodal language detection with CVM-enhanced translation.
    """
    
    def __init__(self, 
                 cvm_model_path: str = "kr_en.model",
                 vocab_path: str = "kr_en.vocab",
                 language_detector_config: Dict = None,
                 device: str = "auto",
                 enable_logging: bool = True):
        """
        Initialize the unified translator.
        
        Args:
            cvm_model_path: Path to CVM model
            vocab_path: Path to vocabulary file
            language_detector_config: Configuration for language detector
            device: Device for inference
            enable_logging: Enable detailed logging
        """
        self.logger = logging.getLogger(__name__)
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            
        # Initialize language detector
        self.language_detector = LanguageDetector(**(language_detector_config or {}))
        
        # Initialize CVM translator components
        self.tokenizer = SPTokenizer(vocab_path)
        self.cvm_model = CVMTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            max_seq_length=512
        )
        
        # Load CVM model if available
        if Path(cvm_model_path).exists():
            try:
                checkpoint = torch.load(cvm_model_path, map_location='cpu')
                self.cvm_model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded CVM model from {cvm_model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load CVM model: {e}")
                self.logger.info("Using uninitialized CVM model")
        
        self.cvm_model.eval()
        self.device = self.language_detector.device
        self.cvm_model.to(self.device)
        
        # Language pair configurations
        self.language_pairs = {
            ('ko', 'en'): 'korean_to_english',
            ('en', 'ko'): 'english_to_korean',
            ('ja', 'en'): 'japanese_to_english',
            ('en', 'ja'): 'english_to_japanese',
            ('zh', 'en'): 'chinese_to_english',
            ('en', 'zh'): 'english_to_chinese'
        }
        
        # Fallback configurations
        self.fallback_configs = {
            'enable_google_translate': False,  # Set to True to enable Google Translate fallback
            'enable_papago': False,  # Set to True to enable Papago fallback
            'confidence_threshold': 0.5
        }
        
        self.logger.info("Unified translator initialized successfully")
    
    async def translate_async(self, request: TranslationRequest) -> TranslationResponse:
        """
        Asynchronous translation with automatic language detection.
        
        Args:
            request: Translation request
            
        Returns:
            Translation response
        """
        start_time = time.time()
        
        try:
            # Step 1: Language detection if not provided
            if request.source_language is None:
                detection_result = await self._detect_language_async(request)
                if detection_result is None:
                    raise ValueError("Language detection failed")
                source_language = detection_result.language
            else:
                source_language = request.source_language
                detection_result = DetectionResult(
                    language=source_language,
                    confidence=1.0,
                    mode=request.input_mode,
                    metadata={'manual_override': True}
                )
            
            # Step 2: Route to appropriate translation pipeline
            translated_text = await self._translate_with_routing(
                request.input_data, 
                source_language, 
                request.target_language,
                request.input_mode
            )
            
            processing_time = time.time() - start_time
            
            return TranslationResponse(
                translated_text=translated_text,
                source_language=source_language,
                target_language=request.target_language,
                confidence=detection_result.confidence,
                detection_result=detection_result,
                processing_time=processing_time,
                fallback_used=False,
                metadata={
                    'detection_model': detection_result.metadata.get('model', 'unknown'),
                    'translation_model': 'cvm_enhanced',
                    'device': self.device
                }
            )
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            
            # Try fallback if enabled
            if request.enable_fallback:
                return await self._handle_fallback(request, e, time.time() - start_time)
            else:
                raise
    
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Synchronous translation with automatic language detection.
        
        Args:
            request: Translation request
            
        Returns:
            Translation response
        """
        # Run async version in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.translate_async(request))
    
    async def _detect_language_async(self, request: TranslationRequest) -> Optional[DetectionResult]:
        """
        Asynchronous language detection.
        
        Args:
            request: Translation request
            
        Returns:
            Detection result
        """
        # Run detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def detect_sync():
            return self.language_detector.detect(
                request.input_data,
                mode=request.input_mode,
                sample_rate=request.audio_sample_rate
            )
        
        return await loop.run_in_executor(None, detect_sync)
    
    async def _translate_with_routing(self, 
                                    input_data: Union[str, bytes, np.ndarray],
                                    source_language: str,
                                    target_language: str,
                                    input_mode: DetectionMode) -> str:
        """
        Route translation request to appropriate pipeline based on language pair.
        
        Args:
            input_data: Input data
            source_language: Source language code
            target_language: Target language code
            input_mode: Input mode
            
        Returns:
            Translated text
        """
        # Convert input to text if needed
        if input_mode != DetectionMode.TEXT:
            input_text = await self._extract_text_from_input(input_data, input_mode)
        else:
            input_text = input_data
        
        # Check if we have a direct translation model for this language pair
        language_pair = (source_language, target_language)
        
        if language_pair in self.language_pairs:
            # Use CVM-enhanced translation
            return await self._translate_cvm(input_text, source_language, target_language)
        else:
            # Use pivot translation (e.g., ko->en->ja)
            return await self._translate_pivot(input_text, source_language, target_language)
    
    async def _extract_text_from_input(self, 
                                     input_data: Union[str, bytes, np.ndarray],
                                     input_mode: DetectionMode) -> str:
        """
        Extract text from non-text input (audio or image).
        
        Args:
            input_data: Input data
            input_mode: Input mode
            
        Returns:
            Extracted text
        """
        if input_mode == DetectionMode.AUDIO:
            # Use Whisper for speech-to-text
            return await self._extract_text_from_audio(input_data)
        elif input_mode == DetectionMode.IMAGE:
            # Use OCR for text extraction
            return await self._extract_text_from_image(input_data)
        else:
            raise ValueError(f"Unsupported input mode: {input_mode}")
    
    async def _extract_text_from_audio(self, audio_data: Union[str, bytes, np.ndarray]) -> str:
        """
        Extract text from audio using Whisper.
        
        Args:
            audio_data: Audio data
            
        Returns:
            Transcribed text
        """
        # Import Whisper interface
        from .whisper_interface import WhisperInterface
        
        whisper = WhisperInterface()
        
        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        
        def transcribe_sync():
            return whisper.transcribe(audio_data)
        
        return await loop.run_in_executor(None, transcribe_sync)
    
    async def _extract_text_from_image(self, image_data: Union[str, bytes, np.ndarray]) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image_data: Image data
            
        Returns:
            Extracted text
        """
        # Use the language detector's OCR capabilities
        detection_result = self.language_detector.detect_image_language(image_data)
        
        if detection_result and 'text_extracted' in detection_result.metadata:
            return detection_result.metadata['text_extracted']
        else:
            raise ValueError("Failed to extract text from image")
    
    async def _translate_cvm(self, text: str, source_language: str, target_language: str) -> str:
        """
        Translate using CVM-enhanced model.
        
        Args:
            text: Input text
            source_language: Source language
            target_language: Target language
            
        Returns:
            Translated text
        """
        # Tokenize input
        tokens = self.tokenizer.encode(text)
        
        # Convert to tensor
        input_tensor = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
        
        # Run translation
        with torch.no_grad():
            output = self.cvm_model(input_tensor, input_tensor)  # Simplified for demo
            
        # Decode output
        output_tokens = output.argmax(dim=-1).squeeze().cpu().numpy()
        translated_text = self.tokenizer.decode(output_tokens)
        
        return translated_text
    
    async def _translate_pivot(self, text: str, source_language: str, target_language: str) -> str:
        """
        Translate using pivot language (usually English).
        
        Args:
            text: Input text
            source_language: Source language
            target_language: Target language
            
        Returns:
            Translated text
        """
        # First translate to pivot language (English)
        if source_language != 'en':
            pivot_text = await self._translate_cvm(text, source_language, 'en')
        else:
            pivot_text = text
        
        # Then translate from pivot to target
        if target_language != 'en':
            return await self._translate_cvm(pivot_text, 'en', target_language)
        else:
            return pivot_text
    
    async def _handle_fallback(self, 
                             request: TranslationRequest,
                             error: Exception,
                             processing_time: float) -> TranslationResponse:
        """
        Handle translation failures with fallback mechanisms.
        
        Args:
            request: Original translation request
            error: Exception that caused the failure
            processing_time: Time spent before failure
            
        Returns:
            Fallback translation response
        """
        self.logger.warning(f"Using fallback translation due to: {error}")
        
        # Simple fallback: return original text with error indication
        fallback_text = f"[Translation Error: {str(error)}] {request.input_data}"
        
        # Try to detect language even if translation failed
        detection_result = None
        if request.source_language is None:
            try:
                detection_result = await self._detect_language_async(request)
                source_language = detection_result.language if detection_result else "unknown"
            except:
                source_language = "unknown"
        else:
            source_language = request.source_language
        
        return TranslationResponse(
            translated_text=fallback_text,
            source_language=source_language,
            target_language=request.target_language,
            confidence=0.0,
            detection_result=detection_result,
            processing_time=processing_time,
            fallback_used=True,
            metadata={'error': str(error), 'fallback': True}
        )
    
    def get_supported_language_pairs(self) -> List[tuple]:
        """Get list of supported language pairs."""
        return list(self.language_pairs.keys())
    
    def get_detection_capabilities(self) -> Dict[str, Any]:
        """Get language detection capabilities."""
        return {
            'available_modes': [mode.value for mode in self.language_detector.get_available_modes()],
            'text_languages': self.language_detector.get_supported_languages(DetectionMode.TEXT),
            'audio_languages': self.language_detector.get_supported_languages(DetectionMode.AUDIO),
            'image_languages': self.language_detector.get_supported_languages(DetectionMode.IMAGE)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components."""
        health_status = {
            'language_detector': self.language_detector.get_available_modes() != [],
            'cvm_model': hasattr(self, 'cvm_model'),
            'tokenizer': hasattr(self, 'tokenizer'),
            'device': self.device,
            'supported_pairs': len(self.language_pairs)
        }
        
        health_status['overall'] = all(health_status.values())
        return health_status


# Convenience functions for direct usage
def translate_text(text: str, 
                  target_language: str = "en",
                  source_language: str = None,
                  config: Dict = None) -> TranslationResponse:
    """
    Convenience function for text translation.
    
    Args:
        text: Text to translate
        target_language: Target language code
        source_language: Source language code (auto-detect if None)
        config: Configuration dictionary
        
    Returns:
        Translation response
    """
    translator = UnifiedTranslator(**(config or {}))
    
    request = TranslationRequest(
        input_data=text,
        input_mode=DetectionMode.TEXT,
        source_language=source_language,
        target_language=target_language
    )
    
    return translator.translate(request)


def translate_audio(audio_data: Union[str, bytes, np.ndarray],
                   target_language: str = "en",
                   sample_rate: int = 16000,
                   config: Dict = None) -> TranslationResponse:
    """
    Convenience function for audio translation.
    
    Args:
        audio_data: Audio data (file path, bytes, or numpy array)
        target_language: Target language code
        sample_rate: Audio sample rate
        config: Configuration dictionary
        
    Returns:
        Translation response
    """
    translator = UnifiedTranslator(**(config or {}))
    
    request = TranslationRequest(
        input_data=audio_data,
        input_mode=DetectionMode.AUDIO,
        target_language=target_language,
        audio_sample_rate=sample_rate
    )
    
    return translator.translate(request)


def translate_image(image_data: Union[str, bytes, np.ndarray],
                   target_language: str = "en",
                   config: Dict = None) -> TranslationResponse:
    """
    Convenience function for image translation (OCR + translation).
    
    Args:
        image_data: Image data (file path, bytes, or numpy array)
        target_language: Target language code
        config: Configuration dictionary
        
    Returns:
        Translation response
    """
    translator = UnifiedTranslator(**(config or {}))
    
    request = TranslationRequest(
        input_data=image_data,
        input_mode=DetectionMode.IMAGE,
        target_language=target_language
    )
    
    return translator.translate(request)


# Example usage
if __name__ == "__main__":
    # Initialize translator
    translator = UnifiedTranslator()
    
    # Test text translation
    print("Testing text translation:")
    response = translate_text("안녕하세요, 어떻게 지내세요?", target_language="en")
    print(f"Korean → English: {response.translated_text}")
    print(f"Detected language: {response.source_language} (confidence: {response.confidence:.3f})")
    print(f"Processing time: {response.processing_time:.3f}s")
    
    # Test with different inputs
    test_cases = [
        ("Hello, world!", "ko"),
        ("こんにちは世界", "en"),
        ("你好世界", "en")
    ]
    
    print("\nTesting multiple language pairs:")
    for text, target_lang in test_cases:
        response = translate_text(text, target_language=target_lang)
        print(f"'{text}' → {target_lang}: {response.translated_text}")
    
    # Show capabilities
    print(f"\nDetection capabilities: {translator.get_detection_capabilities()}")
    print(f"Supported language pairs: {translator.get_supported_language_pairs()}")
    print(f"Health check: {translator.health_check()}")