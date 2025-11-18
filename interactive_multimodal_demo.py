#!/usr/bin/env python3
"""
Interactive Multimodal Korean-English Translation Demo
Real-time translation with text input, live audio capture, and image upload/camera
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from pathlib import Path
import time
import threading
import queue
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from multimodal.multimodal_processor import MultimodalTranslationProcessor
    MULTIMODAL_AVAILABLE = True
    logger.info("Multimodal processor loaded successfully")
except ImportError as e:
    logger.error(f"Multimodal processor not available: {e}")
    MULTIMODAL_AVAILABLE = False

try:
    import sounddevice as sd
    import speech_recognition as sr
    HAS_AUDIO = True
    logger.info("Audio libraries available")
except ImportError:
    HAS_AUDIO = False
    logger.warning("Audio libraries not available - voice input disabled")

try:
    import pytesseract
    HAS_OCR = True
    logger.info("OCR libraries available")
except ImportError:
    HAS_OCR = False
    logger.warning("OCR libraries not available - image text extraction disabled")


class TranslationSession:
    """Manages translation session state and history."""
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.current_translation: Optional[str] = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def add_translation(self, original: str, translated: str, modality: str, confidence: float):
        """Add a translation to history."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original": original,
            "translated": translated,
            "modality": modality,
            "confidence": confidence
        }
        self.history.append(entry)
        self.current_translation = translated
        return entry
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get translation history."""
        return self.history
    
    def clear_history(self):
        """Clear translation history."""
        self.history.clear()
        self.current_translation = None


class AudioCapture:
    """Handles real-time audio capture and processing."""
    
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 3.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.recognizer = sr.Recognizer() if HAS_AUDIO else None
        self.is_recording = False
        self.audio_buffer = []
        
    def start_recording(self):
        """Start audio recording."""
        if not HAS_AUDIO:
            logger.error("Audio libraries not available")
            return
            
        self.is_recording = True
        self.audio_buffer = []
        logger.info("Started audio recording")
        
    def stop_recording(self) -> Optional[str]:
        """Stop recording and return audio data."""
        self.is_recording = False
        logger.info("Stopped audio recording")
        
        if not self.audio_buffer:
            return None
            
        # Combine audio chunks
        audio_data = np.concatenate(self.audio_buffer)
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Save as WAV file
            import soundfile as sf
            sf.write(temp_path, audio_data, self.sample_rate)
            return temp_path
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return None
    
    def record_chunk(self, duration: float = None) -> np.ndarray:
        """Record a single audio chunk."""
        if not HAS_AUDIO:
            logger.error("Audio libraries not available for recording")
            return np.array([])
            
        if duration is None:
            duration = self.chunk_duration
            
        try:
            # Record audio
            audio = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float32')
            sd.wait()  # Wait for recording to complete
            
            return audio.flatten()
        except Exception as e:
            logger.error(f"Error recording audio chunk: {e}")
            return np.array([])


class ImageProcessor:
    """Handles image processing and text extraction."""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
    def extract_text_from_image(self, image: Image.Image) -> Optional[str]:
        """Extract Korean text from image using OCR."""
        if not HAS_OCR:
            logger.warning("OCR not available")
            return None
            
        try:
            # Configure for Korean text
            custom_config = r'--oem 3 --psm 6 -l kor'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            if text.strip():
                logger.info(f"OCR extracted text: {text[:100]}...")
                return text.strip()
            else:
                logger.warning("No text found in image")
                return None
                
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return None
    
    def preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply threshold for better contrast
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(thresh)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image


class MultimodalTranslatorApp:
    """Main application class for the multimodal translation demo."""
    
    def __init__(self):
        self.session = TranslationSession()
        self.audio_capture = AudioCapture()
        self.image_processor = ImageProcessor()
        
        # Initialize multimodal processor
        if MULTIMODAL_AVAILABLE:
            try:
                self.processor = MultimodalTranslationProcessor(
                    tokenizer_path="data/processed_large_simple/sentencepiece_large.model",
                    model_path="models/extended/best_model.pt"
                )
                logger.info("Multimodal processor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize multimodal processor: {e}")
                self.processor = None
        else:
            self.processor = None
    
    def process_text_input(self, text: str) -> Dict[str, Any]:
        """Process text input for translation."""
        if not text.strip():
            return {"error": "No text provided"}
        
        if not self.processor:
            return {"error": "Translation processor not available"}
        
        try:
            results = self.processor.multimodal_translate(text_input=text)
            
            if results["status"] == "success":
                self.session.add_translation(
                    original=text,
                    translated=results["translation"],
                    modality="text",
                    confidence=results["confidence"]
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Text translation failed: {e}")
            return {"error": str(e)}
    
    def process_image_input(self, image: Image.Image) -> Dict[str, Any]:
        """Process image input for translation."""
        if not self.processor:
            return {"error": "Translation processor not available"}
        
        try:
            # Extract text from image
            extracted_text = self.image_processor.extract_text_from_image(image)
            
            if not extracted_text:
                # Try preprocessing
                processed_image = self.image_processor.preprocess_image_for_ocr(image)
                extracted_text = self.image_processor.extract_text_from_image(processed_image)
            
            if not extracted_text:
                return {"error": "No Korean text found in image"}
            
            # Translate the extracted text
            results = self.processor.multimodal_translate(text_input=extracted_text)
            
            if results["status"] == "success":
                self.session.add_translation(
                    original=extracted_text,
                    translated=results["translation"],
                    modality="image",
                    confidence=results["confidence"]
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Image translation failed: {e}")
            return {"error": str(e)}
    
    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Process audio file for translation."""
        if not self.processor:
            return {"error": "Translation processor not available"}
        
        try:
            results = self.processor.multimodal_translate(audio_path=audio_path)
            
            if results["status"] == "success":
                self.session.add_translation(
                    original=results["extracted_text"].get("audio", ""),
                    translated=results["translation"],
                    modality="audio",
                    confidence=results["confidence"]
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Audio translation failed: {e}")
            return {"error": str(e)}


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Korean-English Multimodal Translator",
        page_icon="🌏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'translator' not in st.session_state:
        st.session_state.translator = MultimodalTranslatorApp()
    
    translator = st.session_state.translator
    
    # Title and description
    st.title("🌏 Korean-English Multimodal Translator")
    st.markdown("""
    ### Real-time translation with text, images, and audio
    
    **Features:**
    - 📝 **Text Translation**: Type Korean text and get instant English translation
    - 🖼️ **Image Translation**: Upload images with Korean text or use camera
    - 🎤 **Audio Translation**: Record or upload Korean speech
    - 📊 **Translation History**: View all your translations with confidence scores
    """)
    
    # Sidebar for controls and history
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            translator.session.clear_history()
            st.success("History cleared!")
        
        # Settings
        with st.expander("🔧 Settings"):
            show_confidence = st.checkbox("Show confidence scores", value=True)
            auto_detect = st.checkbox("Auto-detect language", value=True)
        
        # Translation history
        st.header("📜 Translation History")
        history = translator.session.get_history()
        
        if history:
            for entry in reversed(history[-10:]):  # Show last 10 translations
                with st.container():
                    st.markdown(f"**{entry['modality'].upper()}** - {entry['timestamp']}")
                    st.text(f"🇰🇷 {entry['original'][:50]}...")
                    st.text(f"🇺🇸 {entry['translated'][:50]}...")
                    if show_confidence:
                        st.caption(f"Confidence: {entry['confidence']:.2f}")
                    st.divider()
        else:
            st.info("No translations yet. Start translating!")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text", "🖼️ Image", "🎤 Audio", "📊 All Features"])
    
    with tab1:
        st.header("Text Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🇰🇷 Korean Input")
            text_input = st.text_area(
                "Enter Korean text:",
                placeholder="안녕하세요\n오늘 날씨가 좋네요\n한국어를 영어로 번역합니다",
                height=150
            )
            
            translate_text_btn = st.button("🔄 Translate Text", type="primary", key="translate_text")
        
        with col2:
            st.subheader("🇺🇸 English Translation")
            
            if translate_text_btn and text_input.strip():
                with st.spinner("Translating..."):
                    results = translator.process_text_input(text_input)
                    
                    if "error" in results:
                        st.error(f"Translation failed: {results['error']}")
                    else:
                        st.success("Translation complete!")
                        st.text_area(
                            "Translation result:",
                            value=results["translation"],
                            height=150,
                            disabled=True
                        )
                        
                        if show_confidence:
                            st.metric("Confidence", f"{results['confidence']:.2%}")
            
            elif translator.session.current_translation:
                st.text_area(
                    "Latest translation:",
                    value=translator.session.current_translation,
                    height=150,
                    disabled=True
                )
    
    with tab2:
        st.header("Image Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Image Input")
            
            # Image upload
            uploaded_image = st.file_uploader(
                "Choose an image with Korean text",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                key="image_upload"
            )
            
            # Camera input
            if st.checkbox("Use camera"):
                camera_image = st.camera_input("Take a photo of Korean text")
                if camera_image:
                    uploaded_image = camera_image
            
            # Display uploaded image
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded image", use_column_width=True)
                
                translate_image_btn = st.button("🔍 Extract & Translate", type="primary", key="translate_image")
        
        with col2:
            st.subheader("🇺🇸 Translation Result")
            
            if uploaded_image and translate_image_btn:
                with st.spinner("Extracting text from image..."):
                    results = translator.process_image_input(image)
                    
                    if "error" in results:
                        st.error(f"Image translation failed: {results['error']}")
                    else:
                        st.success("Image translation complete!")
                        
                        # Show extracted text
                        st.text("Extracted Korean text:")
                        st.text_area(
                            "",
                            value=results["extracted_text"].get("text", ""),
                            height=80,
                            disabled=True
                        )
                        
                        # Show translation
                        st.text("English translation:")
                        st.text_area(
                            "",
                            value=results["translation"],
                            height=80,
                            disabled=True
                        )
                        
                        if show_confidence:
                            st.metric("Confidence", f"{results['confidence']:.2%}")
    
    with tab3:
        st.header("Audio Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎤 Audio Input")
            
            if not HAS_AUDIO:
                st.warning("Audio libraries not available. Please install: pip install sounddevice speechrecognition")
            else:
                # Audio recording
                recording_duration = st.slider("Recording duration (seconds)", 1, 10, 3)
                
                col_record1, col_record2 = st.columns(2)
                
                with col_record1:
                    if st.button("🔴 Start Recording", type="secondary", key="start_recording"):
                        with st.spinner(f"Recording for {recording_duration} seconds..."):
                            audio_path = translator.audio_capture.record_chunk(recording_duration)
                            
                            if audio_path:
                                st.success("Recording complete!")
                                st.audio(audio_path)
                                
                                # Translate the recorded audio
                                with st.spinner("Translating audio..."):
                                    results = translator.process_audio_file(audio_path)
                                    
                                    if "error" in results:
                                        st.error(f"Audio translation failed: {results['error']}")
                                    else:
                                        st.session_state.audio_results = results
                                        st.success("Audio translation complete!")
                            else:
                                st.error("Recording failed")
                
                with col_record2:
                    # Audio file upload
                    uploaded_audio = st.file_uploader(
                        "Or upload an audio file",
                        type=['wav', 'mp3', 'm4a', 'flac'],
                        key="audio_upload"
                    )
                    
                    if uploaded_audio:
                        # Save uploaded audio to temporary file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            temp_file.write(uploaded_audio.read())
                            temp_path = temp_file.name
                        
                        st.audio(temp_path)
                        
                        if st.button("🔄 Translate Audio File", type="primary", key="translate_audio_file"):
                            with st.spinner("Translating uploaded audio..."):
                                results = translator.process_audio_file(temp_path)
                                
                                if "error" in results:
                                    st.error(f"Audio translation failed: {results['error']}")
                                else:
                                    st.session_state.audio_results = results
                                    st.success("Audio translation complete!")
        
        with col2:
            st.subheader("🇺🇸 Translation Result")
            
            if 'audio_results' in st.session_state and st.session_state.audio_results:
                results = st.session_state.audio_results
                
                if results["status"] == "success":
                    st.text("Transcribed Korean text:")
                    st.text_area(
                        "",
                        value=results["extracted_text"].get("audio", ""),
                        height=80,
                        disabled=True
                    )
                    
                    st.text("English translation:")
                    st.text_area(
                        "",
                        value=results["translation"],
                        height=80,
                        disabled=True
                    )
                    
                    if show_confidence:
                        st.metric("Confidence", f"{results['confidence']:.2%}")
    
    with tab4:
        st.header("All Features Combined")
        
        st.markdown("""
        This tab demonstrates combining multiple input modalities:
        - Type text and upload an image
        - Record audio and add text context
        - Mix and match any combination!
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔄 Combined Input")
            
            # Text input
            combined_text = st.text_area(
                "Add text context (optional):",
                placeholder="Additional context or text...",
                height=100,
                key="combined_text"
            )
            
            # Image input
            combined_image = st.file_uploader(
                "Upload an image (optional):",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                key="combined_image"
            )
            
            # Audio input
            if HAS_AUDIO:
                st.write("Or record audio (optional):")
                combined_audio_duration = st.slider("Audio duration", 1, 5, 2, key="combined_audio_duration")
                
                if st.button("🔴 Record Audio", key="record_combined_audio"):
                    with st.spinner(f"Recording {combined_audio_duration} seconds..."):
                        audio_path = translator.audio_capture.record_chunk(combined_audio_duration)
                        if audio_path:
                            st.session_state.combined_audio_path = audio_path
                            st.audio(audio_path)
                            st.success("Audio recorded!")
            
            # Combine and translate button
            combine_translate = st.button(
                "🚀 Translate All Inputs",
                type="primary",
                key="translate_combined",
                disabled=not (combined_text or combined_image or st.session_state.get('combined_audio_path'))
            )
        
        with col2:
            st.subheader("🇺🇸 Combined Translation")
            
            if combine_translate:
                with st.spinner("Processing all inputs..."):
                    all_texts = []
                    all_translations = []
                    
                    # Process text
                    if combined_text.strip():
                        text_results = translator.process_text_input(combined_text)
                        if "error" not in text_results:
                            all_texts.append(f"Text: {combined_text}")
                            all_translations.append(text_results["translation"])
                    
                    # Process image
                    if combined_image:
                        image = Image.open(combined_image)
                        image_results = translator.process_image_input(image)
                        if "error" not in image_results:
                            all_texts.append(f"Image: {image_results['extracted_text'].get('text', '')}")
                            all_translations.append(image_results["translation"])
                    
                    # Process audio
                    if 'combined_audio_path' in st.session_state and st.session_state.combined_audio_path:
                        audio_results = translator.process_audio_file(st.session_state.combined_audio_path)
                        if "error" not in audio_results:
                            all_texts.append(f"Audio: {audio_results['extracted_text'].get('audio', '')}")
                            all_translations.append(audio_results["translation"])
                    
                    if all_texts and all_translations:
                        st.success("Combined translation complete!")
                        
                        st.text("Original inputs:")
                        for text in all_texts:
                            st.text(f"• {text}")
                        
                        st.text("Combined translation:")
                        combined_translation = " ".join(all_translations)
                        st.text_area(
                            "",
                            value=combined_translation,
                            height=100,
                            disabled=True
                        )
                    else:
                        st.warning("No valid inputs to translate")


def run_app():
    """Run the Streamlit application."""
    if not MULTIMODAL_AVAILABLE:
        st.error("""
        ❌ Multimodal translation components are not available.
        
        Please install the required dependencies:
        ```bash
        pip install torch torchvision torchaudio pillow pytesseract opencv-python
        pip install streamlit sounddevice speechrecognition
        ```
        """)
        return
    
    main()


if __name__ == "__main__":
    run_app()
