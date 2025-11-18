#!/usr/bin/env python3
"""
Multimodal Translation Demo
Demonstrates image-to-text and audio-to-text translation capabilities
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import logging
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.multimodal.multimodal_processor import MultimodalTranslationProcessor
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    print(f"Multimodal components not available: {e}")
    MULTIMODAL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_sample_files():
    """Create sample files for demonstration if they don't exist."""
    sample_dir = Path("sample_multimodal")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a sample Korean text file
    text_file = sample_dir / "korean_text.txt"
    if not text_file.exists():
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("안녕하세요\n")
            f.write("오늘 날씨가 좋네요\n")
            f.write("한국어를 영어로 번역합니다\n")
        print(f"Created sample text file: {text_file}")
    
    # Create a simple image with Korean text (placeholder)
    # In a real scenario, users would provide actual images with Korean text
    image_file = sample_dir / "korean_text_image.jpg"
    if not image_file.exists():
        print(f"Note: Please place a Korean text image at {image_file} for image translation demo")
        print("You can create one using any image editor with Korean text, or use a photo of Korean signs/documents")
    
    # Create a simple audio file placeholder
    audio_file = sample_dir / "korean_speech.wav"
    if not audio_file.exists():
        print(f"Note: Please place a Korean speech audio file at {audio_file} for audio translation demo")
        print("You can record Korean speech or use existing Korean audio files")
    
    return sample_dir


def demo_text_translation(processor: MultimodalTranslationProcessor, text: str):
    """Demonstrate text translation."""
    print("\n" + "="*60)
    print("TEXT TRANSLATION DEMO")
    print("="*60)
    
    print(f"Input Text: {text}")
    
    results = processor.multimodal_translate(text_input=text)
    
    print(f"Status: {results['status']}")
    print(f"Extracted Text: {results['extracted_text']}")
    print(f"Translation: {results['translation']}")
    print(f"Confidence: {results['confidence']:.2f}")
    
    return results


def demo_image_translation(processor: MultimodalTranslationProcessor, image_path: Path):
    """Demonstrate image-to-text translation."""
    print("\n" + "="*60)
    print("IMAGE TRANSLATION DEMO")
    print("="*60)
    
    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        print("Please provide a Korean text image for translation")
        return None
    
    print(f"Input Image: {image_path}")
    
    results = processor.multimodal_translate(image_path=image_path)
    
    print(f"Status: {results['status']}")
    print(f"Input Modalities: {results['input_modalities']}")
    print(f"Extracted Text: {results['extracted_text']}")
    print(f"Translation: {results['translation']}")
    print(f"Confidence: {results['confidence']:.2f}")
    
    if "error" in results:
        print(f"Error: {results['error']}")
    
    return results


def demo_audio_translation(processor: MultimodalTranslationProcessor, audio_path: Path):
    """Demonstrate audio-to-text translation."""
    print("\n" + "="*60)
    print("AUDIO TRANSLATION DEMO")
    print("="*60)
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        print("Please provide a Korean speech audio file for translation")
        return None
    
    print(f"Input Audio: {audio_path}")
    
    results = processor.multimodal_translate(audio_path=audio_path)
    
    print(f"Status: {results['status']}")
    print(f"Input Modalities: {results['input_modalities']}")
    print(f"Extracted Text: {results['extracted_text']}")
    print(f"Translation: {results['translation']}")
    print(f"Confidence: {results['confidence']:.2f}")
    
    if "error" in results:
        print(f"Error: {results['error']}")
    
    return results


def demo_multimodal_translation(
    processor: MultimodalTranslationProcessor, 
    text: Optional[str] = None,
    image_path: Optional[Path] = None,
    audio_path: Optional[Path] = None
):
    """Demonstrate combined multimodal translation."""
    print("\n" + "="*60)
    print("MULTIMODAL TRANSLATION DEMO")
    print("="*60)
    
    print(f"Text Input: {text if text else 'None'}")
    print(f"Image Input: {image_path if image_path else 'None'}")
    print(f"Audio Input: {audio_path if audio_path else 'None'}")
    
    results = processor.multimodal_translate(
        text_input=text,
        image_path=image_path,
        audio_path=audio_path
    )
    
    print(f"Status: {results['status']}")
    print(f"Input Modalities: {results['input_modalities']}")
    print(f"Extracted Text: {results['extracted_text']}")
    print(f"Translation: {results['translation']}")
    print(f"Confidence: {results['confidence']:.2f}")
    
    if "error" in results:
        print(f"Error: {results['error']}")
    
    return results


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Multimodal Translation Demo")
    parser.add_argument("--mode", choices=["text", "image", "audio", "multimodal", "all"], 
                       default="all", help="Demo mode")
    parser.add_argument("--text", type=str, help="Text input for translation")
    parser.add_argument("--image", type=str, help="Image file path")
    parser.add_argument("--audio", type=str, help="Audio file path")
    parser.add_argument("--create-samples", action="store_true", 
                       help="Create sample files for demonstration")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if multimodal components are available
    if not MULTIMODAL_AVAILABLE:
        print("❌ Multimodal translation components are not available.")
        print("Please ensure all dependencies are installed:")
        print("pip install torch torchvision torchaudio pillow pytesseract opencv-python")
        return 1
    
    print("🚀 Korean-English Multimodal Translation Demo")
    print("="*60)
    
    # Create sample files if requested
    if args.create_samples:
        sample_dir = create_sample_files()
        print(f"Sample files created in: {sample_dir}")
    
    # Initialize processor
    print(f"Initializing multimodal processor on {args.device}...")
    try:
        processor = MultimodalTranslationProcessor(device=args.device)
        print("✅ Multimodal processor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return 1
    
    # Run demos based on mode
    if args.mode == "text" or args.mode == "all":
        # Text translation demo
        text_input = args.text or "안녕하세요, 오늘 날씨가 좋네요."
        demo_text_translation(processor, text_input)
    
    if args.mode == "image" or args.mode == "all":
        # Image translation demo
        image_path = Path(args.image) if args.image else Path("sample_multimodal/korean_text_image.jpg")
        demo_image_translation(processor, image_path)
    
    if args.mode == "audio" or args.mode == "all":
        # Audio translation demo
        audio_path = Path(args.audio) if args.audio else Path("sample_multimodal/korean_speech.wav")
        demo_audio_translation(processor, audio_path)
    
    if args.mode == "multimodal" or args.mode == "all":
        # Combined multimodal demo
        text_input = args.text if args.text else "이것은 멀티모달 번역 테스트입니다"
        image_path = Path(args.image) if args.image else None
        audio_path = Path(args.audio) if args.audio else None
        
        demo_multimodal_translation(processor, text_input, image_path, audio_path)
    
    print("\n" + "="*60)
    print("✅ Demo completed!")
    print("You can also use the pipeline directly:")
    print("python pipeline.py --stage multimodal --mode text --input-text '안녕하세요'")
    print("python pipeline.py --stage multimodal --mode image --input-file image.jpg")
    print("python pipeline.py --stage multimodal --mode audio --input-file audio.wav")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())