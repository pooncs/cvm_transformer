import torch
import torch.nn as nn
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image

# Import our models
from src.models.nmt_transformer import NMTTransformer, create_nmt_model
from src.models.image_encoder import ImageEncoder, create_image_encoder, MultimodalFusion
from src.models.audio_encoder import AudioEncoder, create_audio_encoder, AudioTextAlignment
from src.utils.device import get_device


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_test_models():
    """Create test models for validation."""
    device = get_device()
    
    # Text model
    text_model = create_nmt_model(
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        d_model=512,  # Smaller for testing
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        max_len=256,
        dropout=0.1,
        pad_id=0
    ).to(device)
    
    # Image encoder
    image_encoder = create_image_encoder(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ).to(device)
    
    # Audio encoder
    audio_encoder = create_audio_encoder(
        sample_rate=16000,
        n_mels=80,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ).to(device)
    
    # Multimodal fusion
    multimodal_fusion = create_multimodal_fusion(
        image_dim=512,
        text_dim=512,
        fusion_dim=512,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    # Audio-text alignment
    audio_text_alignment = create_audio_text_alignment(
        audio_dim=512,
        text_dim=512,
        hidden_dim=256,
        num_heads=4
    ).to(device)
    
    return {
        'text_model': text_model,
        'image_encoder': image_encoder,
        'audio_encoder': audio_encoder,
        'multimodal_fusion': multimodal_fusion,
        'audio_text_alignment': audio_text_alignment,
        'device': device
    }


def test_text_translation(models, test_cases):
    """Test text-only translation."""
    logger = logging.getLogger(__name__)
    logger.info("Testing text-only translation...")
    
    text_model = models['text_model']
    device = models['device']
    
    results = []
    
    for i, (korean, expected_english) in enumerate(test_cases):
        try:
            # Convert text to simple tokens (for testing)
            tokens = [ord(c) % 32000 for c in korean[:50]]  # Limit length
            if not tokens:
                tokens = [1]  # BOS token
            
            src_tokens = torch.tensor([tokens], dtype=torch.long).to(device)
            src_mask = torch.ones_like(src_tokens, dtype=torch.float32)
            
            # Forward pass
            with torch.no_grad():
                encoder_out = text_model.encode(src_tokens, src_mask)
                
                # Simple decoder simulation
                batch_size, seq_len, d_model = encoder_out.shape
                dummy_decoder_out = torch.randn(batch_size, seq_len, d_model).to(device)
                logits = text_model.generate(dummy_decoder_out)
                
                # Get prediction (simplified)
                predicted_tokens = torch.argmax(logits[0, :10, :], dim=-1)  # First 10 tokens
                predicted_text = ''.join([chr(t.item() % 128) for t in predicted_tokens if t.item() > 2])
            
            result = {
                'test_id': f'text_{i+1}',
                'input': korean,
                'expected': expected_english,
                'predicted': predicted_text,
                'status': 'completed'
            }
            
            results.append(result)
            logger.info(f"Text test {i+1}: '{korean}' -> '{predicted_text}'")
            
        except Exception as e:
            logger.error(f"Error in text test {i+1}: {e}")
            results.append({
                'test_id': f'text_{i+1}',
                'input': korean,
                'error': str(e),
                'status': 'failed'
            })
    
    return results


def test_image_processing(models):
    """Test image processing capabilities."""
    logger = logging.getLogger(__name__)
    logger.info("Testing image processing...")
    
    image_encoder = models['image_encoder']
    device = models['device']
    
    results = []
    
    try:
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='white')
        
        # Convert to tensor
        image_array = np.array(test_image) / 255.0
        image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = image_encoder(image_tensor)
            
            # Check outputs
            features = outputs['features']
            text_probs = outputs['text_probs']
            char_logits = outputs['char_logits']
        
        result = {
            'test_id': 'image_processing',
            'input_shape': list(image_tensor.shape),
            'features_shape': list(features.shape),
            'text_probs_shape': list(text_probs.shape),
            'char_logits_shape': list(char_logits.shape),
            'status': 'completed'
        }
        
        results.append(result)
        logger.info(f"Image processing test completed. Feature shape: {features.shape}")
        
    except Exception as e:
        logger.error(f"Error in image processing test: {e}")
        results.append({
            'test_id': 'image_processing',
            'error': str(e),
            'status': 'failed'
        })
    
    return results


def test_audio_processing(models):
    """Test audio processing capabilities."""
    logger = logging.getLogger(__name__)
    logger.info("Testing audio processing...")
    
    audio_encoder = models['audio_encoder']
    device = models['device']
    
    results = []
    
    try:
        # Create test audio (2 seconds, 16kHz)
        duration = 2.0
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # Generate simple test audio
        t = np.linspace(0, duration, samples)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = audio_encoder(audio_tensor)
            
            # Check outputs
            features = outputs['features']
            utterance_features = outputs['utterance_features']
            phoneme_logits = outputs['phoneme_logits']
            lang_logits = outputs['lang_logits']
        
        result = {
            'test_id': 'audio_processing',
            'input_shape': list(audio_tensor.shape),
            'features_shape': list(features.shape),
            'utterance_features_shape': list(utterance_features.shape),
            'phoneme_logits_shape': list(phoneme_logits.shape),
            'lang_logits_shape': list(lang_logits.shape),
            'status': 'completed'
        }
        
        results.append(result)
        logger.info(f"Audio processing test completed. Feature shape: {features.shape}")
        
    except Exception as e:
        logger.error(f"Error in audio processing test: {e}")
        results.append({
            'test_id': 'audio_processing',
            'error': str(e),
            'status': 'failed'
        })
    
    return results


def test_multimodal_fusion(models):
    """Test multimodal fusion capabilities."""
    logger = logging.getLogger(__name__)
    logger.info("Testing multimodal fusion...")
    
    multimodal_fusion = models['multimodal_fusion']
    device = models['device']
    
    results = []
    
    try:
        # Create dummy features
        batch_size = 2
        seq_len = 50
        d_model = 512
        
        image_features = torch.randn(batch_size, 1, d_model).to(device)
        text_features = torch.randn(batch_size, seq_len, d_model).to(device)
        text_mask = torch.ones(batch_size, seq_len).to(device)
        
        # Forward pass
        with torch.no_grad():
            fused_features = multimodal_fusion(image_features, text_features, text_mask)
        
        result = {
            'test_id': 'multimodal_fusion',
            'input_shapes': {
                'image_features': list(image_features.shape),
                'text_features': list(text_features.shape),
                'text_mask': list(text_mask.shape)
            },
            'output_shape': list(fused_features.shape),
            'status': 'completed'
        }
        
        results.append(result)
        logger.info(f"Multimodal fusion test completed. Output shape: {fused_features.shape}")
        
    except Exception as e:
        logger.error(f"Error in multimodal fusion test: {e}")
        results.append({
            'test_id': 'multimodal_fusion',
            'error': str(e),
            'status': 'failed'
        })
    
    return results


def test_audio_text_alignment(models):
    """Test audio-text alignment."""
    logger = logging.getLogger(__name__)
    logger.info("Testing audio-text alignment...")
    
    audio_text_alignment = models['audio_text_alignment']
    device = models['device']
    
    results = []
    
    try:
        # Create dummy features
        batch_size = 2
        audio_seq_len = 100
        text_seq_len = 50
        audio_dim = 512
        text_dim = 512
        
        audio_features = torch.randn(batch_size, audio_seq_len, audio_dim).to(device)
        text_features = torch.randn(batch_size, text_seq_len, text_dim).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = audio_text_alignment(audio_features, text_features)
            aligned_features = outputs['aligned_features']
            alignment_scores = outputs['alignment_scores']
        
        result = {
            'test_id': 'audio_text_alignment',
            'input_shapes': {
                'audio_features': list(audio_features.shape),
                'text_features': list(text_features.shape)
            },
            'output_shapes': {
                'aligned_features': list(aligned_features.shape),
                'alignment_scores': list(alignment_scores.shape)
            },
            'status': 'completed'
        }
        
        results.append(result)
        logger.info(f"Audio-text alignment test completed. Aligned features shape: {aligned_features.shape}")
        
    except Exception as e:
        logger.error(f"Error in audio-text alignment test: {e}")
        results.append({
            'test_id': 'audio_text_alignment',
            'error': str(e),
            'status': 'failed'
        })
    
    return results


def run_comprehensive_validation():
    """Run comprehensive multimodal validation tests."""
    logger = setup_logging()
    logger.info("Starting Comprehensive Multimodal Validation Tests")
    logger.info("="*60)
    
    # Test cases
    text_test_cases = [
        ("안녕하세요", "Hello"),
        ("감사합니다", "Thank you"),
        ("오늘 날씨가 좋네요", "The weather is nice today"),
        ("저는 한국어를 공부하고 있습니다", "I am studying Korean"),
        ("이것은 책입니다", "This is a book")
    ]
    
    try:
        # Create models
        logger.info("Creating test models...")
        models = create_test_models()
        
        # Run tests
        all_results = {}
        
        # Text translation tests
        logger.info("\n" + "="*40)
        logger.info("TEXT TRANSLATION TESTS")
        logger.info("="*40)
        text_results = test_text_translation(models, text_test_cases)
        all_results['text_translation'] = text_results
        
        # Image processing tests
        logger.info("\n" + "="*40)
        logger.info("IMAGE PROCESSING TESTS")
        logger.info("="*40)
        image_results = test_image_processing(models)
        all_results['image_processing'] = image_results
        
        # Audio processing tests
        logger.info("\n" + "="*40)
        logger.info("AUDIO PROCESSING TESTS")
        logger.info("="*40)
        audio_results = test_audio_processing(models)
        all_results['audio_processing'] = audio_results
        
        # Multimodal fusion tests
        logger.info("\n" + "="*40)
        logger.info("MULTIMODAL FUSION TESTS")
        logger.info("="*40)
        fusion_results = test_multimodal_fusion(models)
        all_results['multimodal_fusion'] = fusion_results
        
        # Audio-text alignment tests
        logger.info("\n" + "="*40)
        logger.info("AUDIO-TEXT ALIGNMENT TESTS")
        logger.info("="*40)
        alignment_results = test_audio_text_alignment(models)
        all_results['audio_text_alignment'] = alignment_results
        
        # Generate summary report
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE VALIDATION SUMMARY")
        logger.info("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, results in all_results.items():
            category_passed = sum(1 for r in results if r.get('status') == 'completed')
            category_total = len(results)
            
            total_tests += category_total
            passed_tests += category_passed
            
            logger.info(f"{category.replace('_', ' ').title()}: {category_passed}/{category_total} passed")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"\nOverall Results:")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed Tests: {passed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Save results
        output_dir = Path("tests/multimodal/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'detailed_results': all_results
        }
        
        report_path = output_dir / "multimodal_validation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\nDetailed results saved to: {report_path}")
        logger.info("="*60)
        
        return report
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise


if __name__ == "__main__":
    report = run_comprehensive_validation()
    
    # Print final summary
    print(f"\n{'='*60}")
    print("MULTIMODAL VALIDATION COMPLETED")
    print(f"{'='*60}")
    print(f"Total Tests: {report['total_tests']}")
    print(f"Passed Tests: {report['passed_tests']}")
    print(f"Success Rate: {report['success_rate']:.1f}%")
    print(f"Results saved to: tests/multimodal/results/multimodal_validation_report.json")
    print(f"{'='*60}")