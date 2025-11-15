import torch
import torch.nn as nn
import numpy as np
import pytest
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from PIL import Image
import tempfile
import soundfile as sf

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.multimodal_nmt import create_multimodal_nmt_model
from src.models.multimodal_encoders import ImageEncoder, AudioEncoder, MultimodalFusion
from src.data.prepare_corpus import ParallelCorpusProcessor


class MultimodalValidationTestSuite:
    """Comprehensive validation test suite for multimodal NMT system."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the validation test suite."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.logger = self._setup_logging()
        
        # Load model and config
        self.model, self.config = self._load_model_and_config(model_path, config_path)
        
        # Test data
        self.test_cases = self._create_test_cases()
        
    def _setup_logging(self):
        """Setup logging for the test suite."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_model_and_config(self, model_path: Optional[str], config_path: Optional[str]):
        """Load the multimodal NMT model and configuration."""
        # Default configuration
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
            'use_images': True,
            'use_audio': True,
            'fusion_strategy': 'cross_attention'
        }
        
        # Load config from file if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config.update(json.load(f))
        
        # Create model
        model = create_multimodal_nmt_model(**config)
        
        # Load model weights if provided
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded model from {model_path}")
        else:
            self.logger.info("Using randomly initialized model for testing")
        
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases covering different scenarios."""
        test_cases = []
        
        # Korean-English translation test cases
        korean_english_pairs = [
            ("안녕하세요", "Hello"),
            ("감사합니다", "Thank you"),
            ("죄송합니다", "I'm sorry"),
            ("네, 알겠습니다", "Yes, I understand"),
            ("아니요, 괜찮습니다", "No, it's okay"),
            ("도와주세요", "Help me"),
            ("어디에 있나요?", "Where is it?"),
            ("얼마예요?", "How much is it?"),
            ("맛있어요", "It's delicious"),
            ("추워요", "It's cold"),
            ("더워요", "It's hot"),
            ("피곤해요", "I'm tired"),
            ("행복해요", "I'm happy"),
            ("슬퍼요", "I'm sad"),
            ("화나요", "I'm angry"),
            ("무서워요", "I'm scared"),
            ("놀랐어요", "I'm surprised"),
            ("궁금해요", "I'm curious"),
            ("졸려요", "I'm sleepy"),
            ("배고파요", "I'm hungry")
        ]
        
        # Create test cases with different modality combinations
        for i, (korean, english) in enumerate(korean_english_pairs):
            base_case = {
                'id': f'text_only_{i}',
                'source': korean,
                'target': english,
                'expected_translation': english,
                'difficulty': 1.0,
                'domain': 'daily_conversation'
            }
            
            # Text-only test case
            text_case = base_case.copy()
            text_case['id'] = f'text_only_{i}'
            text_case['modalities'] = ['text']
            test_cases.append(text_case)
            
            # Text + Image test case
            image_case = base_case.copy()
            image_case['id'] = f'text_image_{i}'
            image_case['modalities'] = ['text', 'image']
            image_case['image_description'] = f"Image showing: {english}"
            test_cases.append(image_case)
            
            # Text + Audio test case
            audio_case = base_case.copy()
            audio_case['id'] = f'text_audio_{i}'
            audio_case['modalities'] = ['text', 'audio']
            audio_case['audio_description'] = f"Audio saying: {korean}"
            test_cases.append(audio_case)
            
            # Multimodal test case (text + image + audio)
            multimodal_case = base_case.copy()
            multimodal_case['id'] = f'multimodal_{i}'
            multimodal_case['modalities'] = ['text', 'image', 'audio']
            multimodal_case['image_description'] = f"Image showing: {english}"
            multimodal_case['audio_description'] = f"Audio saying: {korean}"
            test_cases.append(multimodal_case)
        
        # Add challenging cases
        challenging_cases = [
            {
                'id': 'long_sentence',
                'source': '오늘 날씨가 정말 좋고 햇살이 따뜻해서 공원에 산책하러 가기로 결정했습니다.',
                'target': 'The weather is really nice today and the sunshine is warm, so I decided to go for a walk in the park.',
                'expected_translation': 'The weather is really nice today and the sunshine is warm, so I decided to go for a walk in the park.',
                'modalities': ['text'],
                'difficulty': 2.0,
                'domain': 'narrative'
            },
            {
                'id': 'complex_grammar',
                'source': '만약 내일 비가 오지 않으면 친구들과 영화를 보러 갈 것입니다.',
                'target': 'If it does not rain tomorrow, I will go watch a movie with my friends.',
                'expected_translation': 'If it does not rain tomorrow, I will go watch a movie with my friends.',
                'modalities': ['text'],
                'difficulty': 2.5,
                'domain': 'conditional'
            },
            {
                'id': 'idiomatic',
                'source': '콧대가 높다',
                'target': 'To be arrogant',
                'expected_translation': 'To be arrogant',
                'modalities': ['text'],
                'difficulty': 3.0,
                'domain': 'idiom'
            }
        ]
        
        test_cases.extend(challenging_cases)
        
        return test_cases
    
    def create_test_images(self, test_case: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Create synthetic test images for multimodal testing."""
        if 'image' not in test_case.get('modalities', []):
            return None
        
        # Create a simple synthetic image with text overlay
        try:
            # Create base image
            image = Image.new('RGB', (224, 224), color='white')
            
            # Add text description to image (simulating OCR scenario)
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Try to use a default font, fallback if not available
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Add Korean text to image
            text = test_case['source']
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the text
            x = (224 - text_width) // 2
            y = (224 - text_height) // 2
            
            # Draw text with shadow effect
            draw.text((x+2, y+2), text, fill='gray', font=font)
            draw.text((x, y), text, fill='black', font=font)
            
            # Convert to tensor and normalize
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            # Ensure tensor is on CPU first, then move to device
            return image_tensor.to(self.device)
            
        except Exception as e:
            self.logger.warning(f"Failed to create test image: {e}")
            return torch.zeros(1, 3, 224, 224, device=self.device)
    
    def create_test_audio(self, test_case: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Create synthetic test audio for multimodal testing."""
        if 'audio' not in test_case.get('modalities', []):
            return None
        
        # Create synthetic audio (sine wave with varying frequencies)
        try:
            duration = 2.0  # 2 seconds
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create a simple melody that represents the text
            frequencies = [440, 523, 659, 784, 880]  # A, C, E, G, A
            audio = np.zeros_like(t)
            
            # Add some harmonic content
            for i, freq in enumerate(frequencies):
                amplitude = 0.2 * (i + 1) / len(frequencies)
                audio += amplitude * np.sin(2 * np.pi * freq * t)
            
            # Add envelope
            envelope = np.exp(-t * 2)  # Exponential decay
            audio *= envelope
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            return audio_tensor.to(self.device)
            
        except Exception as e:
            self.logger.warning(f"Failed to create test audio: {e}")
            return torch.zeros(1, 1, 2048).to(self.device)
    
    def tokenize_text(self, text: str, max_len: int = 512) -> torch.Tensor:
        """Tokenize text for model input."""
        # Simple tokenization (replace with actual tokenizer)
        tokens = [hash(char) % 30000 + 10 for char in text]  # Dummy tokenization, avoid special tokens
        tokens = [2] + tokens[:max_len-2] + [3]  # Add BOS and EOS
        
        # Pad to max length
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))
        
        return torch.tensor(tokens[:max_len], dtype=torch.long).unsqueeze(0).to(self.device)
    
    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case."""
        self.logger.info(f"Running test: {test_case['id']}")
        
        start_time = time.time()
        
        # Prepare inputs
        src_tokens = self.tokenize_text(test_case['source'])
        images = self.create_test_images(test_case)
        audio = self.create_test_audio(test_case)
        
        # Create dummy target tokens for evaluation
        tgt_tokens = self.tokenize_text(test_case['expected_translation'])
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
                images=images,
                audio=audio,
                return_features=True
            )
            
            # Get predictions
            if 'logits' in outputs:
                predictions = torch.argmax(outputs['logits'], dim=-1)
                predicted_text = self._tokens_to_text(predictions[0])
            else:
                predicted_text = ""
            
            # Calculate metrics
            bleu_score = self._calculate_bleu(
                predicted_text, test_case['expected_translation']
            )
            
            exact_match = float(
                predicted_text.strip().lower() == test_case['expected_translation'].strip().lower()
            )
            
            # Calculate confidence if available
            confidence = outputs.get('confidence', torch.tensor(0.0)).item()
            
            # Check if this is a perfect translation
            is_perfect = exact_match == 1.0
        
        execution_time = time.time() - start_time
        
        result = {
            'test_id': test_case['id'],
            'source': test_case['source'],
            'expected': test_case['expected_translation'],
            'predicted': predicted_text,
            'bleu_score': bleu_score,
            'exact_match': exact_match,
            'confidence': confidence,
            'is_perfect': is_perfect,
            'execution_time': execution_time,
            'modalities': test_case.get('modalities', ['text']),
            'difficulty': test_case.get('difficulty', 1.0),
            'domain': test_case.get('domain', 'general')
        }
        
        return result
    
    def _tokens_to_text(self, tokens: torch.Tensor) -> str:
        """Convert tokens back to text (simplified)."""
        # Remove special tokens
        tokens = tokens[tokens != 0]  # Remove padding
        tokens = tokens[tokens != 2]  # Remove BOS
        tokens = tokens[tokens != 3]  # Remove EOS
        
        # Simple detokenization (replace with actual detokenizer)
        if len(tokens) > 0:
            # Convert back to characters (simplified)
            text = ''.join([chr(token % 1000 + 44032) for token in tokens if token > 0])
            return text
        return ""
    
    def _calculate_bleu(self, predicted: str, reference: str) -> float:
        """Calculate BLEU score (simplified)."""
        if not predicted or not reference:
            return 0.0
        
        # Simple word-level BLEU calculation
        pred_words = predicted.lower().split()
        ref_words = reference.lower().split()
        
        if not pred_words or not ref_words:
            return 0.0
        
        # Calculate precision
        matches = sum(1 for word in pred_words if word in ref_words)
        precision = matches / len(pred_words) if pred_words else 0.0
        
        # Calculate brevity penalty
        bp = min(1.0, len(pred_words) / len(ref_words)) if ref_words else 0.0
        
        return bp * precision
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and compile results."""
        self.logger.info("Starting comprehensive multimodal validation tests...")
        
        all_results = []
        perfect_translations = 0
        total_bleu = 0.0
        total_time = 0.0
        
        # Run each test case
        for test_case in self.test_cases:
            try:
                result = self.run_single_test(test_case)
                all_results.append(result)
                
                if result['is_perfect']:
                    perfect_translations += 1
                
                total_bleu += result['bleu_score']
                total_time += result['execution_time']
                
            except Exception as e:
                self.logger.error(f"Test {test_case['id']} failed: {e}")
                # Add failed test result
                all_results.append({
                    'test_id': test_case['id'],
                    'source': test_case['source'],
                    'expected': test_case['expected_translation'],
                    'predicted': 'FAILED',
                    'bleu_score': 0.0,
                    'exact_match': 0.0,
                    'confidence': 0.0,
                    'is_perfect': False,
                    'execution_time': 0.0,
                    'modalities': test_case.get('modalities', ['text']),
                    'difficulty': test_case.get('difficulty', 1.0),
                    'domain': test_case.get('domain', 'general'),
                    'error': str(e)
                })
        
        # Calculate summary statistics
        total_tests = len(all_results)
        perfect_rate = perfect_translations / total_tests if total_tests > 0 else 0.0
        average_bleu = total_bleu / total_tests if total_tests > 0 else 0.0
        average_time = total_time / total_tests if total_tests > 0 else 0.0
        
        # Analyze by modality
        modality_results = self._analyze_by_modality(all_results)
        
        # Analyze by difficulty
        difficulty_results = self._analyze_by_difficulty(all_results)
        
        # Analyze by domain
        domain_results = self._analyze_by_domain(all_results)
        
        summary = {
            'total_tests': total_tests,
            'perfect_translations': perfect_translations,
            'perfect_translation_rate': perfect_rate,
            'average_bleu_score': average_bleu,
            'average_execution_time': average_time,
            'target_99_percent_achieved': perfect_rate >= 0.99,
            'modality_analysis': modality_results,
            'difficulty_analysis': difficulty_results,
            'domain_analysis': domain_results,
            'detailed_results': all_results
        }
        
        return summary
    
    def _analyze_by_modality(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results by input modality."""
        modality_stats = {}
        
        for result in results:
            modalities = tuple(sorted(result['modalities']))
            if modalities not in modality_stats:
                modality_stats[modalities] = {
                    'count': 0,
                    'perfect_count': 0,
                    'total_bleu': 0.0,
                    'total_time': 0.0
                }
            
            stats = modality_stats[modalities]
            stats['count'] += 1
            stats['perfect_count'] += result['is_perfect']
            stats['total_bleu'] += result['bleu_score']
            stats['total_time'] += result['execution_time']
        
        # Calculate averages
        analysis = {}
        for modalities, stats in modality_stats.items():
            mod_key = '+'.join(modalities)
            analysis[mod_key] = {
                'count': stats['count'],
                'perfect_rate': stats['perfect_count'] / stats['count'] if stats['count'] > 0 else 0.0,
                'average_bleu': stats['total_bleu'] / stats['count'] if stats['count'] > 0 else 0.0,
                'average_time': stats['total_time'] / stats['count'] if stats['count'] > 0 else 0.0
            }
        
        return analysis
    
    def _analyze_by_difficulty(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results by difficulty level."""
        difficulty_stats = {}
        
        for result in results:
            difficulty = result['difficulty']
            difficulty_bucket = f"level_{int(difficulty)}"
            
            if difficulty_bucket not in difficulty_stats:
                difficulty_stats[difficulty_bucket] = {
                    'count': 0,
                    'perfect_count': 0,
                    'total_bleu': 0.0,
                    'total_time': 0.0
                }
            
            stats = difficulty_stats[difficulty_bucket]
            stats['count'] += 1
            stats['perfect_count'] += result['is_perfect']
            stats['total_bleu'] += result['bleu_score']
            stats['total_time'] += result['execution_time']
        
        # Calculate averages
        analysis = {}
        for difficulty, stats in difficulty_stats.items():
            analysis[difficulty] = {
                'count': stats['count'],
                'perfect_rate': stats['perfect_count'] / stats['count'] if stats['count'] > 0 else 0.0,
                'average_bleu': stats['total_bleu'] / stats['count'] if stats['count'] > 0 else 0.0,
                'average_time': stats['total_time'] / stats['count'] if stats['count'] > 0 else 0.0
            }
        
        return analysis
    
    def _analyze_by_domain(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results by domain."""
        domain_stats = {}
        
        for result in results:
            domain = result['domain']
            if domain not in domain_stats:
                domain_stats[domain] = {
                    'count': 0,
                    'perfect_count': 0,
                    'total_bleu': 0.0,
                    'total_time': 0.0
                }
            
            stats = domain_stats[domain]
            stats['count'] += 1
            stats['perfect_count'] += result['is_perfect']
            stats['total_bleu'] += result['bleu_score']
            stats['total_time'] += result['execution_time']
        
        # Calculate averages
        analysis = {}
        for domain, stats in domain_stats.items():
            analysis[domain] = {
                'count': stats['count'],
                'perfect_rate': stats['perfect_count'] / stats['count'] if stats['count'] > 0 else 0.0,
                'average_bleu': stats['total_bleu'] / stats['count'] if stats['count'] > 0 else 0.0,
                'average_time': stats['total_time'] / stats['count'] if stats['count'] > 0 else 0.0
            }
        
        return analysis
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save test results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MULTIMODAL NMT VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL PERFORMANCE:")
        report.append(f"Total Tests: {results['total_tests']}")
        report.append(f"Perfect Translations: {results['perfect_translations']}")
        report.append(f"Perfect Translation Rate: {results['perfect_translation_rate']:.2%}")
        report.append(f"Average BLEU Score: {results['average_bleu_score']:.4f}")
        report.append(f"Average Execution Time: {results['average_execution_time']:.4f}s")
        report.append(f"Target 99% Achieved: {'✅ YES' if results['target_99_percent_achieved'] else '❌ NO'}")
        report.append("")
        
        # Modality analysis
        report.append("PERFORMANCE BY INPUT MODALITY:")
        for modality, stats in results['modality_analysis'].items():
            report.append(f"  {modality}:")
            report.append(f"    Tests: {stats['count']}")
            report.append(f"    Perfect Rate: {stats['perfect_rate']:.2%}")
            report.append(f"    Average BLEU: {stats['average_bleu']:.4f}")
            report.append(f"    Average Time: {stats['average_time']:.4f}s")
        report.append("")
        
        # Difficulty analysis
        report.append("PERFORMANCE BY DIFFICULTY:")
        for difficulty, stats in results['difficulty_analysis'].items():
            report.append(f"  {difficulty}:")
            report.append(f"    Tests: {stats['count']}")
            report.append(f"    Perfect Rate: {stats['perfect_rate']:.2%}")
            report.append(f"    Average BLEU: {stats['average_bleu']:.4f}")
        report.append("")
        
        # Domain analysis
        report.append("PERFORMANCE BY DOMAIN:")
        for domain, stats in results['domain_analysis'].items():
            report.append(f"  {domain}:")
            report.append(f"    Tests: {stats['count']}")
            report.append(f"    Perfect Rate: {stats['perfect_rate']:.2%}")
            report.append(f"    Average BLEU: {stats['average_bleu']:.4f}")
        report.append("")
        
        # Show some example translations
        report.append("EXAMPLE TRANSLATIONS:")
        perfect_examples = [r for r in results['detailed_results'] if r['is_perfect']][:3]
        imperfect_examples = [r for r in results['detailed_results'] if not r['is_perfect']][:3]
        
        if perfect_examples:
            report.append("  Perfect Translations:")
            for example in perfect_examples:
                report.append(f"    Korean: {example['source']}")
                report.append(f"    English: {example['predicted']}")
                report.append("")
        
        if imperfect_examples:
            report.append("  Imperfect Translations:")
            for example in imperfect_examples:
                report.append(f"    Korean: {example['source']}")
                report.append(f"    Expected: {example['expected']}")
                report.append(f"    Predicted: {example['predicted']}")
                report.append(f"    BLEU: {example['bleu_score']:.4f}")
                report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main function to run the validation test suite."""
    print("Starting Multimodal NMT Validation Test Suite...")
    
    # Initialize test suite
    test_suite = MultimodalValidationTestSuite()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Generate and print report
    report = test_suite.generate_report(results)
    print(report)
    
    # Save results
    test_suite.save_results(results, 'tests/comprehensive/multimodal_validation_results.json')
    
    # Save detailed report
    with open('tests/comprehensive/multimodal_validation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\nValidation test suite completed!")
    print(f"Results saved to tests/comprehensive/multimodal_validation_results.json")
    print(f"Report saved to tests/comprehensive/multimodal_validation_report.txt")


if __name__ == '__main__':
    main()