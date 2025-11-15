import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Add project root to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.multimodal_nmt import create_multimodal_nmt_model


class SimpleMultimodalValidation:
    """Simplified validation test for multimodal NMT system."""
    
    def __init__(self):
        """Initialize the validation test."""
        self.device = torch.device('cpu')  # Use CPU to avoid device mismatches
        self.logger = self._setup_logging()
        
        # Create model with smaller dimensions for testing
        self.model = self._create_test_model()
        
    def _setup_logging(self):
        """Setup logging for the test suite."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _create_test_model(self):
        """Create a smaller model for testing."""
        config = {
            'src_vocab_size': 1000,
            'tgt_vocab_size': 1000,
            'd_model': 256,  # Smaller dimension
            'n_heads': 8,    # Fewer heads
            'n_encoder_layers': 4,   # Fewer layers
            'n_decoder_layers': 4,   # Fewer layers
            'd_ff': 1024,    # Smaller feedforward
            'max_len': 128,  # Shorter sequences
            'dropout': 0.1,
            'use_images': True,
            'use_audio': True,
            'fusion_strategy': 'cross_attention'
        }
        
        model = create_multimodal_nmt_model(**config)
        model.to(self.device)
        model.eval()
        
        self.logger.info("Created test model with reduced dimensions")
        return model
    
    def create_test_data(self) -> List[Dict[str, Any]]:
        """Create simple test data."""
        test_cases = []
        
        # Simple Korean-English pairs
        pairs = [
            ("안녕", "Hello"),
            ("감사", "Thanks"),
            ("미안", "Sorry"),
            ("네", "Yes"),
            ("아니요", "No")
        ]
        
        for i, (korean, english) in enumerate(pairs):
            # Text-only test
            test_cases.append({
                'id': f'text_{i}',
                'source': korean,
                'target': english,
                'modalities': ['text']
            })
            
            # Text + Image test
            test_cases.append({
                'id': f'text_image_{i}',
                'source': korean,
                'target': english,
                'modalities': ['text', 'image']
            })
            
            # Text + Audio test
            test_cases.append({
                'id': f'text_audio_{i}',
                'source': korean,
                'target': english,
                'modalities': ['text', 'audio']
            })
        
        return test_cases
    
    def create_dummy_image(self, batch_size: int = 1) -> torch.Tensor:
        """Create dummy image tensor."""
        return torch.randn(batch_size, 3, 224, 224).to(self.device)
    
    def create_dummy_audio(self, batch_size: int = 1) -> torch.Tensor:
        """Create dummy audio tensor."""
        return torch.randn(batch_size, 1, 2048).to(self.device)
    
    def create_dummy_tokens(self, text: str, vocab_size: int = 1000, max_len: int = 128) -> torch.Tensor:
        """Create dummy tokenized text."""
        # Simple character-based tokenization
        tokens = [ord(c) % (vocab_size - 10) + 5 for c in text]  # Avoid special tokens
        tokens = [2] + tokens[:max_len-2] + [3]  # BOS and EOS
        
        # Pad or truncate
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
        
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def test_model_forward(self) -> Dict[str, Any]:
        """Test model forward pass with different modality combinations."""
        self.logger.info("Testing model forward pass...")
        
        results = []
        test_cases = self.create_test_data()
        
        for test_case in test_cases:
            try:
                start_time = time.time()
                
                # Create dummy inputs
                src_tokens = self.create_dummy_tokens(test_case['source'])
                tgt_tokens = self.create_dummy_tokens(test_case['target'])
                
                # Create modality-specific inputs
                images = None
                audio = None
                
                if 'image' in test_case['modalities']:
                    images = self.create_dummy_image()
                
                if 'audio' in test_case['modalities']:
                    audio = self.create_dummy_audio()
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        src_tokens=src_tokens,
                        tgt_tokens=tgt_tokens,
                        images=images,
                        audio=audio
                    )
                
                execution_time = time.time() - start_time
                
                # Check if outputs are valid
                success = True
                error_msg = None
                
                if 'logits' in outputs:
                    logits = outputs['logits']
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        success = False
                        error_msg = "NaN or Inf values in logits"
                else:
                    success = False
                    error_msg = "No logits in output"
                
                result = {
                    'test_id': test_case['id'],
                    'modalities': test_case['modalities'],
                    'success': success,
                    'error': error_msg,
                    'execution_time': execution_time,
                    'output_shape': outputs['logits'].shape if 'logits' in outputs else None,
                    'has_char_logits': 'char_logits' in outputs,
                    'has_phoneme_logits': 'phoneme_logits' in outputs,
                    'has_confidence': 'confidence' in outputs
                }
                
                results.append(result)
                
                self.logger.info(f"Test {test_case['id']}: {'PASSED' if success else 'FAILED'}")
                
            except Exception as e:
                self.logger.error(f"Test {test_case['id']} failed with exception: {e}")
                results.append({
                    'test_id': test_case['id'],
                    'modalities': test_case['modalities'],
                    'success': False,
                    'error': str(e),
                    'execution_time': 0.0,
                    'output_shape': None,
                    'has_char_logits': False,
                    'has_phoneme_logits': False,
                    'has_confidence': False
                })
        
        return results
    
    def test_model_inference(self) -> Dict[str, Any]:
        """Test model inference (generation) mode."""
        self.logger.info("Testing model inference...")
        
        results = []
        test_cases = self.create_test_data()[:3]  # Use fewer cases for inference
        
        for test_case in test_cases:
            try:
                start_time = time.time()
                
                # Create dummy inputs (no target tokens for inference)
                src_tokens = self.create_dummy_tokens(test_case['source'])
                
                # Create modality-specific inputs
                images = None
                audio = None
                
                if 'image' in test_case['modalities']:
                    images = self.create_dummy_image()
                
                if 'audio' in test_case['modalities']:
                    audio = self.create_dummy_audio()
                
                # Inference mode (no tgt_tokens)
                with torch.no_grad():
                    outputs = self.model(
                        src_tokens=src_tokens,
                        images=images,
                        audio=audio,
                        max_len=50,  # Shorter generation for testing
                        beam_size=1  # Greedy decoding
                    )
                
                execution_time = time.time() - start_time
                
                # Check if outputs are valid
                success = True
                error_msg = None
                
                if 'generated' in outputs:
                    generated = outputs['generated']
                    if torch.isnan(generated).any() or torch.isinf(generated).any():
                        success = False
                        error_msg = "NaN or Inf values in generated tokens"
                else:
                    success = False
                    error_msg = "No generated tokens in output"
                
                result = {
                    'test_id': test_case['id'],
                    'modalities': test_case['modalities'],
                    'success': success,
                    'error': error_msg,
                    'execution_time': execution_time,
                    'generated_shape': outputs['generated'].shape if 'generated' in outputs else None
                }
                
                results.append(result)
                self.logger.info(f"Inference test {test_case['id']}: {'PASSED' if success else 'FAILED'}")
                
            except Exception as e:
                self.logger.error(f"Inference test {test_case['id']} failed: {e}")
                results.append({
                    'test_id': test_case['id'],
                    'modalities': test_case['modalities'],
                    'success': False,
                    'error': str(e),
                    'execution_time': 0.0,
                    'generated_shape': None
                })
        
        return results
    
    def generate_report(self, forward_results: List[Dict], inference_results: List[Dict]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("SIMPLE MULTIMODAL NMT VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Forward pass results
        report.append("FORWARD PASS TESTS:")
        total_forward = len(forward_results)
        passed_forward = sum(1 for r in forward_results if r['success'])
        
        report.append(f"Total Tests: {total_forward}")
        report.append(f"Passed: {passed_forward}")
        report.append(f"Success Rate: {passed_forward/total_forward:.2%}")
        report.append("")
        
        # Analyze by modality
        modality_stats = {}
        for result in forward_results:
            modality = '+'.join(result['modalities'])
            if modality not in modality_stats:
                modality_stats[modality] = {'total': 0, 'passed': 0}
            modality_stats[modality]['total'] += 1
            if result['success']:
                modality_stats[modality]['passed'] += 1
        
        report.append("PERFORMANCE BY MODALITY:")
        for modality, stats in modality_stats.items():
            success_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            report.append(f"  {modality}: {success_rate:.2%} ({stats['passed']}/{stats['total']})")
        report.append("")
        
        # Inference results
        report.append("INFERENCE TESTS:")
        total_inference = len(inference_results)
        passed_inference = sum(1 for r in inference_results if r['success'])
        
        report.append(f"Total Tests: {total_inference}")
        report.append(f"Passed: {passed_inference}")
        report.append(f"Success Rate: {passed_inference/total_inference:.2%}")
        report.append("")
        
        # Average execution times
        avg_forward_time = np.mean([r['execution_time'] for r in forward_results if r['success']])
        avg_inference_time = np.mean([r['execution_time'] for r in inference_results if r['success']])
        
        report.append("PERFORMANCE METRICS:")
        report.append(f"Average Forward Pass Time: {avg_forward_time:.4f}s")
        report.append(f"Average Inference Time: {avg_inference_time:.4f}s")
        report.append("")
        
        # Feature analysis
        char_detection = sum(1 for r in forward_results if r['has_char_logits'])
        phoneme_detection = sum(1 for r in forward_results if r['has_phoneme_logits'])
        confidence_scores = sum(1 for r in forward_results if r['has_confidence'])
        
        report.append("FEATURE ANALYSIS:")
        report.append(f"Character Detection Available: {char_detection}/{total_forward}")
        report.append(f"Phoneme Detection Available: {phoneme_detection}/{total_forward}")
        report.append(f"Confidence Scores Available: {confidence_scores}/{total_forward}")
        report.append("")
        
        # Failed tests
        failed_tests = [r for r in forward_results if not r['success']]
        if failed_tests:
            report.append("FAILED TESTS:")
            for test in failed_tests[:5]:  # Show first 5 failures
                report.append(f"  {test['test_id']}: {test['error']}")
            if len(failed_tests) > 5:
                report.append(f"  ... and {len(failed_tests) - 5} more failures")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_validation(self) -> Dict[str, Any]:
        """Run the complete validation test suite."""
        self.logger.info("Starting multimodal NMT validation...")
        
        # Test forward pass
        forward_results = self.test_model_forward()
        
        # Test inference
        inference_results = self.test_model_inference()
        
        # Generate report
        report = self.generate_report(forward_results, inference_results)
        
        # Calculate summary statistics
        total_forward = len(forward_results)
        passed_forward = sum(1 for r in forward_results if r['success'])
        
        total_inference = len(inference_results)
        passed_inference = sum(1 for r in inference_results if r['success'])
        
        summary = {
            'total_forward_tests': total_forward,
            'passed_forward_tests': passed_forward,
            'forward_success_rate': passed_forward / total_forward if total_forward > 0 else 0,
            'total_inference_tests': total_inference,
            'passed_inference_tests': passed_inference,
            'inference_success_rate': passed_inference / total_inference if total_inference > 0 else 0,
            'forward_results': forward_results,
            'inference_results': inference_results,
            'report': report
        }
        
        return summary


def main():
    """Main function to run the validation test suite."""
    print("Starting Simple Multimodal NMT Validation...")
    
    # Initialize validation test
    validator = SimpleMultimodalValidation()
    
    # Run validation
    results = validator.run_validation()
    
    # Print report
    print(results['report'])
    
    # Save results
    output_dir = Path('tests/comprehensive')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    with open(output_dir / 'simple_multimodal_results.json', 'w') as f:
        json.dump({
            'summary': {
                'forward_success_rate': results['forward_success_rate'],
                'inference_success_rate': results['inference_success_rate'],
                'total_forward_tests': results['total_forward_tests'],
                'passed_forward_tests': results['passed_forward_tests'],
                'total_inference_tests': results['total_inference_tests'],
                'passed_inference_tests': results['passed_inference_tests']
            },
            'forward_results': results['forward_results'],
            'inference_results': results['inference_results']
        }, f, indent=2)
    
    # Save text report
    with open(output_dir / 'simple_multimodal_report.txt', 'w') as f:
        f.write(results['report'])
    
    print(f"\nValidation completed!")
    print(f"Forward pass success rate: {results['forward_success_rate']:.2%}")
    print(f"Inference success rate: {results['inference_success_rate']:.2%}")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()