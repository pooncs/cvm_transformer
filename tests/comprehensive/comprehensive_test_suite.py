"""
Comprehensive test suite for Korean words to English translation.
Includes text, images, and audio clips for multimodal testing.
"""

import torch
import torch.nn as nn
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
import sys
sys.path.append('.')
# Import the EnhancedTranslationModel from the training script
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/training'))
from train_optimized import EnhancedTranslationModel
from src.models.sp_tokenizer import SPTokenizer
from src.utils.metrics import BLEUScore, compute_translation_accuracy

@dataclass
class TestCase:
    """Test case data structure."""
    id: str
    korean_text: str
    expected_english: str
    category: str  # 'basic', 'intermediate', 'advanced', 'domain_specific'
    difficulty: int  # 1-5 scale
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Optional[Dict] = None

class ComprehensiveTestSuite:
    """Comprehensive test suite for Korean-English translation."""
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = 'auto'):
        self.device = torch.device(device if device != 'auto' else 
                                  ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load model and tokenizer
        self.model = self._load_model(model_path)
        self.tokenizer = SPTokenizer(tokenizer_path)
        self.bleu_metric = BLEUScore()
        
        # Test results
        self.results = {}
        self.test_cases = []
        self.execution_times = []
        
        # Create test directories
        Path("tests/comprehensive/data").mkdir(parents=True, exist_ok=True)
        Path("tests/comprehensive/results").mkdir(parents=True, exist_ok=True)
        Path("tests/comprehensive/reports").mkdir(parents=True, exist_ok=True)
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with saved config
        config = checkpoint.get('config', {})
        model = EnhancedTranslationModel(
            vocab_size=config.get('vocab_size', 32000),
            d_model=config.get('d_model', 1024),
            n_heads=config.get('nhead', 16),
            n_layers=config.get('n_layers_student', 8),
            ff_dim=config.get('dim_feedforward', 4096),
            max_len=config.get('max_len', 128),
            pad_id=0
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def generate_test_cases(self) -> List[TestCase]:
        """Generate comprehensive test cases covering various categories."""
        test_cases = []
        
        # Basic vocabulary tests
        basic_cases = [
            ("안녕하세요", "Hello", 1),
            ("감사합니다", "Thank you", 1),
            ("미안합니다", "Sorry", 1),
            ("네", "Yes", 1),
            ("아니요", "No", 1),
            ("물", "Water", 1),
            ("밥", "Rice", 1),
            ("학교", "School", 2),
            ("집", "House", 1),
            ("차", "Car", 1),
            ("책", "Book", 1),
            ("컴퓨터", "Computer", 2),
            ("전화", "Phone", 2),
            ("시간", "Time", 2),
            ("날씨", "Weather", 2),
            ("돈", "Money", 1),
            ("사랑", "Love", 2),
            ("친구", "Friend", 1),
            ("가족", "Family", 2),
            ("일", "Work", 1),
        ]
        
        for i, (korean, english, difficulty) in enumerate(basic_cases):
            test_cases.append(TestCase(
                id=f"basic_{i:03d}",
                korean_text=korean,
                expected_english=english,
                category="basic",
                difficulty=difficulty
            ))
        
        # Intermediate phrase tests
        intermediate_cases = [
            ("오늘 날씨가 어때요?", "How is the weather today?", 3),
            ("점심 먹었어요?", "Did you eat lunch?", 3),
            ("어디 가세요?", "Where are you going?", 3),
            ("몇 시예요?", "What time is it?", 3),
            ("얼마예요?", "How much is it?", 3),
            ("도와주세요", "Please help me", 3),
            ("같이 갈래요?", "Do you want to go together?", 3),
            ("기다려주세요", "Please wait", 3),
            ("빨리 와주세요", "Please come quickly", 3),
            ("조용히 해주세요", "Please be quiet", 3),
            ("즐거운 시간 되세요", "Have a good time", 3),
            ("다음에 봐요", "See you next time", 3),
            ("오랜만이에요", "Long time no see", 3),
            ("건강하세요", "Stay healthy", 3),
            ("행복하세요", "Be happy", 3),
        ]
        
        for i, (korean, english, difficulty) in enumerate(intermediate_cases):
            test_cases.append(TestCase(
                id=f"intermediate_{i:03d}",
                korean_text=korean,
                expected_english=english,
                category="intermediate",
                difficulty=difficulty
            ))
        
        # Advanced sentence tests
        advanced_cases = [
            ("한국 문화는 매우 흥미롭고 독특합니다", "Korean culture is very interesting and unique", 4),
            ("기술 발전으로 인해 우리의 삶이 크게 변했습니다", "Our lives have changed significantly due to technological development", 5),
            ("환경 보호는 우리 모두의 책임입니다", "Environmental protection is everyone's responsibility", 4),
            ("교육은 개인의 미래를 밝게 만듭니다", "Education brightens an individual's future", 4),
            ("건강을 유지하는 것은 매우 중요합니다", "Maintaining health is very important", 4),
            ("친구와의 관계를 소중히 여겨야 합니다", "We should cherish relationships with friends", 4),
            ("미래를 위해 지금 무엇을 해야 할지 생각해봅시다", "Let's think about what we should do now for the future", 5),
            ("다른 문화를 이해하는 것은 중요한 능력입니다", "Understanding different cultures is an important skill", 5),
            ("노력 없이는 성공을 기대할 수 없습니다", "We cannot expect success without effort", 4),
            ("매일 조금씩 발전하는 것이 중요합니다", "It's important to improve a little bit every day", 4),
        ]
        
        for i, (korean, english, difficulty) in enumerate(advanced_cases):
            test_cases.append(TestCase(
                id=f"advanced_{i:03d}",
                korean_text=korean,
                expected_english=english,
                category="advanced",
                difficulty=difficulty
            ))
        
        # Domain-specific tests
        domain_cases = [
            # Technology
            ("인공지능 기술이 빠르게 발전하고 있습니다", "Artificial intelligence technology is developing rapidly", 4),
            ("머신러닝은 데이터로부터 패턴을 학습합니다", "Machine learning learns patterns from data", 4),
            ("딥러닝 신경망은 복잡한 문제를 해결합니다", "Deep neural networks solve complex problems", 5),
            
            # Business
            ("시장 조사는 비즈니스 전략에 중요합니다", "Market research is important for business strategy", 4),
            ("고객 만족도를 높이는 것이 우선입니다", "Increasing customer satisfaction is a priority", 4),
            ("효과적인 마케팅은 브랜드 인지도를 높입니다", "Effective marketing increases brand awareness", 4),
            
            # Healthcare
            ("정기적인 건강 검진은 질병을 예방합니다", "Regular health checkups prevent diseases", 4),
            ("균형 잡힌 식단은 건강에 필수적입니다", "A balanced diet is essential for health", 4),
            ("충분한 수면은 면역 체계를 강화합니다", "Sufficient sleep strengthens the immune system", 4),
            
            # Education
            ("평생 학습은 현대 사회에서 중요합니다", "Lifelong learning is important in modern society", 4),
            ("창의적 사고는 문제 해결에 도움이 됩니다", "Creative thinking helps in problem solving", 4),
            ("협업 능력은 직장에서 필수적입니다", "Collaboration skills are essential in the workplace", 4),
        ]
        
        for i, (korean, english, difficulty) in enumerate(domain_cases):
            domain = ["technology", "business", "healthcare", "education"][i // 3]
            test_cases.append(TestCase(
                id=f"domain_{domain}_{i%3:03d}",
                korean_text=korean,
                expected_english=english,
                category=f"domain_{domain}",
                difficulty=difficulty
            ))
        
        return test_cases
    
    def generate_test_images(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Generate test images for visual translation testing."""
        print("Generating test images...")
        
        # Create images with Korean text
        for i, case in enumerate(test_cases[:20]):  # Generate images for first 20 cases
            # Create image with Korean text
            img = Image.new('RGB', (400, 200), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a font that supports Korean (fallback to default)
            try:
                font = ImageFont.truetype("malgun.ttf", 36)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 36)
                except:
                    font = ImageFont.load_default()
            
            # Draw Korean text
            draw.text((50, 80), case.korean_text, fill='black', font=font)
            
            # Save image
            image_path = f"tests/comprehensive/data/image_{case.id}.png"
            img.save(image_path)
            
            # Update test case
            case.image_path = image_path
            
            if i % 5 == 0:
                print(f"  Generated {i+1}/{min(20, len(test_cases))} images")
        
        return test_cases
    
    def generate_test_audio(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Generate test audio files for audio translation testing."""
        print("Generating test audio files...")
        
        # Generate synthetic audio (sine wave tones representing speech patterns)
        for i, case in enumerate(test_cases[:10]):  # Generate audio for first 10 cases
            # Create synthetic audio (this is a placeholder - in real implementation,
            # you would use text-to-speech)
            sample_rate = 16000
            duration = 2.0  # 2 seconds
            
            # Generate a simple tone pattern (placeholder for speech)
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create a pattern that varies with the text length
            freq_base = 200 + len(case.korean_text) * 10
            audio = np.sin(2 * np.pi * freq_base * t)
            
            # Add some modulation
            modulation = np.sin(2 * np.pi * 3 * t)
            audio = audio * (0.8 + 0.2 * modulation)
            
            # Fade in/out
            fade_samples = int(0.1 * sample_rate)
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
            
            # Save audio file
            audio_path = f"tests/comprehensive/data/audio_{case.id}.wav"
            sf.write(audio_path, audio, sample_rate)
            
            # Update test case
            case.audio_path = audio_path
            
            if i % 3 == 0:
                print(f"  Generated {i+1}/{min(10, len(test_cases))} audio files")
        
        return test_cases
    
    def translate_text(self, korean_text: str) -> Tuple[str, float]:
        """Translate Korean text to English."""
        start_time = time.time()
        
        # Tokenize input
        src_tokens = self.tokenizer.encode(korean_text)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
        
        # Generate translation using the model's autoregressive generation
        with torch.no_grad():
            # Use the model's forward method for inference
            logits = self.model(src_tensor)
            
            # The model returns the generated sequence directly
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
        
        execution_time = time.time() - start_time
        return english_text, execution_time
    
    def evaluate_single_case(self, test_case: TestCase) -> Dict:
        """Evaluate a single test case."""
        # Text translation
        predicted_english, execution_time = self.translate_text(test_case.korean_text)
        
        # Calculate metrics
        bleu_score = self.bleu_metric.compute([predicted_english], [[test_case.expected_english]])
        
        # Calculate accuracy (exact match)
        exact_match = predicted_english.lower().strip() == test_case.expected_english.lower().strip()
        
        # Calculate semantic similarity (simple word overlap)
        pred_words = set(predicted_english.lower().split())
        expected_words = set(test_case.expected_english.lower().split())
        
        if len(expected_words) > 0:
            word_overlap = len(pred_words.intersection(expected_words)) / len(expected_words)
        else:
            word_overlap = 0.0
        
        return {
            'test_case_id': test_case.id,
            'korean_text': test_case.korean_text,
            'expected_english': test_case.expected_english,
            'predicted_english': predicted_english,
            'bleu_score': bleu_score,
            'exact_match': exact_match,
            'word_overlap': word_overlap,
            'execution_time': execution_time,
            'category': test_case.category,
            'difficulty': test_case.difficulty,
            'image_path': test_case.image_path,
            'audio_path': test_case.audio_path
        }
    
    def run_comprehensive_test(self, num_workers: int = 4) -> Dict:
        """Run the comprehensive test suite."""
        print("Starting comprehensive test suite...")
        start_time = time.time()
        
        # Generate test cases
        print("Generating test cases...")
        self.test_cases = self.generate_test_cases()
        self.test_cases = self.generate_test_images(self.test_cases)
        self.test_cases = self.generate_test_audio(self.test_cases)
        
        print(f"Total test cases: {len(self.test_cases)}")
        
        # Run tests in parallel
        print("Running translation tests...")
        results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.evaluate_single_case, case) for case in self.test_cases]
            
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{len(self.test_cases)} tests")
        
        # Calculate aggregate statistics
        total_time = time.time() - start_time
        
        # Calculate category-wise statistics
        category_stats = {}
        for category in ['basic', 'intermediate', 'advanced', 'domain_technology', 'domain_business', 'domain_healthcare', 'domain_education']:
            category_results = [r for r in results if r['category'] == category]
            if category_results:
                category_stats[category] = {
                    'count': len(category_results),
                    'avg_bleu': np.mean([r['bleu_score'] for r in category_results]),
                    'exact_match_rate': np.mean([r['exact_match'] for r in category_results]),
                    'avg_execution_time': np.mean([r['execution_time'] for r in category_results])
                }
        
        # Calculate difficulty-wise statistics
        difficulty_stats = {}
        for difficulty in range(1, 6):
            diff_results = [r for r in results if r['difficulty'] == difficulty]
            if diff_results:
                difficulty_stats[f'difficulty_{difficulty}'] = {
                    'count': len(diff_results),
                    'avg_bleu': np.mean([r['bleu_score'] for r in diff_results]),
                    'exact_match_rate': np.mean([r['exact_match'] for r in diff_results])
                }
        
        # Overall statistics
        overall_stats = {
            'total_tests': len(results),
            'overall_bleu': np.mean([r['bleu_score'] for r in results]),
            'overall_exact_match_rate': np.mean([r['exact_match'] for r in results]),
            'avg_execution_time': np.mean([r['execution_time'] for r in results]),
            'total_execution_time': total_time,
            'tests_per_second': len(results) / total_time
        }
        
        # Check if 99% target is achieved
        perfect_translation_rate = np.mean([r['exact_match'] for r in results])
        target_achieved = perfect_translation_rate >= 0.99
        
        comprehensive_results = {
            'overall_stats': overall_stats,
            'category_stats': category_stats,
            'difficulty_stats': difficulty_stats,
            'individual_results': results,
            'target_achieved': target_achieved,
            'perfect_translation_rate': perfect_translation_rate,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.results = comprehensive_results
        return comprehensive_results
    
    def generate_report(self, results: Dict, output_path: str = "tests/comprehensive/reports/comprehensive_test_report.html"):
        """Generate a comprehensive HTML report."""
        print("Generating comprehensive test report...")
        
        # Create visualizations
        self._create_visualizations(results)
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Korean-English Translation Comprehensive Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .category-stats {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
        .category-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; flex: 1; min-width: 200px; }}
        .results-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .results-table th {{ background-color: #f2f2f2; }}
        .exact-match {{ background-color: #d4edda; }}
        .no-match {{ background-color: #f8d7da; }}
        .target-achieved {{ background-color: #28a745; color: white; padding: 10px; border-radius: 5px; text-align: center; }}
        .target-not-achieved {{ background-color: #dc3545; color: white; padding: 10px; border-radius: 5px; text-align: center; }}
        .charts {{ margin: 20px 0; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Korean-English Translation Comprehensive Test Report</h1>
        <p><strong>Test Date:</strong> {results['timestamp']}</p>
        <p><strong>Model:</strong> Optimized CVM Transformer</p>
        <p><strong>Total Test Cases:</strong> {results['overall_stats']['total_tests']}</p>
    </div>
    
    <div class="summary">
        <h2>Overall Performance Summary</h2>
        <p><strong>Average BLEU Score:</strong> {results['overall_stats']['overall_bleu']:.4f}</p>
        <p><strong>Perfect Translation Rate:</strong> {results['overall_stats']['overall_exact_match_rate']:.2%}</p>
        <p><strong>Average Execution Time:</strong> {results['overall_stats']['avg_execution_time']:.4f} seconds</p>
        <p><strong>Tests per Second:</strong> {results['overall_stats']['tests_per_second']:.2f}</p>
        
        <div class="{'target-achieved' if results['target_achieved'] else 'target-not-achieved'}">
            <h3>{'🎉 TARGET ACHIEVED! 🎉' if results['target_achieved'] else 'Target Not Achieved'}</h3>
            <p>Perfect Translation Rate: {results['perfect_translation_rate']:.2%} (Target: 99%)</p>
        </div>
    </div>
    
    <div class="charts">
        <h2>Performance Visualizations</h2>
        <div class="chart">
            <img src="category_performance.png" alt="Category Performance" style="max-width: 100%; height: auto;">
        </div>
        <div class="chart">
            <img src="difficulty_analysis.png" alt="Difficulty Analysis" style="max-width: 100%; height: auto;">
        </div>
        <div class="chart">
            <img src="bleu_distribution.png" alt="BLEU Score Distribution" style="max-width: 100%; height: auto;">
        </div>
    </div>
    
    <div class="category-stats">
        <h2>Category-wise Performance</h2>
        """
        
        for category, stats in results['category_stats'].items():
            html_content += f"""
        <div class="category-card">
            <h3>{category.replace('_', ' ').title()}</h3>
            <p><strong>Test Count:</strong> {stats['count']}</p>
            <p><strong>Avg BLEU:</strong> {stats['avg_bleu']:.4f}</p>
            <p><strong>Exact Match Rate:</strong> {stats['exact_match_rate']:.2%}</p>
            <p><strong>Avg Execution Time:</strong> {stats['avg_execution_time']:.4f}s</p>
        </div>
            """
        
        html_content += """
    </div>
    
    <h2>Individual Test Results</h2>
    <table class="results-table">
        <thead>
            <tr>
                <th>ID</th>
                <th>Category</th>
                <th>Difficulty</th>
                <th>Korean Text</th>
                <th>Expected English</th>
                <th>Predicted English</th>
                <th>BLEU Score</th>
                <th>Exact Match</th>
                <th>Execution Time</th>
            </tr>
        </thead>
        <tbody>
        """
        
        for result in results['individual_results']:
            row_class = 'exact-match' if result['exact_match'] else 'no-match'
            html_content += f"""
            <tr class="{row_class}">
                <td>{result['test_case_id']}</td>
                <td>{result['category']}</td>
                <td>{result['difficulty']}</td>
                <td>{result['korean_text']}</td>
                <td>{result['expected_english']}</td>
                <td>{result['predicted_english']}</td>
                <td>{result['bleu_score']:.4f}</td>
                <td>{'✓' if result['exact_match'] else '✗'}</td>
                <td>{result['execution_time']:.4f}s</td>
            </tr>
            """
        
        html_content += """
        </tbody>
    </table>
</body>
</html>
        """
        
        # Save HTML report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comprehensive test report saved to: {output_path}")
    
    def _create_visualizations(self, results: Dict):
        """Create performance visualizations."""
        plt.style.use('seaborn-v0_8')
        
        # Category performance chart
        plt.figure(figsize=(12, 6))
        categories = list(results['category_stats'].keys())
        bleu_scores = [results['category_stats'][cat]['avg_bleu'] for cat in categories]
        exact_match_rates = [results['category_stats'][cat]['exact_match_rate'] * 100 for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, bleu_scores, width, label='BLEU Score', alpha=0.8)
        plt.bar(x + width/2, exact_match_rates, width, label='Exact Match Rate (%)', alpha=0.8)
        
        plt.xlabel('Category')
        plt.ylabel('Score')
        plt.title('Performance by Category')
        plt.xticks(x, [cat.replace('_', ' ').title() for cat in categories], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('tests/comprehensive/reports/category_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Difficulty analysis chart
        plt.figure(figsize=(10, 6))
        difficulties = list(results['difficulty_stats'].keys())
        bleu_by_diff = [results['difficulty_stats'][diff]['avg_bleu'] for diff in difficulties]
        exact_by_diff = [results['difficulty_stats'][diff]['exact_match_rate'] * 100 for diff in difficulties]
        
        x = np.arange(len(difficulties))
        
        plt.plot(x, bleu_by_diff, 'o-', label='BLEU Score', linewidth=2, markersize=8)
        plt.plot(x, exact_by_diff, 's-', label='Exact Match Rate (%)', linewidth=2, markersize=8)
        
        plt.xlabel('Difficulty Level')
        plt.ylabel('Score')
        plt.title('Performance vs Difficulty Level')
        plt.xticks(x, [f'Level {diff.split("_")[1]}' for diff in difficulties])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tests/comprehensive/reports/difficulty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # BLEU score distribution histogram
        plt.figure(figsize=(10, 6))
        bleu_scores = [r['bleu_score'] for r in results['individual_results']]
        
        plt.hist(bleu_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(results['overall_stats']['overall_bleu'], color='red', linestyle='--', 
                   label=f'Mean BLEU: {results["overall_stats"]["overall_bleu"]:.4f}')
        plt.axvline(0.99, color='green', linestyle='--', label='Target: 0.99')
        
        plt.xlabel('BLEU Score')
        plt.ylabel('Frequency')
        plt.title('BLEU Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tests/comprehensive/reports/bleu_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations created and saved")

def main():
    """Main function to run the comprehensive test suite."""
    print("=== Korean-English Translation Comprehensive Test Suite ===")
    
    # Initialize test suite
    model_path = "models/production/optimized_model.pth"
    tokenizer_path = "data/tokenizers/kr_en_diverse.model"
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run the optimized training first.")
        return
    
    if not Path(tokenizer_path).exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Please ensure the tokenizer is trained.")
        return
    
    test_suite = ComprehensiveTestSuite(model_path, tokenizer_path)
    
    # Run comprehensive tests
    results = test_suite.run_comprehensive_test(num_workers=4)
    
    # Generate report
    test_suite.generate_report(results)
    
    # Save detailed results (convert numpy types to Python types)
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    with open('tests/comprehensive/results/detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(results), f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {results['overall_stats']['total_tests']}")
    print(f"Average BLEU Score: {results['overall_stats']['overall_bleu']:.4f}")
    print(f"Perfect Translation Rate: {results['overall_stats']['overall_exact_match_rate']:.2%}")
    print(f"Target (99%) Achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")
    print(f"Average Execution Time: {results['overall_stats']['avg_execution_time']:.4f} seconds")
    print(f"Tests per Second: {results['overall_stats']['tests_per_second']:.2f}")
    print("="*60)
    
    if results['target_achieved']:
        print("🎉 CONGRATULATIONS! 99% PERFECT TRANSLATION TARGET ACHIEVED! 🎉")
    else:
        improvement_needed = (0.99 - results['overall_stats']['overall_exact_match_rate']) / results['overall_stats']['overall_exact_match_rate'] * 100
        print(f"📈 Need {improvement_needed:.1f}% improvement to reach 99% target")
    
    print(f"\n📊 Detailed report saved to: tests/comprehensive/reports/comprehensive_test_report.html")
    print(f"📋 Detailed results saved to: tests/comprehensive/results/detailed_results.json")

if __name__ == "__main__":
    main()