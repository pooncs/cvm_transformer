import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time
from pathlib import Path
import logging
from dataclasses import dataclass
from tqdm import tqdm

# Import our models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nmt_transformer import NMTTransformer
from src.models.image_encoder import EnhancedMultimodalNMT, KoreanTextImageEncoder
from src.models.audio_encoder import MultimodalAudioNMT, KoreanSpeechEncoder
from src.data.prepare_corpus import ParallelCorpusProcessor
from src.training.train_nmt import NMTTrainer
from src.utils.metrics import BLEUScore, ExactMatchScore
from src.models.sp_tokenizer import SPTokenizer as SentencePieceTokenizer


@dataclass
class TestResult:
    """Result of a single test case."""
    input_text: str
    input_image: Optional[np.ndarray]
    input_audio: Optional[np.ndarray]
    expected_translation: str
    predicted_translation: str
    bleu_score: float
    exact_match: bool
    execution_time: float
    modality: str  # 'text', 'image', 'audio', 'multimodal'


@dataclass
class ValidationReport:
    """Complete validation report."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_bleu: float
    perfect_translation_rate: float
    average_execution_time: float
    modality_breakdown: Dict[str, Dict[str, float]]
    detailed_results: List[TestResult]
    timestamp: str


class MultimodalValidator:
    """
    Comprehensive validator for multimodal Korean-English translation.
    Tests text, image, audio, and multimodal inputs.
    """
    
    def __init__(self,
                 text_model_path: str,
                 image_model_path: Optional[str] = None,
                 audio_model_path: Optional[str] = None,
                 tokenizer_path: str = "models/tokenizers/korean_english_spm",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Load tokenizer
        self.tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
        
        # Load models
        self.text_model = self._load_text_model(text_model_path)
        self.image_model = self._load_image_model(image_model_path) if image_model_path else None
        self.audio_model = self._load_audio_model(audio_model_path) if audio_model_path else None
        
        # Metrics
        self.bleu_metric = BLEUScore()
        self.exact_match_metric = ExactMatchScore()
        
    def _load_text_model(self, model_path: str) -> NMTTransformer:
        """Load the text NMT model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = NMTTransformer(
            src_vocab_size=checkpoint['config']['src_vocab_size'],
            tgt_vocab_size=checkpoint['config']['tgt_vocab_size'],
            d_model=checkpoint['config']['d_model'],
            n_heads=checkpoint['config']['n_heads'],
            n_encoder_layers=checkpoint['config']['n_encoder_layers'],
            n_decoder_layers=checkpoint['config']['n_decoder_layers'],
            d_ff=checkpoint['config']['d_ff'],
            max_len=checkpoint['config']['max_len'],
            dropout=checkpoint['config']['dropout'],
            pad_id=checkpoint['config']['pad_id'],
            use_flash=checkpoint['config']['use_flash']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
        
    def _load_image_model(self, model_path: str) -> EnhancedMultimodalNMT:
        """Load the image-based NMT model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = EnhancedMultimodalNMT(
            src_vocab_size=checkpoint['config']['src_vocab_size'],
            tgt_vocab_size=checkpoint['config']['tgt_vocab_size'],
            d_model=checkpoint['config']['d_model'],
            n_heads=checkpoint['config']['n_heads'],
            n_encoder_layers=checkpoint['config']['n_encoder_layers'],
            n_decoder_layers=checkpoint['config']['n_decoder_layers'],
            d_ff=checkpoint['config']['d_ff'],
            max_len=checkpoint['config']['max_len'],
            dropout=checkpoint['config']['dropout'],
            pad_id=checkpoint['config']['pad_id'],
            use_flash=checkpoint['config']['use_flash']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
        
    def _load_audio_model(self, model_path: str) -> MultimodalAudioNMT:
        """Load the audio-based NMT model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = MultimodalAudioNMT(
            src_vocab_size=checkpoint['config']['src_vocab_size'],
            tgt_vocab_size=checkpoint['config']['tgt_vocab_size'],
            d_model=checkpoint['config']['d_model'],
            n_heads=checkpoint['config']['n_heads'],
            n_encoder_layers=checkpoint['config']['n_encoder_layers'],
            n_decoder_layers=checkpoint['config']['n_decoder_layers'],
            d_ff=checkpoint['config']['d_ff'],
            max_len=checkpoint['config']['max_len'],
            dropout=checkpoint['config']['dropout'],
            pad_id=checkpoint['config']['pad_id'],
            use_flash=checkpoint['config']['use_flash']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
        
    def create_test_image(self, text: str, image_size: int = 224) -> np.ndarray:
        """Create synthetic Korean text image for testing."""
        # This would use PIL/OpenCV to create actual images
        # For now, create random tensor as placeholder
        return np.random.randn(3, image_size, image_size).astype(np.float32)
        
    def create_test_audio(self, text: str, duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
        """Create synthetic Korean speech audio for testing."""
        # This would use TTS to create actual audio
        # For now, create random tensor as placeholder
        audio_length = int(duration * sample_rate)
        return np.random.randn(audio_length).astype(np.float32)
        
    def test_text_translation(self, test_cases: List[Dict]) -> List[TestResult]:
        """Test text-only translation."""
        results = []
        
        for case in tqdm(test_cases, desc="Text Translation Tests"):
            start_time = time.time()
            
            # Tokenize input
            src_tokens = self.tokenizer.encode(case['korean'])
            src_tensor = torch.tensor([src_tokens]).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                pred_tokens = self.text_model.generate(
                    src_tensor,
                    max_length=512,
                    beam_size=5,
                    temperature=1.0
                )
                
            # Decode prediction
            predicted = self.tokenizer.decode(pred_tokens[0].cpu().tolist())
            
            # Calculate metrics
            bleu_score = self.bleu_metric([predicted], [case['english']])
            exact_match = self.exact_match_metric(predicted, case['english'])
            
            execution_time = time.time() - start_time
            
            result = TestResult(
                input_text=case['korean'],
                input_image=None,
                input_audio=None,
                expected_translation=case['english'],
                predicted_translation=predicted,
                bleu_score=bleu_score,
                exact_match=exact_match,
                execution_time=execution_time,
                modality='text'
            )
            
            results.append(result)
            
        return results
        
    def test_image_translation(self, test_cases: List[Dict]) -> List[TestResult]:
        """Test image-based translation (Korean text in images)."""
        if not self.image_model:
            self.logger.warning("Image model not available, skipping image tests")
            return []
            
        results = []
        
        for case in tqdm(test_cases, desc="Image Translation Tests"):
            start_time = time.time()
            
            # Create test image
            test_image = self.create_test_image(case['korean'])
            image_tensor = torch.tensor(test_image).unsqueeze(0).to(self.device)
            
            # Create dummy text input (for batch compatibility)
            dummy_tokens = [self.tokenizer.pad_id] * 10
            src_tensor = torch.tensor([dummy_tokens]).to(self.device)
            
            # Generate translation
            self.image_model.set_mode('image')
            with torch.no_grad():
                pred_tokens = self.image_model.generate(
                    src_tensor,
                    src_images=image_tensor,
                    max_length=512,
                    beam_size=5,
                    temperature=1.0
                )
                
            # Decode prediction
            predicted = self.tokenizer.decode(pred_tokens[0].cpu().tolist())
            
            # Calculate metrics
            bleu_score = self.bleu_metric([predicted], [case['english']])
            exact_match = self.exact_match_metric(predicted, case['english'])
            
            execution_time = time.time() - start_time
            
            result = TestResult(
                input_text=case['korean'],
                input_image=test_image,
                input_audio=None,
                expected_translation=case['english'],
                predicted_translation=predicted,
                bleu_score=bleu_score,
                exact_match=exact_match,
                execution_time=execution_time,
                modality='image'
            )
            
            results.append(result)
            
        return results
        
    def test_audio_translation(self, test_cases: List[Dict]) -> List[TestResult]:
        """Test audio-based translation (Korean speech)."""
        if not self.audio_model:
            self.logger.warning("Audio model not available, skipping audio tests")
            return []
            
        results = []
        
        for case in tqdm(test_cases, desc="Audio Translation Tests"):
            start_time = time.time()
            
            # Create test audio
            test_audio = self.create_test_audio(case['korean'])
            audio_tensor = torch.tensor(test_audio).unsqueeze(0).to(self.device)
            
            # Create dummy text input (for batch compatibility)
            dummy_tokens = [self.tokenizer.pad_id] * 10
            src_tensor = torch.tensor([dummy_tokens]).to(self.device)
            
            # Generate translation
            self.audio_model.set_mode('audio')
            with torch.no_grad():
                pred_tokens = self.audio_model.generate(
                    src_tensor,
                    src_audio=audio_tensor,
                    max_length=512,
                    beam_size=5,
                    temperature=1.0
                )
                
            # Decode prediction
            predicted = self.tokenizer.decode(pred_tokens[0].cpu().tolist())
            
            # Calculate metrics
            bleu_score = self.bleu_metric([predicted], [case['english']])
            exact_match = self.exact_match_metric(predicted, case['english'])
            
            execution_time = time.time() - start_time
            
            result = TestResult(
                input_text=case['korean'],
                input_image=None,
                input_audio=test_audio,
                expected_translation=case['english'],
                predicted_translation=predicted,
                bleu_score=bleu_score,
                exact_match=exact_match,
                execution_time=execution_time,
                modality='audio'
            )
            
            results.append(result)
            
        return results
        
    def test_multimodal_translation(self, test_cases: List[Dict]) -> List[TestResult]:
        """Test multimodal translation (text + image/audio)."""
        results = []
        
        # Test text + image
        if self.image_model:
            for case in tqdm(test_cases[:len(test_cases)//2], desc="Multimodal Text+Image Tests"):
                start_time = time.time()
                
                # Create test image and tokenize text
                test_image = self.create_test_image(case['korean'])
                image_tensor = torch.tensor(test_image).unsqueeze(0).to(self.device)
                
                src_tokens = self.tokenizer.encode(case['korean'])
                src_tensor = torch.tensor([src_tokens]).to(self.device)
                
                # Generate translation
                self.image_model.set_mode('multimodal')
                with torch.no_grad():
                    pred_tokens = self.image_model.generate(
                        src_tensor,
                        src_images=image_tensor,
                        max_length=512,
                        beam_size=5,
                        temperature=1.0
                    )
                    
                # Decode prediction
                predicted = self.tokenizer.decode(pred_tokens[0].cpu().tolist())
                
                # Calculate metrics
                bleu_score = self.bleu_metric([predicted], [case['english']])
                exact_match = self.exact_match_metric(predicted, case['english'])
                
                execution_time = time.time() - start_time
                
                result = TestResult(
                    input_text=case['korean'],
                    input_image=test_image,
                    input_audio=None,
                    expected_translation=case['english'],
                    predicted_translation=predicted,
                    bleu_score=bleu_score,
                    exact_match=exact_match,
                    execution_time=execution_time,
                    modality='multimodal_text_image'
                )
                
                results.append(result)
                
        # Test text + audio
        if self.audio_model:
            for case in tqdm(test_cases[len(test_cases)//2:], desc="Multimodal Text+Audio Tests"):
                start_time = time.time()
                
                # Create test audio and tokenize text
                test_audio = self.create_test_audio(case['korean'])
                audio_tensor = torch.tensor(test_audio).unsqueeze(0).to(self.device)
                
                src_tokens = self.tokenizer.encode(case['korean'])
                src_tensor = torch.tensor([src_tokens]).to(self.device)
                
                # Generate translation
                self.audio_model.set_mode('multimodal')
                with torch.no_grad():
                    pred_tokens = self.audio_model.generate(
                        src_tensor,
                        src_audio=audio_tensor,
                        max_length=512,
                        beam_size=5,
                        temperature=1.0
                    )
                    
                # Decode prediction
                predicted = self.tokenizer.decode(pred_tokens[0].cpu().tolist())
                
                # Calculate metrics
                bleu_score = self.bleu_metric([predicted], [case['english']])
                exact_match = self.exact_match_metric(predicted, case['english'])
                
                execution_time = time.time() - start_time
                
                result = TestResult(
                    input_text=case['korean'],
                    input_image=None,
                    input_audio=test_audio,
                    expected_translation=case['english'],
                    predicted_translation=predicted,
                    bleu_score=bleu_score,
                    exact_match=exact_match,
                    execution_time=execution_time,
                    modality='multimodal_text_audio'
                )
                
                results.append(result)
                
        return results
        
    def run_comprehensive_validation(self, test_cases: List[Dict]) -> ValidationReport:
        """Run comprehensive validation across all modalities."""
        self.logger.info("Starting comprehensive multimodal validation...")
        
        all_results = []
        
        # Run tests for each modality
        text_results = self.test_text_translation(test_cases)
        image_results = self.test_image_translation(test_cases)
        audio_results = self.test_audio_translation(test_cases)
        multimodal_results = self.test_multimodal_translation(test_cases)
        
        # Combine all results
        all_results.extend(text_results)
        all_results.extend(image_results)
        all_results.extend(audio_results)
        all_results.extend(multimodal_results)
        
        # Calculate statistics
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.exact_match)
        failed_tests = total_tests - passed_tests
        average_bleu = np.mean([r.bleu_score for r in all_results])
        perfect_translation_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        average_execution_time = np.mean([r.execution_time for r in all_results])
        
        # Modality breakdown
        modality_breakdown = {}
        for modality in set(r.modality for r in all_results):
            modality_results = [r for r in all_results if r.modality == modality]
            modality_breakdown[modality] = {
                'total_tests': len(modality_results),
                'passed_tests': sum(1 for r in modality_results if r.exact_match),
                'average_bleu': np.mean([r.bleu_score for r in modality_results]),
                'perfect_rate': (sum(1 for r in modality_results if r.exact_match) / len(modality_results)) * 100 if modality_results else 0,
                'average_time': np.mean([r.execution_time for r in modality_results])
            }
        
        # Create report
        report = ValidationReport(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            average_bleu=average_bleu,
            perfect_translation_rate=perfect_translation_rate,
            average_execution_time=average_execution_time,
            modality_breakdown=modality_breakdown,
            detailed_results=all_results,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return report
        
    def save_report(self, report: ValidationReport, output_path: str):
        """Save validation report to JSON and HTML."""
        # Save JSON report
        json_path = Path(output_path) / "multimodal_validation_report.json"
        
        # Convert to serializable format
        report_dict = {
            'total_tests': report.total_tests,
            'passed_tests': report.passed_tests,
            'failed_tests': report.failed_tests,
            'average_bleu': report.average_bleu,
            'perfect_translation_rate': report.perfect_translation_rate,
            'average_execution_time': report.average_execution_time,
            'modality_breakdown': report.modality_breakdown,
            'timestamp': report.timestamp,
            'detailed_results': [
                {
                    'input_text': r.input_text,
                    'expected_translation': r.expected_translation,
                    'predicted_translation': r.predicted_translation,
                    'bleu_score': r.bleu_score,
                    'exact_match': r.exact_match,
                    'execution_time': r.execution_time,
                    'modality': r.modality
                }
                for r in report.detailed_results
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
            
        # Save HTML report
        html_path = Path(output_path) / "multimodal_validation_report.html"
        html_content = self._generate_html_report(report)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"Validation reports saved to {json_path} and {html_path}")
        
    def _generate_html_report(self, report: ValidationReport) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multimodal Korean-English Translation Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .modality-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .test-case {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Multimodal Korean-English Translation Validation Report</h1>
                <p>Generated: {report.timestamp}</p>
            </div>
            
            <div class="summary">
                <h2>Overall Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Tests</td><td>{report.total_tests}</td></tr>
                    <tr><td>Passed Tests</td><td>{report.passed_tests}</td></tr>
                    <tr><td>Failed Tests</td><td>{report.failed_tests}</td></tr>
                    <tr><td>Average BLEU Score</td><td>{report.average_bleu:.4f}</td></tr>
                    <tr><td>Perfect Translation Rate</td><td>{report.perfect_translation_rate:.2f}%</td></tr>
                    <tr><td>Average Execution Time</td><td>{report.average_execution_time:.4f}s</td></tr>
                </table>
            </div>
            
            <div class="modality-section">
                <h2>Modality Breakdown</h2>
                {self._generate_modality_tables(report.modality_breakdown)}
            </div>
            
            <div class="modality-section">
                <h2>Detailed Test Results</h2>
                {self._generate_detailed_results(report.detailed_results)}
            </div>
        </body>
        </html>
        """
        
        return html
        
    def _generate_modality_tables(self, modality_breakdown: Dict) -> str:
        """Generate HTML tables for modality breakdown."""
        html = ""
        
        for modality, stats in modality_breakdown.items():
            html += f"""
            <h3>{modality.replace('_', ' ').title()}</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Tests</td><td>{stats['total_tests']}</td></tr>
                <tr><td>Passed Tests</td><td>{stats['passed_tests']}</td></tr>
                <tr><td>Average BLEU Score</td><td>{stats['average_bleu']:.4f}</td></tr>
                <tr><td>Perfect Translation Rate</td><td>{stats['perfect_rate']:.2f}%</td></tr>
                <tr><td>Average Execution Time</td><td>{stats['average_time']:.4f}s</td></tr>
            </table>
            """
            
        return html
        
    def _generate_detailed_results(self, results: List[TestResult]) -> str:
        """Generate HTML for detailed results."""
        html = ""
        
        for i, result in enumerate(results[:50]):  # Show first 50 results
            status_class = "pass" if result.exact_match else "fail"
            status_text = "✓ PASS" if result.exact_match else "✗ FAIL"
            
            html += f"""
            <div class="test-case">
                <h4>Test {i+1}: {result.modality.replace('_', ' ').title()} - <span class="{status_class}">{status_text}</span></h4>
                <p><strong>Korean Input:</strong> {result.input_text}</p>
                <p><strong>Expected English:</strong> {result.expected_translation}</p>
                <p><strong>Predicted English:</strong> {result.predicted_translation}</p>
                <p><strong>BLEU Score:</strong> {result.bleu_score:.4f} | 
                   <strong>Execution Time:</strong> {result.execution_time:.4f}s</p>
            </div>
            """
            
        return html


def create_comprehensive_test_suite() -> List[Dict]:
    """Create comprehensive Korean-English test cases."""
    test_cases = [
        # Basic greetings and common phrases
        {"korean": "안녕하세요", "english": "Hello"},
        {"korean": "감사합니다", "english": "Thank you"},
        {"korean": "죄송합니다", "english": "Sorry"},
        {"korean": "네", "english": "Yes"},
        {"korean": "아니요", "english": "No"},
        
        # Daily conversations
        {"korean": "오늘 날씨가 좋네요", "english": "The weather is nice today"},
        {"korean": "밥 먹었어요?", "english": "Did you eat?"},
        {"korean": "어디 가세요?", "english": "Where are you going?"},
        {"korean": "잘 지내셨어요?", "english": "Have you been well?"},
        
        # Complex sentences
        {"korean": "저는 한국어를 배우고 있어요", "english": "I am learning Korean"},
        {"korean": "이 책은 정말 흥미로워요", "english": "This book is really interesting"},
        {"korean": "내일 학교에 가야 해요", "english": "I have to go to school tomorrow"},
        {"korean": "커피 마시고 싶어요", "english": "I want to drink coffee"},
        
        # Technical/business Korean
        {"korean": "회의는 오후 3시에 시작됩니다", "english": "The meeting starts at 3 PM"},
        {"korean": "프로젝트 일정을 확인해 주세요", "english": "Please check the project schedule"},
        {"korean": "보고서를 제출해야 합니다", "english": "I need to submit the report"},
        
        # Cultural expressions
        {"korean": "많이 드세요", "english": "Please eat a lot"},
        {"korean": "수고하셨습니다", "english": "Thank you for your hard work"},
        {"korean": "들어오세요", "english": "Please come in"},
        
        # Question forms
        {"korean": "이것은 무엇입니까?", "english": "What is this?"},
        {"korean": "언제 도착했어요?", "english": "When did you arrive?"},
        {"korean": "어떻게 가요?", "english": "How do I get there?"},
        {"korean": "누구세요?", "english": "Who are you?"},
        
        # Past/present/future tenses
        {"korean": "어제 영화를 봤어요", "english": "I watched a movie yesterday"},
        {"korean": "지금 공부하고 있어요", "english": "I am studying now"},
        {"korean": "내일 친구를 만날 거예요", "english": "I will meet my friend tomorrow"},
        
        # Honorifics and politeness levels
        {"korean": "선생님, 질문이 있어요", "english": "Teacher, I have a question"},
        {"korean": "부모님께 감사드립니다", "english": "I thank my parents"},
        {"korean": "할아버지께 안부를 전해 주세요", "english": "Please give my regards to grandfather"},
        
        # Numbers and quantities
        {"korean": "사과 두 개 주세요", "english": "Please give me two apples"},
        {"korean": "시간이 얼마나 걸려요?", "english": "How long does it take?"},
        {"korean": "가격이 얼마예요?", "english": "How much is the price?"},
        
        # Emotions and feelings
        {"korean": "정말 기뻐요", "english": "I am really happy"},
        {"korean": "조금 걱정돼요", "english": "I am a little worried"},
        {"korean": "너무 피곤해요", "english": "I am very tired"},
        {"korean": "정말 놀랐어요", "english": "I was really surprised"},
        
        # Directions and locations
        {"korean": "화장실은 어디 있어요?", "english": "Where is the bathroom?"},
        {"korean": "여기서 얼마나 멀어요?", "english": "How far is it from here?"},
        {"korean": "왼쪽으로 가세요", "english": "Go to the left"},
        {"korean": "직진하세요", "english": "Go straight"},
        
        # Shopping and restaurants
        {"korean": "이거 주문할게요", "english": "I will order this"},
        {"korean": "계산해 주세요", "english": "Please calculate the bill"},
        {"korean": "영수증 주세요", "english": "Please give me a receipt"},
        {"korean": "포장해 주세요", "english": "Please wrap it up"},
        
        # Transportation
        {"korean": "버스를 타야 해요", "english": "I need to take the bus"},
        {"korean": "지하철역은 어디에 있어요?", "english": "Where is the subway station?"},
        {"korean": "표 한 장 주세요", "english": "Please give me one ticket"},
        
        # Health and medical
        {"korean": "아파요", "english": "I am sick"},
        {"korean": "약이 필요해요", "english": "I need medicine"},
        {"korean": "병원에 가야 해요", "english": "I need to go to the hospital"},
        
        # Weather and seasons
        {"korean": "오늘 비가 올 거예요", "english": "It will rain today"},
        {"korean": "너무 추워요", "english": "It is very cold"},
        {"korean": "봄이 왔어요", "english": "Spring has come"},
        
        # Family and relationships
        {"korean": "가족이 몇 명이에요?", "english": "How many family members do you have?"},
        {"korean": "형제가 있어요?", "english": "Do you have siblings?"},
        {"korean": "결혼했어요?", "english": "Are you married?"},
        
        # Work and study
        {"korean": "무슨 일을 하세요?", "english": "What do you do for work?"},
        {"korean": "어디서 일하세요?", "english": "Where do you work?"},
        {"korean": "한국어를 얼마나 공부했어요?", "english": "How long have you studied Korean?"},
        
        # Hobbies and interests
        {"korean": "취미가 뭐예요?", "english": "What is your hobby?"},
        {"korean": "음악을 좋아해요", "english": "I like music"},
        {"korean": "운동을 자주 해요", "english": "I exercise often"},
        
        # Time and dates
        {"korean": "지금 몇 시예요?", "english": "What time is it now?"},
        {"korean": "오늘은 며칠이에요?", "english": "What is today's date?"},
        {"korean": "생일이 언제예요?", "english": "When is your birthday?"},
        
        # Descriptions and comparisons
        {"korean": "이것보다 저것이 더 좋아요", "english": "That is better than this"},
        {"korean": "가장 좋아하는 것은 뭐예요?", "english": "What do you like the most?"},
        {"korean": "이게 더 싸요", "english": "This is cheaper"}
    ]
    
    return test_cases


def main():
    """Main validation function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test cases
    print("Creating comprehensive test suite...")
    test_cases = create_comprehensive_test_suite()
    print(f"Created {len(test_cases)} test cases")
    
    # Initialize validator
    print("Initializing multimodal validator...")
    validator = MultimodalValidator(
        text_model_path="models/checkpoints/nmt_transformer_best.pt",
        image_model_path="models/checkpoints/multimodal_image_best.pt",
        audio_model_path="models/checkpoints/multimodal_audio_best.pt",
        tokenizer_path="models/tokenizers/korean_english_spm",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run validation
    print("Running comprehensive validation...")
    report = validator.run_comprehensive_validation(test_cases)
    
    # Print summary
    print("\n" + "="*60)
    print("MULTIMODAL VALIDATION SUMMARY")
    print("="*60)
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed Tests: {report.passed_tests}")
    print(f"Failed Tests: {report.failed_tests}")
    print(f"Average BLEU Score: {report.average_bleu:.4f}")
    print(f"Perfect Translation Rate: {report.perfect_translation_rate:.2f}%")
    print(f"Average Execution Time: {report.average_execution_time:.4f}s")
    print("\nModality Breakdown:")
    
    for modality, stats in report.modality_breakdown.items():
        print(f"  {modality}:")
        print(f"    Tests: {stats['total_tests']}, Passed: {stats['passed_tests']}")
        print(f"    Avg BLEU: {stats['average_bleu']:.4f}, Perfect Rate: {stats['perfect_rate']:.2f}%")
        
    print("\n" + "="*60)
    
    # Save report
    output_dir = Path("tests/multimodal_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    validator.save_report(report, str(output_dir))
    
    # Check if target is achieved
    if report.perfect_translation_rate >= 99.0:
        print("🎉 TARGET ACHIEVED! 99% perfect translation rate reached!")
    else:
        improvement_needed = 99.0 - report.perfect_translation_rate
        print(f"📈 Need {improvement_needed:.2f}% improvement to reach 99% target")
        
    return report


if __name__ == "__main__":
    main()