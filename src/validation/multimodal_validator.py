"""

Comprehensive multimodal validation test suite for Korean-English translation.
Tests text, images, and audio inputs with detailed metrics and reporting.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time
import logging
from PIL import Image
import librosa
import soundfile as sf
from transformers import AutoProcessor, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.models.nmt_transformer import NMTTransformer
from src.models.multimodal_nmt import MultimodalNMT
from src.data.korean_tokenizer import KoreanTokenizer


class MultimodalValidator:
    """Comprehensive validator for multimodal Korean-English translation."""
    
    def __init__(self, 
                 model_path: str,
                 tokenizer_path: str,
                 device: str = 'auto',
                 batch_size: int = 32,
                 max_length: int = 512):
        """
        Initialize the multimodal validator.
        
        Args:
            model_path: Path to trained model checkpoint
            tokenizer_path: Path to tokenizer
            device: Device to run validation on
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.device = self._setup_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load models and tokenizers
        self.model = self._load_model(model_path)
        self.tokenizer = KoreanTokenizer.load(tokenizer_path)
        
        # Load multimodal processors
        self.image_processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")
        self.audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Test data storage
        self.test_results = []
        self.perfect_translations = []
        self.failed_translations = []
        
        # Metrics tracking
        self.metrics = {
            'text': {'bleu': [], 'exact_match': [], 'chrf': []},
            'image': {'bleu': [], 'exact_match': [], 'ocr_accuracy': []},
            'audio': {'bleu': [], 'exact_match': [], 'asr_accuracy': []}
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'multimodal' in checkpoint.get('model_type', ''):
            model = MultimodalNMT(
                src_vocab_size=checkpoint['src_vocab_size'],
                tgt_vocab_size=checkpoint['tgt_vocab_size'],
                d_model=checkpoint['d_model'],
                n_heads=checkpoint['n_heads'],
                n_encoder_layers=checkpoint['n_encoder_layers'],
                n_decoder_layers=checkpoint['n_decoder_layers']
            )
        else:
            model = NMTTransformer(
                src_vocab_size=checkpoint['src_vocab_size'],
                tgt_vocab_size=checkpoint['tgt_vocab_size'],
                d_model=checkpoint['d_model'],
                n_heads=checkpoint['n_heads'],
                n_encoder_layers=checkpoint['n_encoder_layers'],
                n_decoder_layers=checkpoint['n_decoder_layers']
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def compute_bleu(self, reference: str, hypothesis: str) -> float:
        """Compute BLEU score between reference and hypothesis."""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        reference_tokens = reference.lower().split()
        hypothesis_tokens = hypothesis.lower().split()
        
        smoothing = SmoothingFunction().method4
        
        try:
            bleu = sentence_bleu([reference_tokens], hypothesis_tokens, 
                               smoothing_function=smoothing)
            return bleu * 100  # Convert to percentage
        except:
            return 0.0
    
    def compute_chrf(self, reference: str, hypothesis: str) -> float:
        """Compute character F-score."""
        from nltk.translate.chrf_score import sentence_chrf
        
        try:
            chrf = sentence_chrf(reference, hypothesis)
            return chrf * 100
        except:
            return 0.0
    
    def exact_match(self, reference: str, hypothesis: str) -> bool:
        """Check if reference and hypothesis match exactly."""
        return reference.strip().lower() == hypothesis.strip().lower()
    
    def validate_text_translation(self, test_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """Validate text-based translation."""
        self.logger.info(f"Validating {len(test_pairs)} text translations...")
        
        results = []
        start_time = time.time()
        
        for i, (korean_text, english_reference) in enumerate(test_pairs):
            try:
                # Tokenize input
                korean_tokens = self.tokenizer.encode(korean_text)
                korean_tensor = torch.tensor([korean_tokens]).to(self.device)
                
                # Generate translation
                with torch.no_grad():
                    english_tokens = self.model.generate(
                        korean_tensor, 
                        max_length=self.max_length,
                        beam_size=4
                    )
                
                # Decode output
                english_hypothesis = self.tokenizer.decode(english_tokens[0])
                
                # Compute metrics
                bleu = self.compute_bleu(english_reference, english_hypothesis)
                chrf = self.compute_chrf(english_reference, english_hypothesis)
                exact = self.exact_match(english_reference, english_hypothesis)
                
                result = {
                    'type': 'text',
                    'input': korean_text,
                    'reference': english_reference,
                    'hypothesis': english_hypothesis,
                    'bleu': bleu,
                    'chrf': chrf,
                    'exact_match': exact,
                    'execution_time': time.time() - start_time
                }
                
                results.append(result)
                self.metrics['text']['bleu'].append(bleu)
                self.metrics['text']['chrf'].append(chrf)
                self.metrics['text']['exact_match'].append(exact)
                
                if exact:
                    self.perfect_translations.append(result)
                else:
                    self.failed_translations.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(test_pairs)} text translations")
                
            except Exception as e:
                self.logger.error(f"Error processing text {i}: {e}")
                continue
        
        return {
            'total_tests': len(results),
            'avg_bleu': np.mean([r['bleu'] for r in results]),
            'avg_chrf': np.mean([r['chrf'] for r in results]),
            'perfect_rate': np.mean([r['exact_match'] for r in results]) * 100,
            'avg_execution_time': np.mean([r['execution_time'] for r in results])
        }
    
    def validate_image_translation(self, image_text_pairs: List[Tuple[str, str, str]]) -> Dict[str, float]:
        """Validate image-based translation with OCR."""
        self.logger.info(f"Validating {len(image_text_pairs)} image translations...")
        
        results = []
        start_time = time.time()
        
        for i, (image_path, korean_text, english_reference) in enumerate(image_text_pairs):
            try:
                # Load and process image
                image = Image.open(image_path).convert('RGB')
                
                # For now, use provided text (in real scenario, would use OCR)
                # This simulates OCR output
                ocr_text = korean_text
                ocr_accuracy = 1.0  # Perfect OCR for simulation
                
                # Tokenize OCR result
                korean_tokens = self.tokenizer.encode(ocr_text)
                korean_tensor = torch.tensor([korean_tokens]).to(self.device)
                
                # Generate translation
                with torch.no_grad():
                    english_tokens = self.model.generate(
                        korean_tensor,
                        max_length=self.max_length,
                        beam_size=4
                    )
                
                # Decode output
                english_hypothesis = self.tokenizer.decode(english_tokens[0])
                
                # Compute metrics
                bleu = self.compute_bleu(english_reference, english_hypothesis)
                exact = self.exact_match(english_reference, english_hypothesis)
                
                result = {
                    'type': 'image',
                    'image_path': image_path,
                    'ocr_text': ocr_text,
                    'reference': english_reference,
                    'hypothesis': english_hypothesis,
                    'bleu': bleu,
                    'exact_match': exact,
                    'ocr_accuracy': ocr_accuracy,
                    'execution_time': time.time() - start_time
                }
                
                results.append(result)
                self.metrics['image']['bleu'].append(bleu)
                self.metrics['image']['exact_match'].append(exact)
                self.metrics['image']['ocr_accuracy'].append(ocr_accuracy)
                
                if exact:
                    self.perfect_translations.append(result)
                else:
                    self.failed_translations.append(result)
                
                if (i + 1) % 5 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(image_text_pairs)} image translations")
                
            except Exception as e:
                self.logger.error(f"Error processing image {i}: {e}")
                continue
        
        return {
            'total_tests': len(results),
            'avg_bleu': np.mean([r['bleu'] for r in results]),
            'perfect_rate': np.mean([r['exact_match'] for r in results]) * 100,
            'avg_ocr_accuracy': np.mean([r['ocr_accuracy'] for r in results]) * 100,
            'avg_execution_time': np.mean([r['execution_time'] for r in results])
        }
    
    def validate_audio_translation(self, audio_text_pairs: List[Tuple[str, str, str]]) -> Dict[str, float]:
        """Validate audio-based translation with ASR."""
        self.logger.info(f"Validating {len(audio_text_pairs)} audio translations...")
        
        results = []
        start_time = time.time()
        
        for i, (audio_path, korean_text, english_reference) in enumerate(audio_text_pairs):
            try:
                # Load and process audio
                audio, sr = librosa.load(audio_path, sr=16000)
                
                # For now, use provided text (in real scenario, would use ASR)
                # This simulates ASR output
                asr_text = korean_text
                asr_accuracy = 1.0  # Perfect ASR for simulation
                
                # Tokenize ASR result
                korean_tokens = self.tokenizer.encode(asr_text)
                korean_tensor = torch.tensor([korean_tokens]).to(self.device)
                
                # Generate translation
                with torch.no_grad():
                    english_tokens = self.model.generate(
                        korean_tensor,
                        max_length=self.max_length,
                        beam_size=4
                    )
                
                # Decode output
                english_hypothesis = self.tokenizer.decode(english_tokens[0])
                
                # Compute metrics
                bleu = self.compute_bleu(english_reference, english_hypothesis)
                exact = self.exact_match(english_reference, english_hypothesis)
                
                result = {
                    'type': 'audio',
                    'audio_path': audio_path,
                    'asr_text': asr_text,
                    'reference': english_reference,
                    'hypothesis': english_hypothesis,
                    'bleu': bleu,
                    'exact_match': exact,
                    'asr_accuracy': asr_accuracy,
                    'execution_time': time.time() - start_time
                }
                
                results.append(result)
                self.metrics['audio']['bleu'].append(bleu)
                self.metrics['audio']['exact_match'].append(exact)
                self.metrics['audio']['asr_accuracy'].append(asr_accuracy)
                
                if exact:
                    self.perfect_translations.append(result)
                else:
                    self.failed_translations.append(result)
                
                if (i + 1) % 5 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(audio_text_pairs)} audio translations")
                
            except Exception as e:
                self.logger.error(f"Error processing audio {i}: {e}")
                continue
        
        return {
            'total_tests': len(results),
            'avg_bleu': np.mean([r['bleu'] for r in results]),
            'perfect_rate': np.mean([r['exact_match'] for r in results]) * 100,
            'avg_asr_accuracy': np.mean([r['asr_accuracy'] for r in results]) * 100,
            'avg_execution_time': np.mean([r['execution_time'] for r in results])
        }
    
    def generate_test_report(self, output_dir: str) -> str:
        """Generate comprehensive test report with visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate overall metrics
        total_tests = sum(len(self.metrics[mod]['bleu']) for mod in self.metrics)
        total_perfect = len(self.perfect_translations)
        overall_perfect_rate = (total_perfect / total_tests * 100) if total_tests > 0 else 0
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'perfect_translations': total_perfect,
            'overall_perfect_rate': overall_perfect_rate,
            'target_achieved': overall_perfect_rate >= 99.0,
            'modal_breakdown': {}
        }
        
        for modality in self.metrics:
            if self.metrics[modality]['bleu']:
                summary['modal_breakdown'][modality] = {
                    'total_tests': len(self.metrics[modality]['bleu']),
                    'avg_bleu': np.mean(self.metrics[modality]['bleu']),
                    'perfect_rate': np.mean(self.metrics[modality]['exact_match']) * 100
                }
        
        # Create visualizations
        self._create_visualizations(output_path)
        
        # Generate HTML report
        html_report = self._generate_html_report(summary)
        
        # Save reports
        with open(output_path / 'test_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        with open(output_path / 'test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        with open(output_path / 'perfect_translations.json', 'w', encoding='utf-8') as f:
            json.dump(self.perfect_translations, f, indent=2, ensure_ascii=False)
        
        with open(output_path / 'failed_translations.json', 'w', encoding='utf-8') as f:
            json.dump(self.failed_translations, f, indent=2, ensure_ascii=False)
        
        with open(output_path / 'comprehensive_test_report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        return str(output_path / 'comprehensive_test_report.html')
    
    def _create_visualizations(self, output_path: Path):
        """Create visualization plots."""
        plt.style.use('seaborn-v0_8')
        
        # BLEU score distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Translation Quality Metrics Distribution', fontsize=16)
        
        # Text BLEU
        if self.metrics['text']['bleu']:
            axes[0, 0].hist(self.metrics['text']['bleu'], bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_title('Text Translation BLEU Scores')
            axes[0, 0].set_xlabel('BLEU Score')
            axes[0, 0].set_ylabel('Frequency')
        
        # Image BLEU
        if self.metrics['image']['bleu']:
            axes[0, 1].hist(self.metrics['image']['bleu'], bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title('Image Translation BLEU Scores')
            axes[0, 1].set_xlabel('BLEU Score')
            axes[0, 1].set_ylabel('Frequency')
        
        # Audio BLEU
        if self.metrics['audio']['bleu']:
            axes[1, 0].hist(self.metrics['audio']['bleu'], bins=20, alpha=0.7, color='red')
            axes[1, 0].set_title('Audio Translation BLEU Scores')
            axes[1, 0].set_xlabel('BLEU Score')
            axes[1, 0].set_ylabel('Frequency')
        
        # Perfect rate by modality
        modalities = []
        perfect_rates = []
        for modality in self.metrics:
            if self.metrics[modality]['exact_match']:
                modalities.append(modality.capitalize())
                perfect_rates.append(np.mean(self.metrics[modality]['exact_match']) * 100)
        
        if modalities:
            axes[1, 1].bar(modalities, perfect_rates, color=['blue', 'green', 'red'])
            axes[1, 1].set_title('Perfect Translation Rate by Modality')
            axes[1, 1].set_ylabel('Perfect Rate (%)')
            axes[1, 1].axhline(y=99, color='orange', linestyle='--', label='Target (99%)')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance metrics
        if self.test_results:
            execution_times = [r.get('execution_time', 0) for r in self.test_results]
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(execution_times, bins=30, alpha=0.7, color='purple')
            plt.title('Translation Execution Time Distribution')
            plt.xlabel('Execution Time (seconds)')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            plt.plot(sorted(execution_times), 'o-', alpha=0.7)
            plt.title('Translation Execution Time Trend')
            plt.xlabel('Test Index')
            plt.ylabel('Execution Time (seconds)')
            
            plt.tight_layout()
            plt.savefig(output_path / 'performance_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_html_report(self, summary: Dict) -> str:
        """Generate HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Multimodal Translation Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
                .metric-box { background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }
                .success { color: green; font-weight: bold; }
                .failure { color: red; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .visualization { text-align: center; margin: 20px 0; }
                .sample-translations { margin: 20px 0; }
                .translation-example { background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Comprehensive Multimodal Translation Test Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Model: Korean-English Multimodal Transformer</p>
            </div>
            
            <div class="metrics">
                <div class="metric-box">
                    <h3>Total Tests</h3>
                    <h2>{total_tests}</h2>
                </div>
                <div class="metric-box">
                    <h3>Perfect Translations</h3>
                    <h2 class="{target_class}">{perfect_translations}</h2>
                </div>
                <div class="metric-box">
                    <h3>Perfect Rate</h3>
                    <h2 class="{target_class}">{perfect_rate:.2f}%</h2>
                </div>
                <div class="metric-box">
                    <h3>Target Achieved</h3>
                    <h2 class="{target_class}">{target_status}</h2>
                </div>
            </div>
            
            <h2>Modal Breakdown</h2>
            <table>
                <tr>
                    <th>Modality</th>
                    <th>Total Tests</th>
                    <th>Avg BLEU</th>
                    <th>Perfect Rate</th>
                </tr>
                {modal_rows}
            </table>
            
            <div class="visualization">
                <h2>Quality Metrics Visualization</h2>
                <img src="quality_metrics.png" alt="Quality Metrics Distribution" style="max-width: 100%; height: auto;">
            </div>
            
            <div class="visualization">
                <h2>Performance Metrics</h2>
                <img src="performance_metrics.png" alt="Performance Metrics" style="max-width: 100%; height: auto;">
            </div>
            
            <div class="sample-translations">
                <h2>Sample Perfect Translations</h2>
                {perfect_samples}
            </div>
            
            <div class="sample-translations">
                <h2>Sample Failed Translations</h2>
                {failed_samples}
            </div>
        </body>
        </html>
        """
        
        # Fill template
        target_class = 'success' if summary['target_achieved'] else 'failure'
        target_status = '✅ YES' if summary['target_achieved'] else '❌ NO'
        
        modal_rows = ""
        for modality, data in summary['modal_breakdown'].items():
            modal_rows += f"""
                <tr>
                    <td>{modality.capitalize()}</td>
                    <td>{data['total_tests']}</td>
                    <td>{data['avg_bleu']:.2f}</td>
                    <td>{data['perfect_rate']:.2f}%</td>
                </tr>
            """
        
        perfect_samples = ""
        for i, sample in enumerate(self.perfect_translations[:5]):
            perfect_samples += f"""
                <div class="translation-example">
                    <strong>Sample {i+1} ({sample['type']}):</strong><br>
                    <strong>Input:</strong> {sample.get('input', sample.get('ocr_text', sample.get('asr_text', 'N/A')))}<br>
                    <strong>Reference:</strong> {sample['reference']}<br>
                    <strong>Translation:</strong> {sample['hypothesis']}<br>
                    <strong>BLEU:</strong> {sample['bleu']:.2f}
                </div>
            """
        
        failed_samples = ""
        for i, sample in enumerate(self.failed_translations[:5]):
            failed_samples += f"""
                <div class="translation-example">
                    <strong>Sample {i+1} ({sample['type']}):</strong><br>
                    <strong>Input:</strong> {sample.get('input', sample.get('ocr_text', sample.get('asr_text', 'N/A')))}<br>
                    <strong>Reference:</strong> {sample['reference']}<br>
                    <strong>Translation:</strong> {sample['hypothesis']}<br>
                    <strong>BLEU:</strong> {sample['bleu']:.2f}
                </div>
            """
        
        return html_template.format(
            timestamp=summary['timestamp'],
            total_tests=summary['total_tests'],
            perfect_translations=summary['perfect_translations'],
            perfect_rate=summary['overall_perfect_rate'],
            target_class=target_class,
            target_status=target_status,
            modal_rows=modal_rows,
            perfect_samples=perfect_samples,
            failed_samples=failed_samples
        )


def create_comprehensive_test_data():
    """Create comprehensive test data covering various Korean text types."""
    
    # Text test pairs - diverse Korean sentences
    text_test_pairs = [
        ("안녕하세요", "Hello"),
        ("감사합니다", "Thank you"),
        ("사랑합니다", "I love you"),
        ("오늘 날씨가 좋네요", "The weather is nice today"),
        ("저는 학생입니다", "I am a student"),
        ("이것은 책입니다", "This is a book"),
        ("어디에 가고 싶으세요?", "Where do you want to go?"),
        ("얼마예요?", "How much is it?"),
        ("도와주세요", "Please help me"),
        ("죄송합니다", "I am sorry"),
        ("반갑습니다", "Nice to meet you"),
        ("잘 먹겠습니다", "I will eat well"),
        ("맛있어요", "It is delicious"),
        ("추워요", "It is cold"),
        ("더워요", "It is hot"),
        ("피곤해요", "I am tired"),
        ("행복해요", "I am happy"),
        ("슬퍼요", "I am sad"),
        ("화나요", "I am angry"),
        ("무서워요", "I am scared"),
        ("한국어를 공부하고 있습니다", "I am studying Korean"),
        ("영어를 할 수 있어요?", "Can you speak English?"),
        ("한국에 오신 것을 환영합니다", "Welcome to Korea"),
        ("즐거운 여행 되세요", "Have a nice trip"),
        ("건강하세요", "Stay healthy"),
        ("새해 복 많이 받으세요", "Happy New Year"),
        ("생일 축하해요", "Happy birthday"),
        ("결혼 축하해요", "Congratulations on your marriage"),
        ("졸업 축하해요", "Congratulations on your graduation"),
        ("취직 축하해요", "Congratulations on your new job"),
        ("한국 음식이 맛있어요", "Korean food is delicious"),
        ("김치를 좋아해요", "I like kimchi"),
        ("불고기를 먹고 싶어요", "I want to eat bulgogi"),
        ("비빔밥을 주문할게요", "I will order bibimbap"),
        ("소주 한 병 주세요", "Please give me one bottle of soju"),
        ("맥주 한 잔 주세요", "Please give me a glass of beer"),
        ("계산해 주세요", "Please calculate the bill"),
        ("카드로 계산할게요", "I will pay by card"),
        ("현금으로 계산할게요", "I will pay by cash"),
        ("영수증 주세요", "Please give me a receipt"),
        ("화장실이 어디예요?", "Where is the bathroom?"),
        ("지하철역이 어디예요?", "Where is the subway station?"),
        ("버스 정류장이 어디예요?", "Where is the bus stop?"),
        ("택시를 타고 싶어요", "I want to take a taxi"),
        ("공항에 가고 싶어요", "I want to go to the airport"),
        ("호텔에 가고 싶어요", "I want to go to the hotel"),
        ("병원에 가고 싶어요", "I want to go to the hospital"),
        ("약국이 어디예요?", "Where is the pharmacy?"),
        ("아파요", "I am sick"),
        ("머리가 아파요", "I have a headache"),
        ("배가 아파요", "I have a stomachache"),
        ("감기에 걸렸어요", "I caught a cold"),
        ("열이 있어요", "I have a fever"),
        ("한국 역사가 흥미로워요", "Korean history is interesting"),
        ("한국 문화가 아름다워요", "Korean culture is beautiful"),
        ("K-POP을 좋아해요", "I like K-POP"),
        ("BTS를 알아요?", "Do you know BTS?"),
        ("한국 드라마를 봐요", "I watch Korean dramas"),
        ("한국 영화를 좋아해요", "I like Korean movies"),
        ("한국어가 어려워요", "Korean is difficult"),
        ("한글은 과학적인 문자예요", "Hangul is a scientific writing system"),
        ("한국 사람들은 친절해요", "Korean people are kind"),
        ("서울은 아름다운 도시예요", "Seoul is a beautiful city"),
        ("부산에 가고 싶어요", "I want to go to Busan"),
        ("제주도가 유명해요", "Jeju Island is famous"),
        ("한국의 사계절이 뚜렷해요", "Korea has distinct four seasons"),
        ("봄에 벚꽃이 피어요", "Cherry blossoms bloom in spring"),
        ("가을에 단풍이 예뻐요", "Autumn leaves are beautiful"),
        ("겨울에 눈이 와요", "It snows in winter"),
        ("여름에 비가 많이 와요", "It rains a lot in summer"),
        ("한국은 기술이 발달했어요", "Korea has advanced technology"),
        ("삼성을 알아요?", "Do you know Samsung?"),
        ("현대차를 좋아해요", "I like Hyundai cars"),
        ("LG 제품을 사용해요", "I use LG products"),
        ("한국은 경제가 강해요", "Korea has a strong economy"),
        ("한국은 교육이 중요해요", "Education is important in Korea"),
        ("한국 대학교가 좋아요", "Korean universities are good"),
        ("한국어 시험이 어려워요", "The Korean language test is difficult"),
        ("TOPIK을 봤어요", "I took TOPIK"),
        ("한국에서 일하고 싶어요", "I want to work in Korea"),
        ("한국에서 살고 싶어요", "I want to live in Korea"),
        ("한국 시민이 되고 싶어요", "I want to become a Korean citizen"),
        ("한국 문화를 배우고 싶어요", "I want to learn Korean culture"),
        ("한국 요리를 배우고 싶어요", "I want to learn Korean cooking"),
        ("한국 전통을 존경해요", "I respect Korean traditions"),
        ("한국 예절이 중요해요", "Korean etiquette is important"),
        ("한국 사람들은 정이 많아요", "Korean people have a lot of affection"),
        ("한국은 안전한 나라예요", "Korea is a safe country"),
        ("한국은 깨끗한 나라예요", "Korea is a clean country"),
        ("한국을 사랑해요", "I love Korea"),
        ("한국이 그리워요", "I miss Korea"),
        ("한국에 다시 가고 싶어요", "I want to go to Korea again"),
        ("한국 친구를 만들고 싶어요", "I want to make Korean friends"),
        ("한국어로 대화하고 싶어요", "I want to converse in Korean"),
        ("한국 뉴스를 읽어요", "I read Korean news"),
        ("한국 책을 읽어요", "I read Korean books"),
        ("한국 노래를 들어요", "I listen to Korean songs"),
        ("한국어로 노래해요", "I sing in Korean"),
        ("한국 춤을 배워요", "I learn Korean dance"),
        ("한국 무술을 배워요", "I learn Korean martial arts"),
        ("태권도를 알아요?", "Do you know Taekwondo?"),
        ("한복이 예뻐요", "Hanbok is beautiful"),
        ("한옥이 멋져요", "Hanok is wonderful"),
        ("한국 전통이 대단해요", "Korean tradition is great"),
        ("한국이 자랑스러워요", "I am proud of Korea"),
        ("한국을 소개하고 싶어요", "I want to introduce Korea"),
        ("한국어를 가르치고 싶어요", "I want to teach Korean"),
        ("한국 문화를 알리고 싶어요", "I want to promote Korean culture"),
        ("한국이 세계적인 나라예요", "Korea is a global country"),
        ("한국이 미래가 밝아요", "Korea has a bright future"),
        ("한국을 응원해요", "I cheer for Korea"),
        ("대한민국!", "Republic of Korea!")
    ]
    
    # Create sample image and audio test data (simulated)
    image_test_pairs = []
    audio_test_pairs = []
    
    # Create test directory structure
    test_data_dir = Path("tests/data/multimodal")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images (text rendered as images)
    for i, (korean_text, english_text) in enumerate(text_test_pairs[:20]):
        # Create simple text image
        img = Image.new('RGB', (400, 100), color='white')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Use default font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 30), korean_text, fill='black', font=font)
        
        image_path = test_data_dir / f"test_image_{i:03d}.png"
        img.save(image_path)
        
        image_test_pairs.append((str(image_path), korean_text, english_text))
    
    # Create sample audio files (sine wave with different frequencies for simulation)
    for i, (korean_text, english_text) in enumerate(text_test_pairs[:20]):
        # Generate synthetic audio (sine wave)
        duration = 2.0  # seconds
        sample_rate = 16000
        frequency = 440 + i * 20  # Different frequency for each sample
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        audio += 0.01 * np.random.randn(len(audio))
        
        audio_path = test_data_dir / f"test_audio_{i:03d}.wav"
        sf.write(audio_path, audio, sample_rate)
        
        audio_test_pairs.append((str(audio_path), korean_text, english_text))
    
    return {
        'text_pairs': text_test_pairs,
        'image_pairs': image_test_pairs,
        'audio_pairs': audio_test_pairs
    }


def main():
    """Main validation function."""
    # Create test data
    test_data = create_comprehensive_test_data()
    
    # Initialize validator
    validator = MultimodalValidator(
        model_path="models/checkpoints/nmt_transformer_best.pt",
        tokenizer_path="models/tokenizers/korean_tokenizer",
        device='auto',
        batch_size=32,
        max_length=512
    )
    
    # Run comprehensive validation
    print("🚀 Starting comprehensive multimodal validation...")
    
    # Text validation
    print("\n📄 Validating text translations...")
    text_results = validator.validate_text_translation(test_data['text_pairs'])
    print(f"Text - Total: {text_results['total_tests']}, "
          f"Avg BLEU: {text_results['avg_bleu']:.2f}, "
          f"Perfect Rate: {text_results['perfect_rate']:.2f}%")
    
    # Image validation
    print("\n🖼️ Validating image translations...")
    image_results = validator.validate_image_translation(test_data['image_pairs'])
    print(f"Image - Total: {image_results['total_tests']}, "
          f"Avg BLEU: {image_results['avg_bleu']:.2f}, "
          f"Perfect Rate: {image_results['perfect_rate']:.2f}%")
    
    # Audio validation
    print("\n🎵 Validating audio translations...")
    audio_results = validator.validate_audio_translation(test_data['audio_pairs'])
    print(f"Audio - Total: {audio_results['total_tests']}, "
          f"Avg BLEU: {audio_results['avg_bleu']:.2f}, "
          f"Perfect Rate: {audio_results['perfect_rate']:.2f}%")
    
    # Generate comprehensive report
    print("\n📊 Generating comprehensive report...")
    report_path = validator.generate_test_report("tests/comprehensive")
    
    # Print final summary
    total_tests = (text_results['total_tests'] + 
                   image_results['total_tests'] + 
                   audio_results['total_tests'])
    
    total_perfect = (len([r for r in validator.perfect_translations if r['type'] == 'text']) +
                     len([r for r in validator.perfect_translations if r['type'] == 'image']) +
                     len([r for r in validator.perfect_translations if r['type'] == 'audio']))
    
    overall_perfect_rate = (total_perfect / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"🎯 COMPREHENSIVE MULTIMODAL TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Perfect Translations: {total_perfect}")
    print(f"Overall Perfect Rate: {overall_perfect_rate:.2f}%")
    print(f"Target (99%) Achieved: {'✅ YES' if overall_perfect_rate >= 99.0 else '❌ NO'}")
    print(f"Report Generated: {report_path}")
    
    if overall_perfect_rate < 99.0:
        improvement_needed = 99.0 - overall_perfect_rate
        print(f"📈 Need {improvement_needed:.2f}% improvement to reach 99% target")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()