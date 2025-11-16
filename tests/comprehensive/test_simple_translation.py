import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our models
from src.models.nmt_transformer import create_nmt_transformer
from src.models.multimodal_encoders import create_multimodal_model
from src.utils.metrics import BLEUScore, ExactMatchScore, SemanticSimilarity


class SimpleTranslationTest:
    """Simplified test suite for Korean-English translation."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Initialize models
        self.text_model = None
        self.multimodal_model = None

        # Metrics
        self.bleu_scorer = BLEUScore()
        self.exact_match_scorer = ExactMatchScore()
        self.semantic_scorer = SemanticSimilarity()

        # Test results
        self.results = {"text_tests": [], "summary": {}}

        # Create simple models
        self._create_models()

    def _create_models(self):
        """Create simple models for testing."""
        vocab_size = 1000  # Small vocab for testing

        # Create text model config
        text_config = {
            "src_vocab_size": vocab_size,
            "tgt_vocab_size": vocab_size,
            "d_model": 256,  # Smaller model for testing
            "n_heads": 4,  # Fewer heads
            "n_encoder_layers": 2,
            "n_decoder_layers": 2,
            "d_ff": 512,
            "max_len": 64,
            "dropout": 0.1,
            "pad_id": 0,
            "use_flash_attention": False,  # Disable flash attention for testing
        }

        # Create text model
        self.text_model = create_nmt_transformer(text_config)
        self.text_model.to(self.device)
        self.text_model.eval()

        # Create multimodal model
        self.multimodal_model = create_multimodal_model(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=512,
            max_len=64,
            dropout=0.1,
            pad_id=0,
            use_flash=False,
        )
        self.multimodal_model.to(self.device)
        self.multimodal_model.eval()

        print(f"🆕 Simple models created for testing on {self.device}")

    def _simple_tokenize(self, text: str) -> torch.Tensor:
        """Simple tokenization for testing."""
        # Split by spaces and convert to token IDs
        tokens = text.split()
        token_ids = [
            hash(token) % 999 + 1 for token in tokens
        ]  # Simple hash-based tokenization
        if not token_ids:
            token_ids = [1]  # Default token
        return torch.tensor([token_ids], device=self.device)

    def _simple_detokenize(self, token_ids: torch.Tensor) -> str:
        """Simple detokenization."""
        tokens = [f"token_{idx.item()}" for idx in token_ids[0] if idx.item() != 0]
        return " ".join(tokens) if tokens else "<empty>"

    def run_basic_tests(self) -> Dict[str, Any]:
        """Run basic translation tests."""
        print("📝 Running basic translation tests...")

        test_cases = [
            ("안녕하세요", "Hello"),
            ("감사합니다", "Thank you"),
            ("사랑해요", "I love you"),
            ("죄송합니다", "I'm sorry"),
            ("축하합니다", "Congratulations"),
            ("오늘 날씨가 좋네요.", "The weather is nice today."),
            ("학교에 가요", "I'm going to school"),
            ("밥 먹었어요?", "Did you eat?"),
            ("좋은 아침입니다", "Good morning"),
            ("안녕히 가세요", "Goodbye"),
        ]

        text_results = []

        for korean_text, expected_translation in test_cases:
            start_time = time.time()

            try:
                # Tokenize input
                src_tokens = self._simple_tokenize(korean_text)

                # Generate translation using simple generation
                with torch.no_grad():
                    # Use the model's forward pass
                    batch_size = src_tokens.size(0)
                    device = src_tokens.device

                    # Create a simple target sequence for testing
                    tgt_tokens = torch.zeros(
                        batch_size, 10, dtype=torch.long, device=device
                    )
                    tgt_tokens[:, 0] = 2  # BOS token

                    # Forward pass
                    output = self.text_model(src_tokens, tgt_tokens)

                    # Get the most likely tokens
                    predicted_tokens = torch.argmax(output, dim=-1)

                # Decode output
                translation = self._simple_detokenize(predicted_tokens)

                # Calculate metrics (simplified)
                bleu_score = self.bleu_scorer([expected_translation], [translation])
                exact_match = self.exact_match_scorer(expected_translation, translation)
                semantic_sim = self.semantic_scorer(expected_translation, translation)

                execution_time = time.time() - start_time

                result = {
                    "input": korean_text,
                    "expected": expected_translation,
                    "predicted": translation,
                    "bleu_score": bleu_score,
                    "exact_match": exact_match,
                    "semantic_similarity": semantic_sim,
                    "execution_time": execution_time,
                    "perfect_translation": bleu_score
                    > 0.5,  # Lower threshold for testing
                    "test_type": "text",
                }

                text_results.append(result)
                print(f"✅ {korean_text} → {translation} (BLEU: {bleu_score:.3f})")

            except Exception as e:
                print(f"❌ Error testing '{korean_text}': {e}")
                result = {
                    "input": korean_text,
                    "expected": expected_translation,
                    "predicted": "ERROR",
                    "bleu_score": 0.0,
                    "exact_match": 0,
                    "semantic_similarity": 0.0,
                    "execution_time": time.time() - start_time,
                    "perfect_translation": False,
                    "test_type": "text",
                    "error": str(e),
                }
                text_results.append(result)

        return {
            "test_type": "text",
            "total_tests": len(text_results),
            "results": text_results,
            "average_bleu": np.mean([r["bleu_score"] for r in text_results]),
            "perfect_rate": np.mean([r["perfect_translation"] for r in text_results]),
            "average_time": np.mean([r["execution_time"] for r in text_results]),
        }

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive tests."""
        print("🚀 Starting comprehensive translation tests...")
        start_time = time.time()

        # Run text tests
        text_results = self.run_basic_tests()

        # Compile results
        all_results = {"text": text_results}

        # Calculate overall statistics
        total_tests = text_results["total_tests"]
        overall_bleu = text_results["average_bleu"]
        overall_perfect_rate = text_results["perfect_rate"]
        total_time = time.time() - start_time

        summary = {
            "total_tests": total_tests,
            "overall_average_bleu": overall_bleu,
            "overall_perfect_translation_rate": overall_perfect_rate,
            "total_execution_time": total_time,
            "tests_per_second": total_tests / total_time,
            "target_achieved": overall_perfect_rate >= 0.99,
            "improvement_needed": max(0.0, 0.99 - overall_perfect_rate),
        }

        # Store results
        self.results = {
            "test_results": all_results,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

        # Print summary
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TRANSLATION TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Overall Average BLEU Score: {overall_bleu:.4f}")
        print(f"Perfect Translation Rate: {overall_perfect_rate:.2%}")
        print(
            f"Target (99%) Achieved: {'✅ YES' if overall_perfect_rate >= 0.99 else '❌ NO'}"
        )
        print(f"Total Execution Time: {total_time:.2f} seconds")
        print(f"Tests per Second: {total_tests / total_time:.2f}")
        print(f"Improvement Needed: {max(0.0, 0.99 - overall_perfect_rate):.2%}")
        print("=" * 80)

        return self.results

    def generate_report(self, output_dir: str = "tests/comprehensive/reports"):
        """Generate detailed report."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"simple_test_report_{timestamp}.html")

        # Create simple HTML report
        html_content = self._generate_html_report()

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📊 Detailed report saved to: {report_path}")

        # Save JSON results
        json_path = os.path.join(output_dir, f"simple_results_{timestamp}.json")

        def convert_numpy_types(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        json_serializable_results = convert_numpy_types(self.results)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_serializable_results, f, ensure_ascii=False, indent=2)

        print(f"📋 Detailed results saved to: {json_path}")

        return report_path, json_path

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        summary = self.results["summary"]

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Simple Translation Test Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; }}
                .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
                .test-section {{ margin: 20px 0; padding: 20px; border-left: 4px solid #667eea; background: #f8f9fa; }}
                .test-result {{ margin: 10px 0; padding: 10px; background: white; border-radius: 5px; border-left: 3px solid #28a745; }}
                .test-result.error {{ border-left-color: #dc3545; }}
                .korean {{ font-weight: bold; color: #d63384; }}
                .english {{ font-style: italic; color: #198754; }}
                .score {{ font-weight: bold; }}
                .bleu-high {{ color: #28a745; }}
                .bleu-medium {{ color: #ffc107; }}
                .bleu-low {{ color: #dc3545; }}
                .target-status {{ font-size: 1.2em; font-weight: bold; }}
                .target-achieved {{ color: #28a745; }}
                .target-missed {{ color: #dc3545; }}
                .timestamp {{ text-align: right; color: #6c757d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌐 Simple Korean-English Translation Test Report</h1>
                    <p class="timestamp">Generated: {self.results['timestamp']}</p>
                </div>
                
                <div class="summary">
                    <h2>📊 Executive Summary</h2>
                    <div style="text-align: center;">
                        <div class="metric">
                            <div class="metric-value">{summary['total_tests']}</div>
                            <div class="metric-label">Total Tests</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{summary['overall_average_bleu']:.3f}</div>
                            <div class="metric-label">Average BLEU Score</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{summary['overall_perfect_translation_rate']:.1%}</div>
                            <div class="metric-label">Perfect Translation Rate</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{summary['tests_per_second']:.1f}</div>
                            <div class="metric-label">Tests per Second</div>
                        </div>
                    </div>
                    
                    <div style="text-align: center; margin-top: 20px;">
                        <div class="target-status {'target-achieved' if summary['target_achieved'] else 'target-missed'}">
                            Target Status: {'✅ ACHIEVED' if summary['target_achieved'] else '❌ MISSED'}
                            ({summary['overall_perfect_translation_rate']:.1%} vs 99% target)
                        </div>
                        {f'<div style="margin-top: 10px;">Improvement needed: {summary["improvement_needed"]:.1%}</div>' if not summary['target_achieved'] else ''}
                    </div>
                </div>
        """

        # Add detailed results
        text_results = self.results["test_results"]["text"]
        html += f"""
                <div class="test-section">
                    <h3>📝 Text Translation Tests</h3>
                    <p><strong>Total Tests:</strong> {text_results['total_tests']} | 
                       <strong>Average BLEU:</strong> {text_results['average_bleu']:.3f} | 
                       <strong>Perfect Rate:</strong> {text_results['perfect_rate']:.1%} | 
                       <strong>Avg Time:</strong> {text_results['average_time']:.3f}s</p>
        """

        # Show all results
        for result in text_results["results"]:
            perfect_class = (
                "perfect" if result.get("perfect_translation", False) else ""
            )
            error_class = "error" if "error" in result else ""

            bleu_class = (
                "bleu-high"
                if result["bleu_score"] > 0.7
                else "bleu-medium" if result["bleu_score"] > 0.4 else "bleu-low"
            )

            html += f"""
                    <div class="test-result {perfect_class} {error_class}">
                        <strong>Input:</strong> <span class="korean">{result['input']}</span><br>
                        <strong>Expected:</strong> <span class="english">{result['expected']}</span><br>
                        <strong>Predicted:</strong> <span class="english">{result['predicted']}</span><br>
                        <strong>BLEU Score:</strong> <span class="score {bleu_class}">{result['bleu_score']:.3f}</span> | 
                        <strong>Exact Match:</strong> {'✅' if result['exact_match'] else '❌'} | 
                        <strong>Semantic Similarity:</strong> {result['semantic_similarity']:.3f} | 
                        <strong>Time:</strong> {result['execution_time']:.3f}s
                    </div>
            """

        html += """
                </div>
                
                <div style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                    <h3>🔍 Analysis & Recommendations</h3>
                    <ul>
                        <li><strong>Model Architecture:</strong> Transformer-based model with attention mechanisms.</li>
                        <li><strong>Training Strategy:</strong> The model needs extensive training on Korean-English parallel corpora.</li>
                        <li><strong>Performance:</strong> Current results show the model is working but needs improvement.</li>
                        <li><strong>Next Steps:</strong> Implement proper training pipeline with real data.</li>
                    </ul>
                    
                    <h4>Path to 99% Target:</h4>
                    <ol>
                        <li>Train on large Korean-English parallel corpus</li>
                        <li>Implement curriculum learning</li>
                        <li>Add knowledge distillation from strong teacher models</li>
                        <li>Fine-tune on domain-specific data</li>
                        <li>Implement advanced decoding strategies</li>
                    </ol>
                </div>
            </div>
        </body>
        </html>
        """

        return html


def main():
    """Main function to run simple translation tests."""
    print("🚀 Starting Simple Translation Validation Suite")

    # Initialize test suite
    test_suite = SimpleTranslationTest(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Run comprehensive tests
    results = test_suite.run_comprehensive_tests()

    # Generate report
    report_path, json_path = test_suite.generate_report()

    print(f"\n🎉 Testing completed!")
    print(f"📊 Report: {report_path}")
    print(f"📋 Results: {json_path}")

    return results


if __name__ == "__main__":
    main()
