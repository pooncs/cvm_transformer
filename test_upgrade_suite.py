import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import logging

# Test imports for our modules
from cvm_translator.language_detector import (
    LanguageDetector, DetectionMode, DetectionResult, 
    create_language_detector
)
from cvm_translator.unified_translator import (
    UnifiedTranslator, TranslationRequest, TranslationResponse,
    translate_text, translate_audio, translate_image
)
from cvm_translator.edge_quantization import (
    EdgeQuantizationEngine, QuantizationConfig, 
    CVMQuantizationOptimizer, quantize_for_edge
)
from cvm_translator.mobile_deployment import (
    ONNXExporter, CoreMLExporter, MobileDeploymentManager,
    ExportConfig, export_for_mobile
)
from cvm_translator.prepare_multimodal_corpus import (
    MultimodalCorpusPreparer, MultimodalCorpusConfig,
    prepare_large_corpus
)


class TestLanguageDetector(unittest.TestCase):
    """Test suite for multimodal language detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = LanguageDetector(
            models_dir=f"{self.temp_dir}/models",
            confidence_threshold=0.5
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_text_language_detection(self):
        """Test text language detection functionality."""
        # Test English
        result = self.detector.detect_text_language("Hello, how are you?")
        if result:  # Only test if models are available
            self.assertEqual(result.mode, DetectionMode.TEXT)
            self.assertGreater(result.confidence, 0.0)
            self.assertIn(result.language, ['en', 'unknown'])
        
        # Test Korean
        result = self.detector.detect_text_language("안녕하세요, 어떻게 지내세요?")
        if result:
            self.assertEqual(result.mode, DetectionMode.TEXT)
            self.assertGreater(result.confidence, 0.0)
            self.assertIn(result.language, ['ko', 'unknown'])
        
        # Test Japanese
        result = self.detector.detect_text_language("こんにちは、お元気ですか？")
        if result:
            self.assertEqual(result.mode, DetectionMode.TEXT)
            self.assertGreater(result.confidence, 0.0)
            self.assertIn(result.language, ['ja', 'unknown'])
    
    def test_auto_detection_mode(self):
        """Test automatic input type detection."""
        # Test text auto-detection
        result = self.detector.detect("Hello world", mode=DetectionMode.AUTO)
        if result:
            self.assertEqual(result.mode, DetectionMode.TEXT)
        
        # Test with empty text
        result = self.detector.detect_text_language("")
        self.assertIsNone(result)
        
        # Test with None
        result = self.detector.detect_text_language(None)
        self.assertIsNone(result)
    
    def test_detection_result_structure(self):
        """Test DetectionResult data structure."""
        result = DetectionResult(
            language="en",
            confidence=0.95,
            mode=DetectionMode.TEXT,
            metadata={"model": "test", "text_length": 20}
        )
        
        self.assertEqual(result.language, "en")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.mode, DetectionMode.TEXT)
        self.assertEqual(result.metadata["model"], "test")
        self.assertIsNotNone(result.metadata)
    
    def test_batch_detection(self):
        """Test batch language detection."""
        test_inputs = [
            "Hello world",
            "안녕하세요",
            "こんにちは",
            "你好"
        ]
        
        results = self.detector.batch_detect(test_inputs)
        
        self.assertEqual(len(results), len(test_inputs))
        
        for result in results:
            if result:  # Only check if detection succeeded
                self.assertIsInstance(result, DetectionResult)
                self.assertGreater(result.confidence, 0.0)
    
    def test_supported_languages(self):
        """Test supported languages query."""
        text_langs = self.detector.get_supported_languages(DetectionMode.TEXT)
        self.assertIsInstance(text_langs, list)
        
        # Should include common languages
        if len(text_langs) > 0:
            self.assertTrue(any(lang in ['en', 'ko', 'ja', 'zh'] for lang in text_langs))
    
    def test_factory_function(self):
        """Test language detector factory function."""
        config = {
            'models_dir': f"{self.temp_dir}/factory_models",
            'confidence_threshold': 0.6
        }
        
        detector = create_language_detector(config)
        self.assertIsInstance(detector, LanguageDetector)
        self.assertEqual(detector.confidence_threshold, 0.6)


class TestUnifiedTranslator(unittest.TestCase):
    """Test suite for unified translator with automatic routing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock models for testing
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.vocab_size = 1000
        self.mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.mock_tokenizer.decode = Mock(return_value="Translated text")
        
        # Mock CVM model
        self.mock_cvm_model = Mock()
        self.mock_cvm_model.eval = Mock()
        self.mock_cvm_model.to = Mock(return_value=self.mock_cvm_model)
        self.mock_cvm_model.load_state_dict = Mock()
        
        # Create translator with mocked dependencies
        with patch('cvm_translator.unified_translator.SentencePieceTokenizer') as mock_spm, \
             patch('cvm_translator.unified_translator.CVMTransformer') as mock_cvm, \
             patch('cvm_translator.unified_translator.LanguageDetector') as mock_detector:
            
            mock_spm.return_value = self.mock_tokenizer
            mock_cvm.return_value = self.mock_cvm_model
            mock_detector.return_value = Mock()
            
            self.translator = UnifiedTranslator(
                cvm_model_path="dummy_model",
                vocab_path="dummy_vocab",
                enable_logging=False
            )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_translation_request_creation(self):
        """Test TranslationRequest creation."""
        request = TranslationRequest(
            input_data="Hello world",
            input_mode=DetectionMode.TEXT,
            target_language="ko",
            source_language="en",
            enable_fallback=True
        )
        
        self.assertEqual(request.input_data, "Hello world")
        self.assertEqual(request.target_language, "ko")
        self.assertEqual(request.source_language, "en")
        self.assertTrue(request.enable_fallback)
        self.assertIsNotNone(request.metadata)
    
    def test_translation_response_structure(self):
        """Test TranslationResponse data structure."""
        detection_result = DetectionResult(
            language="en",
            confidence=0.95,
            mode=DetectionMode.TEXT
        )
        
        response = TranslationResponse(
            translated_text="안녕하세요",
            source_language="en",
            target_language="ko",
            confidence=0.95,
            detection_result=detection_result,
            processing_time=0.1,
            fallback_used=False
        )
        
        self.assertEqual(response.translated_text, "안녕하세요")
        self.assertEqual(response.source_language, "en")
        self.assertEqual(response.target_language, "ko")
        self.assertEqual(response.confidence, 0.95)
        self.assertEqual(response.detection_result, detection_result)
        self.assertGreater(response.processing_time, 0)
        self.assertFalse(response.fallback_used)
    
    def test_convenience_functions(self):
        """Test convenience translation functions."""
        # Test text translation
        with patch('cvm_translator.unified_translator.UnifiedTranslator') as mock_translator_class:
            mock_instance = Mock()
            mock_response = TranslationResponse(
                translated_text="안녕하세요",
                source_language="en",
                target_language="ko",
                confidence=0.95,
                detection_result=Mock(),
                processing_time=0.1
            )
            mock_instance.translate = Mock(return_value=mock_response)
            mock_translator_class.return_value = mock_instance
            
            response = translate_text("Hello", target_language="ko")
            self.assertIsInstance(response, TranslationResponse)
            self.assertEqual(response.translated_text, "안녕하세요")
    
    def test_health_check(self):
        """Test translator health check."""
        health = self.translator.health_check()
        
        self.assertIsInstance(health, dict)
        self.assertIn('language_detector', health)
        self.assertIn('cvm_model', health)
        self.assertIn('tokenizer', health)
        self.assertIn('device', health)
        self.assertIn('supported_pairs', health)
        self.assertIn('overall', health)


class TestEdgeQuantization(unittest.TestCase):
    """Test suite for edge quantization engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = QuantizationConfig(method="int8", bits=8)
        self.engine = EdgeQuantizationEngine(self.config)
        
        # Create a simple test model
        self.test_model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_quantization_config(self):
        """Test quantization configuration."""
        self.assertEqual(self.config.method, "int8")
        self.assertEqual(self.config.bits, 8)
        self.assertEqual(self.config.group_size, 128)
        self.assertFalse(self.config.desc_act)
    
    def test_calibration_data_generation(self):
        """Test calibration data generation."""
        calibration_data = self.engine._generate_calibration_data(None, size=10)
        
        self.assertEqual(len(calibration_data), 10)
        self.assertIsInstance(calibration_data[0], str)
        self.assertGreater(len(calibration_data[0]), 0)
    
    def test_benchmark_quantized_model(self):
        """Test quantized model benchmarking."""
        test_data = ["Hello world", "Test sentence", "Another example"]
        
        # Create a mock quantized model path
        quantized_path = f"{self.temp_dir}/test_quantized"
        Path(quantized_path).mkdir(exist_ok=True)
        
        # Create mock model files
        (Path(quantized_path) / "pytorch_model.bin").touch()
        
        results = self.engine.benchmark_quantized_model(
            quantized_path, 
            test_data=test_data
        )
        
        self.assertIn('model_path', results)
        self.assertIn('test_samples', results)
        self.assertEqual(results['test_samples'], len(test_data))
    
    def test_convenience_function(self):
        """Test quantization convenience function."""
        with patch('cvm_translator.edge_quantization.EdgeQuantizationEngine.quantize_model') as mock_quantize:
            mock_quantize.return_value = f"{self.temp_dir}/quantized_model"
            
            result = quantize_for_edge(
                self.test_model,
                method="int8",
                bits=8,
                output_dir=self.temp_dir
            )
            
            self.assertEqual(result, f"{self.temp_dir}/quantized_model")


class TestMobileDeployment(unittest.TestCase):
    """Test suite for mobile deployment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExportConfig(
            format="onnx",
            target_devices=["ios", "android"],
            enable_quantization=True
        )
        
        # Create a simple test model
        self.test_model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_export_config(self):
        """Test export configuration."""
        self.assertEqual(self.config.format, "onnx")
        self.assertIn("ios", self.config.target_devices)
        self.assertIn("android", self.config.target_devices)
        self.assertTrue(self.config.enable_quantization)
        self.assertEqual(self.config.opset_version, 11)
    
    def test_onnx_exporter_creation(self):
        """Test ONNX exporter creation."""
        try:
            exporter = ONNXExporter(self.config)
            self.assertIsInstance(exporter, ONNXExporter)
        except RuntimeError as e:
            # ONNX might not be available in test environment
            self.assertIn("ONNX export requires", str(e))
    
    def test_dummy_input_creation(self):
        """Test dummy input creation for export."""
        try:
            exporter = ONNXExporter(self.config)
            
            # Test with tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {
                'input_ids': torch.randint(0, 1000, (1, 128)),
                'attention_mask': torch.ones((1, 128))
            }
            
            dummy_input = exporter._create_dummy_input(mock_tokenizer)
            self.assertIsInstance(dummy_input, tuple)
            self.assertEqual(len(dummy_input), 2)
            
        except RuntimeError as e:
            # ONNX might not be available
            self.assertIn("ONNX export requires", str(e))
    
    def test_dynamic_axes_configuration(self):
        """Test dynamic axes configuration."""
        try:
            exporter = ONNXExporter(self.config)
            dynamic_axes = exporter._get_dynamic_axes()
            
            self.assertIn('input_ids', dynamic_axes)
            self.assertIn('attention_mask', dynamic_axes)
            self.assertIn('logits', dynamic_axes)
            
        except RuntimeError as e:
            self.assertIn("ONNX export requires", str(e))
    
    def test_convenience_function(self):
        """Test mobile deployment convenience function."""
        with patch('cvm_translator.mobile_deployment.MobileDeploymentManager.deploy_model') as mock_deploy:
            mock_deploy.return_value = {
                "ios": f"{self.temp_dir}/ios/model.mlmodel",
                "android": f"{self.temp_dir}/android/model.onnx"
            }
            
            result = export_for_mobile(
                self.test_model,
                target_platforms=["ios", "android"],
                output_dir=self.temp_dir
            )
            
            self.assertIn("ios", result)
            self.assertIn("android", result)


class TestMultimodalCorpusPreparation(unittest.TestCase):
    """Test suite for multimodal corpus preparation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MultimodalCorpusConfig(
            vocab_size=8000,
            min_sentence_length=3,
            max_sentence_length=128,
            enable_multilingual=True
        )
        self.preparer = MultimodalCorpusPreparer(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_corpus_config(self):
        """Test corpus configuration."""
        self.assertEqual(self.config.vocab_size, 8000)
        self.assertEqual(self.config.min_sentence_length, 3)
        self.assertEqual(self.config.max_sentence_length, 128)
        self.assertTrue(self.config.enable_multilingual)
    
    def test_text_line_validation(self):
        """Test text line validation."""
        # Valid lines
        self.assertTrue(self.preparer._is_valid_text_line("Hello world, this is a test."))
        self.assertTrue(self.preparer._is_valid_text_line("안녕하세요, 이것은 테스트입니다."))
        
        # Invalid lines
        self.assertFalse(self.preparer._is_valid_text_line(""))
        self.assertFalse(self.preparer._is_valid_text_line("A"))
        self.assertFalse(self.preparer._is_valid_text_line("http://example.com"))
    
    def test_language_extraction_from_path(self):
        """Test language code extraction from file paths."""
        test_cases = [
            ("/data/korean/file.txt", "ko"),
            ("/data/english/file.txt", "en"),
            ("/data/japanese/file.txt", "ja"),
            ("/data/chinese/file.txt", "zh"),
            ("/data/unknown/file.txt", None),
        ]
        
        for path, expected_lang in test_cases:
            extracted_lang = self.preparer._extract_language_from_path(path)
            self.assertEqual(extracted_lang, expected_lang)
    
    def test_unified_corpus_creation(self):
        """Test unified corpus creation."""
        text_corpora = {
            'en': ["Hello world", "How are you?"],
            'ko': ["안녕하세요", "어떻게 지내세요?"]
        }
        audio_corpora = {
            'en': [{'audio': np.array([1,2,3]), 'duration': 3.0}],
            'ko': [{'audio': np.array([4,5,6]), 'duration': 3.0}]
        }
        image_corpora = {
            'en': [{'image_path': 'test.jpg', 'extracted_text': 'Hello'}],
            'ko': [{'image_path': 'test2.jpg', 'extracted_text': '안녕'}]
        }
        
        unified = self.preparer.create_unified_corpus(
            text_corpora, audio_corpora, image_corpora
        )
        
        self.assertIn('en', unified)
        self.assertIn('ko', unified)
        
        for lang in ['en', 'ko']:
            self.assertIn('text', unified[lang])
            self.assertIn('audio', unified[lang])
            self.assertIn('image', unified[lang])
            self.assertIn('statistics', unified[lang])
    
    def test_corpus_metadata_saving(self):
        """Test corpus metadata saving."""
        unified_corpus = {
            'en': {
                'text': ["Hello", "World"],
                'audio': [],
                'image': [],
                'statistics': {'total_samples': 2}
            }
        }
        
        tokenizer_path = f"{self.temp_dir}/test_tokenizer.model"
        metadata_path = self.preparer.save_corpus_metadata(unified_corpus, tokenizer_path)
        
        self.assertTrue(Path(metadata_path).exists())
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertIn('config', metadata)
        self.assertIn('corpus_statistics', metadata)
        self.assertIn('tokenizer_info', metadata)
        self.assertEqual(metadata['tokenizer_info']['model_path'], tokenizer_path)
    
    def test_convenience_function(self):
        """Test large corpus preparation convenience function."""
        with patch('cvm_translator.prepare_multimodal_corpus.MultimodalCorpusPreparer.run_full_preparation') as mock_prepare:
            mock_prepare.return_value = {
                'tokenizer_model': f"{self.temp_dir}/tokenizer.model",
                'corpus_metadata': f"{self.temp_dir}/metadata.json"
            }
            
            result = prepare_large_corpus(vocab_size=8000)
            self.assertEqual(result, f"{self.temp_dir}/tokenizer.model")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock components
        self.mock_detector = Mock()
        self.mock_detector.detect = Mock(return_value=DetectionResult(
            language="en", confidence=0.95, mode=DetectionMode.TEXT
        ))
        
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.vocab_size = 1000
        self.mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.mock_tokenizer.decode = Mock(return_value="Translated text")
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_text_translation(self):
        """Test end-to-end text translation workflow."""
        with patch('cvm_translator.unified_translator.LanguageDetector') as mock_detector_class, \
             patch('cvm_translator.unified_translator.SentencePieceTokenizer') as mock_spm, \
             patch('cvm_translator.unified_translator.CVMTransformer') as mock_cvm:
            
            mock_detector_class.return_value = self.mock_detector
            mock_spm.return_value = self.mock_tokenizer
            mock_cvm.return_value = Mock()
            
            # Test the complete workflow
            response = translate_text("Hello world", target_language="ko")
            
            self.assertIsInstance(response, TranslationResponse)
            self.assertEqual(response.translated_text, "Translated text")
            self.assertEqual(response.source_language, "en")
            self.assertEqual(response.target_language, "ko")
    
    def test_multimodal_detection_integration(self):
        """Test multimodal language detection integration."""
        detector = LanguageDetector(models_dir=f"{self.temp_dir}/models")
        
        # Test that detector can be created and configured
        self.assertIsInstance(detector, LanguageDetector)
        self.assertIsNotNone(detector.confidence_threshold)
        
        # Test available modes
        available_modes = detector.get_available_modes()
        self.assertIsInstance(available_modes, list)
    
    def test_system_health_check(self):
        """Test overall system health."""
        with patch('cvm_translator.unified_translator.LanguageDetector') as mock_detector_class, \
             patch('cvm_translator.unified_translator.SentencePieceTokenizer') as mock_spm, \
             patch('cvm_translator.unified_translator.CVMTransformer') as mock_cvm:
            
            mock_detector_class.return_value = self.mock_detector
            mock_spm.return_value = self.mock_tokenizer
            mock_cvm.return_value = Mock()
            
            translator = UnifiedTranslator(enable_logging=False)
            health = translator.health_check()
            
            self.assertTrue(health['overall'])


class TestPerformance(unittest.TestCase):
    """Performance benchmarks for the system."""
    
    def test_language_detection_performance(self):
        """Benchmark language detection performance."""
        detector = LanguageDetector(confidence_threshold=0.5)
        
        test_texts = [
            "Hello world, this is a test sentence for language detection.",
            "안녕하세요, 이것은 언어 감지를 위한 테스트 문장입니다.",
            "こんにちは、これは言語検出のためのテスト文です。",
            "你好，这是用于语言检测的测试句子。"
        ]
        
        import time
        
        # Warm up
        for text in test_texts:
            detector.detect_text_language(text)
        
        # Benchmark
        start_time = time.time()
        results = []
        for text in test_texts:
            result = detector.detect_text_language(text)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(test_texts)
        
        print(f"\nLanguage Detection Performance:")
        print(f"Total time for {len(test_texts)} detections: {total_time:.4f}s")
        print(f"Average time per detection: {avg_time:.4f}s")
        print(f"Detections per second: {1/avg_time:.2f}")
        
        # Assert reasonable performance
        self.assertLess(avg_time, 1.0)  # Should be faster than 1 second per detection
    
    def test_batch_detection_performance(self):
        """Benchmark batch language detection performance."""
        detector = LanguageDetector(confidence_threshold=0.5)
        
        # Create larger test dataset
        test_texts = ["Hello world"] * 100
        
        import time
        
        # Benchmark batch detection
        start_time = time.time()
        results = detector.batch_detect(test_texts)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / len(test_texts)
        
        print(f"\nBatch Detection Performance:")
        print(f"Total time for {len(test_texts)} batch detections: {total_time:.4f}s")
        print(f"Average time per detection: {avg_time:.4f}s")
        print(f"Detections per second: {1/avg_time:.2f}")
        
        self.assertEqual(len(results), len(test_texts))


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestLanguageDetector,
        TestUnifiedTranslator,
        TestEdgeQuantization,
        TestMobileDeployment,
        TestMultimodalCorpusPreparation,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    if result.failures or result.errors:
        exit(1)
    else:
        exit(0)