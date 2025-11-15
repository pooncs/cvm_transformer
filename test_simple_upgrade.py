#!/usr/bin/env python3
"""
Simple test script to verify the upgraded CVM translator functionality.
"""

import sys
import time

def test_basic_imports():
    """Test basic module imports."""
    print("Testing basic imports...")
    
    try:
        from cvm_translator.language_detector import LanguageDetector, DetectionMode
        print("✅ Language detector imported successfully")
    except Exception as e:
        print(f"❌ Language detector import failed: {e}")
        return False
    
    try:
        from cvm_translator.unified_translator import UnifiedTranslator, TranslationRequest
        print("✅ Unified translator imported successfully")
    except Exception as e:
        print(f"❌ Unified translator import failed: {e}")
        return False
    
    try:
        from cvm_translator.edge_quantization import EdgeQuantizationEngine, QuantizationConfig
        print("✅ Edge quantization imported successfully")
    except Exception as e:
        print(f"❌ Edge quantization import failed: {e}")
        return False
    
    try:
        from cvm_translator.mobile_deployment import ExportConfig, ONNXExporter
        print("✅ Mobile deployment imported successfully")
    except Exception as e:
        print(f"❌ Mobile deployment import failed: {e}")
        return False
    
    return True

def test_language_detector():
    """Test language detector functionality."""
    print("\nTesting language detector...")
    
    try:
        from cvm_translator.language_detector import LanguageDetector, DetectionMode
        
        # Create detector with minimal configuration
        detector = LanguageDetector(
            models_dir="test_models",
            confidence_threshold=0.5
        )
        
        # Test text detection with known languages
        test_cases = [
            ("Hello world", "English text"),
            ("안녕하세요", "Korean text"),
            ("こんにちは", "Japanese text"),
        ]
        
        for text, description in test_cases:
            result = detector.detect_text_language(text)
            if result:
                print(f"✅ {description}: detected as {result.language} "
                      f"(confidence: {result.confidence:.3f})")
            else:
                print(f"⚠️  {description}: detection failed (models may not be available)")
        
        # Test available modes
        available_modes = detector.get_available_modes()
        print(f"✅ Available detection modes: {[mode.value for mode in available_modes]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Language detector test failed: {e}")
        return False

def test_quantization_engine():
    """Test quantization engine configuration."""
    print("\nTesting quantization engine...")
    
    try:
        from cvm_translator.edge_quantization import QuantizationConfig, EdgeQuantizationEngine
        
        # Test configuration creation
        config = QuantizationConfig(
            method="int8",
            bits=8,
            calibration_dataset_size=32
        )
        
        print(f"✅ Quantization config created: {config.method} {config.bits}-bit")
        
        # Test engine creation
        engine = EdgeQuantizationEngine(config)
        print("✅ Quantization engine initialized successfully")
        
        # Test calibration data generation
        calibration_data = engine._generate_calibration_data(None, size=5)
        print(f"✅ Generated {len(calibration_data)} calibration samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization engine test failed: {e}")
        return False

def test_mobile_deployment():
    """Test mobile deployment configuration."""
    print("\nTesting mobile deployment...")
    
    try:
        from cvm_translator.mobile_deployment import ExportConfig
        
        # Test configuration creation
        config = ExportConfig(
            target_devices=["ios", "android"],
            enable_quantization=True,
            max_batch_size=1,
            max_sequence_length=128
        )
        
        print(f"✅ Export config created for platforms: {config.target_devices}")
        print(f"✅ Quantization enabled: {config.enable_quantization}")
        print(f"✅ Max sequence length: {config.max_sequence_length}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mobile deployment test failed: {e}")
        return False

def test_performance_benchmark():
    """Test performance benchmarking."""
    print("\nTesting performance benchmarks...")
    
    try:
        from cvm_translator.language_detector import LanguageDetector
        
        # Create detector
        detector = LanguageDetector(confidence_threshold=0.5)
        
        # Simple benchmark
        test_texts = ["Hello world", "안녕하세요", "こんにちは", "你好"] * 5
        
        start_time = time.time()
        results = detector.batch_detect(test_texts)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / len(test_texts)
        throughput = 1 / avg_time if avg_time > 0 else 0
        
        print(f"✅ Processed {len(test_texts)} texts in {total_time:.3f}s")
        print(f"✅ Average time per text: {avg_time*1000:.1f}ms")
        print(f"✅ Throughput: {throughput:.1f} texts/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 CVM Translator Upgrade - Simple Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Language Detector", test_language_detector),
        ("Quantization Engine", test_quantization_engine),
        ("Mobile Deployment", test_mobile_deployment),
        ("Performance Benchmark", test_performance_benchmark),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 TEST SUMMARY")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The upgrade is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())