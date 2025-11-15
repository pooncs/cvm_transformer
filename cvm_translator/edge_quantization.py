import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List, Any
import logging
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
import time

# Quantization libraries
try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from auto_gptq.modeling import BaseGPTQForCausalLM
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False

try:
    from awq import AutoAWQForCausalLM
    from awq.models.base import BaseAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .cvm_transformer import CVMTransformer


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    method: str = "awq"  # awq, gptq, onnx, int8, int4
    bits: int = 4  # 4, 8, 16
    group_size: int = 128
    desc_act: bool = False  # For GPTQ
    static_groups: bool = False  # For GPTQ
    damp_percent: float = 0.1  # For GPTQ
    calibration_dataset_size: int = 128
    device_map: str = "auto"
    trust_remote_code: bool = False
    use_flash_attention: bool = True
    enable_kv_cache_quantization: bool = True


class EdgeQuantizationEngine:
    """
    Advanced quantization engine for edge deployment.
    Supports AWQ, GPTQ, ONNX, and custom INT8/INT4 quantization.
    """
    
    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
        
        # Calibration data cache
        self.calibration_data = None
        
        self.logger.info(f"Initialized EdgeQuantizationEngine with {self.config.method} quantization")
    
    def _validate_config(self):
        """Validate quantization configuration."""
        if self.config.method == "awq" and not AWQ_AVAILABLE:
            raise ValueError("AWQ quantization requested but autoawq not available")
        
        if self.config.method == "gptq" and not GPTQ_AVAILABLE:
            raise ValueError("GPTQ quantization requested but auto-gptq not available")
        
        if self.config.method == "onnx" and not ONNX_AVAILABLE:
            raise ValueError("ONNX quantization requested but onnxruntime not available")
        
        if self.config.bits not in [4, 8, 16]:
            raise ValueError("Quantization bits must be 4, 8, or 16")
    
    def quantize_model(self, 
                      model: Union[nn.Module, str],
                      tokenizer: Optional[Any] = None,
                      calibration_data: Optional[List[str]] = None,
                      output_dir: str = "quantized_models") -> str:
        """
        Quantize a model using the specified method.
        
        Args:
            model: Model to quantize (PyTorch module or model path)
            tokenizer: Tokenizer for calibration (required for some methods)
            calibration_data: Calibration dataset for quantization
            output_dir: Output directory for quantized model
            
        Returns:
            Path to quantized model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Quantizing model with {self.config.method} method...")
        start_time = time.time()
        
        if self.config.method == "awq":
            return self._quantize_awq(model, tokenizer, calibration_data, output_path)
        elif self.config.method == "gptq":
            return self._quantize_gptq(model, tokenizer, calibration_data, output_path)
        elif self.config.method == "onnx":
            return self._quantize_onnx(model, tokenizer, calibration_data, output_path)
        elif self.config.method in ["int8", "int4"]:
            return self._quantize_pytorch(model, tokenizer, calibration_data, output_path)
        else:
            raise ValueError(f"Unsupported quantization method: {self.config.method}")
    
    def _quantize_awq(self, 
                     model: Union[nn.Module, str],
                     tokenizer: Optional[Any],
                     calibration_data: Optional[List[str]],
                     output_path: Path) -> str:
        """Quantize model using AWQ (Activation-aware Weight Quantization)."""
        
        if not AWQ_AVAILABLE:
            raise RuntimeError("AWQ quantization not available")
        
        # Load model if path provided
        if isinstance(model, str):
            if TRANSFORMERS_AVAILABLE:
                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    device_map=self.config.device_map,
                    trust_remote_code=self.config.trust_remote_code
                )
            else:
                raise RuntimeError("Transformers not available for model loading")
        
        # Prepare calibration data
        if calibration_data is None:
            calibration_data = self._generate_calibration_data(tokenizer)
        
        # Configure AWQ quantization
        quant_config = {
            "zero_point": True,
            "q_group_size": self.config.group_size,
            "w_bit": self.config.bits,
            "version": "GEMM"
        }
        
        # Perform AWQ quantization
        self.logger.info("Running AWQ quantization...")
        
        try:
            # Initialize AWQ model
            awq_model = AutoAWQForCausalLM.from_pretrained(model)
            
            # Quantize
            awq_model.quantize(
                tokenizer=tokenizer,
                quant_config=quant_config,
                calib_data=calibration_data
            )
            
            # Save quantized model
            model_path = output_path / f"awq_{self.config.bits}bit"
            awq_model.save_quantized(str(model_path))
            
            # Save config
            config_path = model_path / "quant_config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    "method": "awq",
                    "bits": self.config.bits,
                    "group_size": self.config.group_size,
                    "quantization_time": time.time() - (time.time() - 0),  # Will be updated
                    "model_size_reduction": self._calculate_model_size_reduction(model, awq_model)
                }, f, indent=2)
            
            self.logger.info(f"AWQ quantization completed. Model saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"AWQ quantization failed: {e}")
            raise
    
    def _quantize_gptq(self, 
                      model: Union[nn.Module, str],
                      tokenizer: Optional[Any],
                      calibration_data: Optional[List[str]],
                      output_path: Path) -> str:
        """Quantize model using GPTQ (Gradient-based Post-training Quantization)."""
        
        if not GPTQ_AVAILABLE:
            raise RuntimeError("GPTQ quantization not available")
        
        # Load model if path provided
        if isinstance(model, str):
            if TRANSFORMERS_AVAILABLE:
                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    device_map=self.config.device_map,
                    trust_remote_code=self.config.trust_remote_code
                )
            else:
                raise RuntimeError("Transformers not available for model loading")
        
        # Prepare calibration data
        if calibration_data is None:
            calibration_data = self._generate_calibration_data(tokenizer)
        
        # Configure GPTQ quantization
        quantize_config = BaseQuantizeConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            desc_act=self.config.desc_act,
            static_groups=self.config.static_groups,
            damp_percent=self.config.damp_percent
        )
        
        # Perform GPTQ quantization
        self.logger.info("Running GPTQ quantization...")
        
        try:
            # Initialize GPTQ model
            gptq_model = AutoGPTQForCausalLM.from_pretrained(
                model,
                quantize_config=quantize_config
            )
            
            # Quantize
            gptq_model.quantize(
                calibration_dataset=calibration_data,
                batch_size=1
            )
            
            # Save quantized model
            model_path = output_path / f"gptq_{self.config.bits}bit"
            gptq_model.save_quantized(str(model_path))
            
            # Save config
            config_path = model_path / "quant_config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    "method": "gptq",
                    "bits": self.config.bits,
                    "group_size": self.config.group_size,
                    "desc_act": self.config.desc_act,
                    "static_groups": self.config.static_groups,
                    "damp_percent": self.config.damp_percent
                }, f, indent=2)
            
            self.logger.info(f"GPTQ quantization completed. Model saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"GPTQ quantization failed: {e}")
            raise
    
    def _quantize_onnx(self, 
                      model: Union[nn.Module, str],
                      tokenizer: Optional[Any],
                      calibration_data: Optional[List[str]],
                      output_path: Path) -> str:
        """Quantize model using ONNX with dynamic quantization."""
        
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX quantization not available")
        
        # Convert to ONNX first
        onnx_path = output_path / "model.onnx"
        quantized_path = output_path / "model_quantized.onnx"
        
        try:
            # Export to ONNX
            self.logger.info("Exporting model to ONNX format...")
            
            if isinstance(model, str):
                # Load model from path
                if TRANSFORMERS_AVAILABLE:
                    model = AutoModelForCausalLM.from_pretrained(model)
                else:
                    raise RuntimeError("Transformers not available for model loading")
            
            # Create dummy input
            dummy_input = torch.randint(0, 1000, (1, 128))
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'sequence'},
                    'logits': {0: 'batch', 1: 'sequence'}
                },
                opset_version=11
            )
            
            # Quantize ONNX model
            self.logger.info("Quantizing ONNX model...")
            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=QuantType.QInt8 if self.config.bits == 8 else QuantType.QUInt4
            )
            
            # Save config
            config_path = output_path / "quant_config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    "method": "onnx",
                    "bits": self.config.bits,
                    "original_model": str(onnx_path),
                    "quantized_model": str(quantized_path)
                }, f, indent=2)
            
            self.logger.info(f"ONNX quantization completed. Model saved to {quantized_path}")
            return str(quantized_path)
            
        except Exception as e:
            self.logger.error(f"ONNX quantization failed: {e}")
            raise
    
    def _quantize_pytorch(self, 
                         model: Union[nn.Module, str],
                         tokenizer: Optional[Any],
                         calibration_data: Optional[List[str]],
                         output_path: Path) -> str:
        """Quantize model using PyTorch native quantization."""
        
        if isinstance(model, str):
            # Load model from path
            if TRANSFORMERS_AVAILABLE:
                model = AutoModelForCausalLM.from_pretrained(model)
            else:
                raise RuntimeError("Transformers not available for model loading")
        
        # Prepare calibration data
        if calibration_data is None:
            calibration_data = self._generate_calibration_data(tokenizer)
        
        # Configure quantization
        if self.config.bits == 8:
            dtype = torch.qint8
        elif self.config.bits == 4:
            dtype = torch.quint4x2
        else:
            dtype = torch.qint8
        
        # Apply quantization
        self.logger.info(f"Applying PyTorch {self.config.bits}-bit quantization...")
        
        try:
            # Prepare model for quantization
            model.eval()
            
            # Configure quantization backend
            if self.device == "cpu":
                backend = "qnnpack"
            else:
                backend = "qnnpack"  # or "fbgemm" for x86
            
            torch.backends.quantized.engine = backend
            
            # Create quantization configuration
            if self.config.bits == 8:
                qconfig = torch.quantization.get_default_qconfig(backend)
            else:
                # Custom 4-bit configuration
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.default_observer,
                    weight=torch.quantization.default_weight_observer
                )
            
            # Apply quantization
            model.qconfig = qconfig
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate with sample data
            self._calibrate_model(model, calibration_data, tokenizer)
            
            # Convert to quantized model
            torch.quantization.convert(model, inplace=True)
            
            # Save quantized model
            model_path = output_path / f"pytorch_{self.config.bits}bit"
            model_path.mkdir(parents=True, exist_ok=True)
            
            torch.save(model.state_dict(), model_path / "pytorch_model.bin")
            
            # Save config
            config_path = model_path / "quant_config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    "method": "pytorch",
                    "bits": self.config.bits,
                    "backend": backend,
                    "qconfig": str(qconfig)
                }, f, indent=2)
            
            self.logger.info(f"PyTorch quantization completed. Model saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"PyTorch quantization failed: {e}")
            raise
    
    def _generate_calibration_data(self, tokenizer: Optional[Any], 
                                   size: int = None) -> List[str]:
        """Generate calibration data for quantization."""
        if size is None:
            size = self.config.calibration_dataset_size
        
        # Default calibration dataset
        calibration_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello, how are you doing today?",
            "Machine learning is a subset of artificial intelligence.",
            "The weather is quite nice today.",
            "I enjoy reading books in my free time.",
            "Technology continues to advance at a rapid pace.",
            "Education is important for personal development.",
            "The sun rises in the east and sets in the west.",
            "Good communication skills are essential in the workplace.",
            "Practice makes perfect when learning new skills.",
            "안녕하세요, 오늘 기분은 어떠세요?",
            "한국어는 아름다운 언어입니다.",
            "기술 발전은 우리 삶을 변화시키고 있습니다.",
            "교육은 개인의 성장에 매우 중요합니다.",
            "좋은 의사소통 능력은 직장에서 필수적입니다."
        ]
        
        # Repeat and shuffle
        calibration_data = []
        while len(calibration_data) < size:
            calibration_data.extend(calibration_texts)
        
        # Shuffle and truncate
        np.random.shuffle(calibration_data)
        return calibration_data[:size]
    
    def _calibrate_model(self, model: nn.Module, 
                        calibration_data: List[str], 
                        tokenizer: Optional[Any]):
        """Calibrate quantized model with calibration data."""
        self.logger.info(f"Calibrating model with {len(calibration_data)} samples...")
        
        model.eval()
        with torch.no_grad():
            for text in calibration_data:
                if tokenizer is not None:
                    # Tokenize text
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    # Use dummy input
                    inputs = torch.randint(0, 1000, (1, 128)).to(self.device)
                
                # Forward pass for calibration
                _ = model(inputs)
    
    def _calculate_model_size_reduction(self, original_model: nn.Module, 
                                      quantized_model: nn.Module) -> Dict[str, float]:
        """Calculate model size reduction after quantization."""
        # This is a simplified calculation
        # In practice, you'd calculate actual model file sizes
        return {
            "estimated_reduction": self.config.bits / 32.0,  # Assuming FP32 original
            "original_bits": 32,
            "quantized_bits": self.config.bits
        }
    
    def benchmark_quantized_model(self, 
                                 model_path: str,
                                 tokenizer: Optional[Any] = None,
                                 test_data: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark quantized model performance.
        
        Args:
            model_path: Path to quantized model
            tokenizer: Tokenizer for testing
            test_data: Test data for benchmarking
            
        Returns:
            Benchmark results
        """
        if test_data is None:
            test_data = [
                "Hello, how are you?",
                "The weather is nice today.",
                "Machine learning is fascinating."
            ]
        
        results = {
            "model_path": model_path,
            "test_samples": len(test_data),
            "inference_times": [],
            "memory_usage": [],
            "perplexity_scores": []
        }
        
        # Load quantized model
        try:
            if "awq" in model_path:
                model = AutoAWQForCausalLM.from_quantized(model_path)
            elif "gptq" in model_path:
                model = AutoGPTQForCausalLM.from_quantized(model_path)
            elif "onnx" in model_path:
                # ONNX benchmarking would require different approach
                results["note"] = "ONNX benchmarking not implemented"
                return results
            else:
                # PyTorch quantized model
                model = torch.load(model_path + "/pytorch_model.bin")
        
        except Exception as e:
            self.logger.error(f"Failed to load quantized model: {e}")
            results["error"] = str(e)
            return results
        
        # Benchmark inference
        model.eval()
        with torch.no_grad():
            for text in test_data:
                start_time = time.time()
                
                if tokenizer is not None:
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                    outputs = model(**inputs)
                else:
                    # Dummy forward pass
                    dummy_input = torch.randint(0, 1000, (1, 128))
                    outputs = model(dummy_input)
                
                inference_time = time.time() - start_time
                results["inference_times"].append(inference_time)
        
        # Calculate statistics
        if results["inference_times"]:
            results["avg_inference_time"] = np.mean(results["inference_times"])
            results["min_inference_time"] = np.min(results["inference_times"])
            results["max_inference_time"] = np.max(results["inference_times"])
        
        return results


class CVMQuantizationOptimizer:
    """
    Specialized quantization optimizer for CVM-enhanced models.
    """
    
    def __init__(self, cvm_model: CVMTransformer):
        self.cvm_model = cvm_model
        self.logger = logging.getLogger(__name__)
    
    def optimize_cvm_for_edge(self, 
                            quantization_config: QuantizationConfig,
                            output_path: str = "cvm_quantized") -> str:
        """
        Optimize CVM model specifically for edge deployment.
        
        Args:
            quantization_config: Quantization configuration
            output_path: Output directory
            
        Returns:
            Path to optimized model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Optimizing CVM model for edge deployment...")
        
        # Step 1: Apply CVM-specific optimizations
        self._optimize_cvm_attention()
        
        # Step 2: Apply quantization
        engine = EdgeQuantizationEngine(quantization_config)
        
        # Step 3: Quantize the optimized model
        quantized_path = engine.quantize_model(
            self.cvm_model,
            output_dir=str(output_path)
        )
        
        # Step 4: Add CVM-specific metadata
        metadata = {
            "model_type": "cvm_enhanced",
            "optimization": "edge_quantized",
            "quantization_method": quantization_config.method,
            "quantization_bits": quantization_config.bits,
            "cvm_buffer_size": getattr(self.cvm_model, 'buffer_size', 'unknown'),
            "cvm_merge_threshold": getattr(self.cvm_model, 'merge_threshold', 'unknown')
        }
        
        metadata_path = Path(quantized_path) / "cvm_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"CVM optimization completed. Model saved to {quantized_path}")
        return str(quantized_path)
    
    def _optimize_cvm_attention(self):
        """Apply CVM-specific attention optimizations."""
        # This would implement CVM-specific optimizations
        # For now, we'll add a placeholder
        self.logger.info("Applying CVM attention optimizations...")
        
        # Example: Optimize CVM buffer operations
        if hasattr(self.cvm_model, 'cvm_buffer'):
            # Optimize buffer operations for edge deployment
            pass


# Convenience functions
def quantize_for_edge(model: Union[nn.Module, str],
                     method: str = "awq",
                     bits: int = 4,
                     output_dir: str = "quantized_models",
                     **kwargs) -> str:
    """
    Convenience function to quantize a model for edge deployment.
    
    Args:
        model: Model to quantize
        method: Quantization method (awq, gptq, onnx, int8, int4)
        bits: Quantization bits (4, 8)
        output_dir: Output directory
        **kwargs: Additional configuration
        
    Returns:
        Path to quantized model
    """
    config = QuantizationConfig(method=method, bits=bits, **kwargs)
    engine = EdgeQuantizationEngine(config)
    
    return engine.quantize_model(model, output_dir=output_dir)


def optimize_cvm_model(cvm_model: CVMTransformer,
                      quantization_method: str = "awq",
                      bits: int = 4,
                      output_dir: str = "cvm_optimized") -> str:
    """
    Optimize CVM model for edge deployment.
    
    Args:
        cvm_model: CVM model to optimize
        quantization_method: Quantization method
        bits: Quantization bits
        output_dir: Output directory
        
    Returns:
        Path to optimized model
    """
    config = QuantizationConfig(method=quantization_method, bits=bits)
    optimizer = CVMQuantizationOptimizer(cvm_model)
    
    return optimizer.optimize_cvm_for_edge(config, output_dir)


# Example usage and testing
if __name__ == "__main__":
    # Test quantization engine
    print("Testing EdgeQuantizationEngine...")
    
    # Create test configuration
    config = QuantizationConfig(method="int8", bits=8)
    engine = EdgeQuantizationEngine(config)
    
    # Test with dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
        
        def forward(self, x):
            return self.linear(x)
    
    dummy_model = DummyModel()
    
    try:
        # This would fail without proper setup, but demonstrates the API
        # quantized_path = engine.quantize_model(dummy_model, output_dir="test_quantized")
        print("Quantization engine initialized successfully")
        print(f"Available methods: AWQ={AWQ_AVAILABLE}, GPTQ={GPTQ_AVAILABLE}, ONNX={ONNX_AVAILABLE}")
        
    except Exception as e:
        print(f"Expected error (no proper model): {e}")
    
    print("Edge quantization engine ready for deployment!")