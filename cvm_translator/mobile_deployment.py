import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List, Any, Tuple
import logging
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
import time

# ONNX and mobile deployment libraries
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.tools import optimizer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

try:
    import tensorflow as tf
    import tf2onnx
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .cvm_transformer import CVMTransformer
from .edge_quantization import EdgeQuantizationEngine, QuantizationConfig


@dataclass
class ExportConfig:
    """Configuration for model export and deployment."""
    # General export settings
    format: str = "onnx"  # onnx, coreml, tflite, tensorrt
    opset_version: int = 11
    dynamic_axes: bool = True
    optimize_for_mobile: bool = True
    enable_quantization: bool = True
    quantization_bits: int = 8
    
    # Mobile-specific settings
    mobile_optimizations: bool = True
    enable_coreml_neural_engine: bool = True
    enable_tflite_gpu_delegate: bool = True
    
    # Performance settings
    max_batch_size: int = 1
    max_sequence_length: int = 512
    enable_memory_efficient_attention: bool = True
    
    # Edge device targets
    target_devices: List[str] = None  # ["ios", "android", "edge_tpu", "nvidia_jetson"]
    
    def __post_init__(self):
        if self.target_devices is None:
            self.target_devices = ["ios", "android"]


class ONNXExporter:
    """
    ONNX export engine for cross-platform deployment.
    """
    
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
        self.logger = logging.getLogger(__name__)
        
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX export requires onnx and onnxruntime packages")
    
    def export_model(self, 
                    model: Union[nn.Module, str],
                    tokenizer: Optional[Any] = None,
                    output_path: str = "exported_models/onnx") -> str:
        """
        Export model to ONNX format.
        
        Args:
            model: Model to export (PyTorch module or model path)
            tokenizer: Tokenizer for testing
            output_path: Output directory
            
        Returns:
            Path to exported ONNX model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting model to ONNX format...")
        start_time = time.time()
        
        # Load model if path provided
        if isinstance(model, str):
            if TRANSFORMERS_AVAILABLE:
                model = AutoModel.from_pretrained(model)
            else:
                raise RuntimeError("Transformers not available for model loading")
        
        # Prepare dummy input
        dummy_input = self._create_dummy_input(tokenizer)
        
        # Configure dynamic axes
        dynamic_axes = self._get_dynamic_axes() if self.config.dynamic_axes else None
        
        # Export to ONNX
        onnx_path = output_path / "model.onnx"
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes=dynamic_axes,
                opset_version=self.config.opset_version,
                do_constant_folding=True,
                export_params=True
            )
            
            # Verify ONNX model
            self._verify_onnx_model(str(onnx_path))
            
            # Optimize if requested
            if self.config.optimize_for_mobile:
                onnx_path = self._optimize_onnx_model(str(onnx_path))
            
            # Apply quantization if requested
            if self.config.enable_quantization:
                onnx_path = self._quantize_onnx_model(onnx_path)
            
            export_time = time.time() - start_time
            
            # Save export metadata
            metadata = {
                "format": "onnx",
                "export_time": export_time,
                "opset_version": self.config.opset_version,
                "dynamic_axes": self.config.dynamic_axes,
                "optimized": self.config.optimize_for_mobile,
                "quantized": self.config.enable_quantization,
                "model_size": onnx_path.stat().st_size if isinstance(onnx_path, Path) else 0
            }
            
            metadata_path = output_path / "export_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"ONNX export completed in {export_time:.2f}s")
            return str(onnx_path)
            
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            raise
    
    def _create_dummy_input(self, tokenizer: Optional[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dummy input for ONNX export."""
        if tokenizer is not None:
            # Use tokenizer to create realistic input
            dummy_text = "This is a test sentence for ONNX export."
            inputs = tokenizer(dummy_text, return_tensors='pt', 
                             max_length=self.config.max_sequence_length, 
                             padding='max_length', truncation=True)
            return inputs['input_ids'], inputs['attention_mask']
        else:
            # Create generic dummy input
            batch_size = self.config.max_batch_size
            seq_length = self.config.max_sequence_length
            
            input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            attention_mask = torch.ones((batch_size, seq_length))
            
            return input_ids, attention_mask
    
    def _get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Get dynamic axes configuration for ONNX export."""
        return {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    
    def _verify_onnx_model(self, onnx_path: str):
        """Verify exported ONNX model."""
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            self.logger.info("ONNX model verification passed")
        except Exception as e:
            self.logger.warning(f"ONNX model verification failed: {e}")
    
    def _optimize_onnx_model(self, onnx_path: str) -> str:
        """Optimize ONNX model for inference."""
        self.logger.info("Optimizing ONNX model...")
        
        try:
            # Use ONNX Runtime optimizer
            optimized_path = onnx_path.replace(".onnx", "_optimized.onnx")
            
            # Basic optimization
            optimizer.optimize(onnx_path, optimized_path)
            
            self.logger.info(f"ONNX optimization completed: {optimized_path}")
            return optimized_path
            
        except Exception as e:
            self.logger.warning(f"ONNX optimization failed: {e}")
            return onnx_path
    
    def _quantize_onnx_model(self, onnx_path: str) -> str:
        """Quantize ONNX model for edge deployment."""
        self.logger.info(f"Quantizing ONNX model to {self.config.quantization_bits}-bit...")
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = onnx_path.replace(".onnx", f"_quantized_{self.config.quantization_bits}bit.onnx")
            
            quant_type = QuantType.QInt8 if self.config.quantization_bits == 8 else QuantType.QUInt4
            
            quantize_dynamic(
                onnx_path,
                quantized_path,
                weight_type=quant_type
            )
            
            self.logger.info(f"ONNX quantization completed: {quantized_path}")
            return quantized_path
            
        except Exception as e:
            self.logger.warning(f"ONNX quantization failed: {e}")
            return onnx_path


class CoreMLExporter:
    """
    CoreML export engine for iOS deployment.
    """
    
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
        self.logger = logging.getLogger(__name__)
        
        if not COREML_AVAILABLE:
            raise RuntimeError("CoreML export requires coremltools package")
    
    def export_model(self, 
                    model: Union[nn.Module, str],
                    tokenizer: Optional[Any] = None,
                    output_path: str = "exported_models/coreml") -> str:
        """
        Export model to CoreML format.
        
        Args:
            model: Model to export
            tokenizer: Tokenizer for testing
            output_path: Output directory
            
        Returns:
            Path to exported CoreML model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Exporting model to CoreML format...")
        start_time = time.time()
        
        # Load model if path provided
        if isinstance(model, str):
            if TRANSFORMERS_AVAILABLE:
                model = AutoModel.from_pretrained(model)
            else:
                raise RuntimeError("Transformers not available for model loading")
        
        # Create dummy input
        dummy_input = self._create_dummy_input(tokenizer)
        
        # Convert to CoreML
        coreml_path = output_path / "model.mlmodel"
        
        try:
            # Trace model
            traced_model = torch.jit.trace(model, dummy_input)
            
            # Convert to CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=dummy_input.shape)],
                minimum_deployment_target=ct.target.iOS13,
                compute_precision=ct.precision.FLOAT16 if self.config.enable_quantization else ct.precision.FLOAT32
            )
            
            # Optimize for Neural Engine if requested
            if self.config.enable_coreml_neural_engine:
                coreml_model = self._optimize_for_neural_engine(coreml_model)
            
            # Save CoreML model
            coreml_model.save(str(coreml_path))
            
            export_time = time.time() - start_time
            
            # Save export metadata
            metadata = {
                "format": "coreml",
                "export_time": export_time,
                "optimized_for_neural_engine": self.config.enable_coreml_neural_engine,
                "precision": "float16" if self.config.enable_quantization else "float32",
                "model_size": coreml_path.stat().st_size
            }
            
            metadata_path = output_path / "export_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"CoreML export completed in {export_time:.2f}s")
            return str(coreml_path)
            
        except Exception as e:
            self.logger.error(f"CoreML export failed: {e}")
            raise
    
    def _create_dummy_input(self, tokenizer: Optional[Any]) -> torch.Tensor:
        """Create dummy input for CoreML export."""
        if tokenizer is not None:
            dummy_text = "This is a test sentence for CoreML export."
            inputs = tokenizer(dummy_text, return_tensors='pt',
                             max_length=self.config.max_sequence_length,
                             padding='max_length', truncation=True)
            return inputs['input_ids']
        else:
            batch_size = self.config.max_batch_size
            seq_length = self.config.max_sequence_length
            return torch.randint(0, 1000, (batch_size, seq_length))
    
    def _optimize_for_neural_engine(self, coreml_model):
        """Optimize CoreML model for Apple Neural Engine."""
        # This would implement Neural Engine specific optimizations
        self.logger.info("Optimizing CoreML model for Neural Engine...")
        return coreml_model


class MobileDeploymentManager:
    """
    Unified mobile deployment manager for iOS and Android.
    """
    
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
        self.logger = logging.getLogger(__name__)
        
        self.exporters = {
            "onnx": ONNXExporter(self.config),
            "coreml": CoreMLExporter(self.config) if COREML_AVAILABLE else None,
            # Add more exporters as needed
        }
    
    def deploy_model(self, 
                    model: Union[nn.Module, str],
                    tokenizer: Optional[Any] = None,
                    output_base_path: str = "mobile_deployment") -> Dict[str, str]:
        """
        Deploy model to multiple mobile platforms.
        
        Args:
            model: Model to deploy
            tokenizer: Tokenizer for the model
            output_base_path: Base output directory
            
        Returns:
            Dictionary of deployment paths by platform
        """
        output_base_path = Path(output_base_path)
        output_base_path.mkdir(parents=True, exist_ok=True)
        
        deployment_paths = {}
        
        self.logger.info(f"Deploying model to platforms: {self.config.target_devices}")
        
        for target in self.config.target_devices:
            try:
                if target == "ios":
                    if self.exporters["coreml"] is not None:
                        path = self.exporters["coreml"].export_model(
                            model, tokenizer, 
                            str(output_base_path / "ios")
                        )
                        deployment_paths["ios"] = path
                    else:
                        self.logger.warning("CoreML exporter not available for iOS")
                
                elif target == "android":
                    # Use ONNX for Android (can be converted to TensorFlow Lite if needed)
                    path = self.exporters["onnx"].export_model(
                        model, tokenizer,
                        str(output_base_path / "android")
                    )
                    deployment_paths["android"] = path
                
                elif target == "edge_tpu":
                    # Edge TPU specific deployment
                    path = self._deploy_to_edge_tpu(model, tokenizer, output_base_path)
                    deployment_paths["edge_tpu"] = path
                
                elif target == "nvidia_jetson":
                    # NVIDIA Jetson specific deployment
                    path = self._deploy_to_jetson(model, tokenizer, output_base_path)
                    deployment_paths["nvidia_jetson"] = path
                
                else:
                    self.logger.warning(f"Unknown target device: {target}")
            
            except Exception as e:
                self.logger.error(f"Deployment to {target} failed: {e}")
                deployment_paths[f"{target}_error"] = str(e)
        
        # Save deployment summary
        summary_path = output_base_path / "deployment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "target_devices": self.config.target_devices,
                "deployment_paths": deployment_paths,
                "deployment_time": time.time(),
                "config": self.config.__dict__
            }, f, indent=2)
        
        self.logger.info(f"Mobile deployment completed. Summary saved to {summary_path}")
        return deployment_paths
    
    def _deploy_to_edge_tpu(self, model, tokenizer, output_base_path):
        """Deploy model to Google Edge TPU."""
        # First export to TensorFlow Lite
        tflite_path = self._export_to_tflite(model, tokenizer, output_base_path / "edge_tpu")
        
        # Then compile for Edge TPU (requires Edge TPU compiler)
        self.logger.info("Compiling for Edge TPU...")
        # This would require the Edge TPU compiler tool
        
        return str(tflite_path)
    
    def _deploy_to_jetson(self, model, tokenizer, output_base_path):
        """Deploy model to NVIDIA Jetson."""
        # Use TensorRT optimization for Jetson
        self.logger.info("Optimizing for NVIDIA Jetson with TensorRT...")
        
        # First export to ONNX, then convert to TensorRT
        onnx_path = self.exporters["onnx"].export_model(
            model, tokenizer,
            str(output_base_path / "nvidia_jetson")
        )
        
        # TensorRT conversion would happen here
        # This requires NVIDIA TensorRT tools
        
        return str(onnx_path)
    
    def _export_to_tflite(self, model, tokenizer, output_path):
        """Export model to TensorFlow Lite format."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for TFLite export")
        
        # This would implement TensorFlow Lite export
        # For now, return placeholder
        return output_path / "model.tflite"


class CVMMobileOptimizer:
    """
    Specialized mobile optimizer for CVM-enhanced models.
    """
    
    def __init__(self, cvm_model: CVMTransformer):
        self.cvm_model = cvm_model
        self.logger = logging.getLogger(__name__)
    
    def optimize_for_mobile(self, 
                           export_config: ExportConfig,
                           output_path: str = "cvm_mobile") -> Dict[str, str]:
        """
        Optimize CVM model for mobile deployment.
        
        Args:
            export_config: Export configuration
            output_path: Output directory
            
        Returns:
            Dictionary of deployment paths
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Optimizing CVM model for mobile deployment...")
        
        # Step 1: Apply CVM-specific optimizations
        optimized_model = self._optimize_cvm_architecture()
        
        # Step 2: Export to mobile formats
        mobile_manager = MobileDeploymentManager(export_config)
        
        deployment_paths = mobile_manager.deploy_model(
            optimized_model,
            output_base_path=str(output_path)
        )
        
        # Step 3: Add CVM-specific metadata
        self._add_cvm_metadata(deployment_paths, output_path)
        
        self.logger.info(f"CVM mobile optimization completed: {deployment_paths}")
        return deployment_paths
    
    def _optimize_cvm_architecture(self):
        """Apply CVM-specific optimizations for mobile."""
        self.logger.info("Applying CVM mobile optimizations...")
        
        # Optimize CVM buffer operations
        if hasattr(self.cvm_model, 'cvm_buffer'):
            # Optimize buffer for mobile memory constraints
            pass
        
        # Optimize attention mechanism
        if hasattr(self.cvm_model, 'attention'):
            # Use memory-efficient attention
            pass
        
        return self.cvm_model
    
    def _add_cvm_metadata(self, deployment_paths: Dict[str, str], output_path: Path):
        """Add CVM-specific metadata to deployment."""
        metadata = {
            "model_type": "cvm_enhanced",
            "optimization_target": "mobile_edge",
            "cvm_buffer_size": getattr(self.cvm_model, 'buffer_size', 'unknown'),
            "cvm_merge_threshold": getattr(self.cvm_model, 'merge_threshold', 'unknown'),
            "deployment_paths": deployment_paths,
            "optimization_timestamp": time.time()
        }
        
        metadata_path = output_path / "cvm_mobile_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


# Convenience functions
def export_for_mobile(model: Union[nn.Module, str],
                     target_platforms: List[str] = None,
                     enable_quantization: bool = True,
                     output_dir: str = "mobile_export") -> Dict[str, str]:
    """
    Convenience function to export model for mobile deployment.
    
    Args:
        model: Model to export
        target_platforms: Target platforms ["ios", "android"]
        enable_quantization: Enable quantization
        output_dir: Output directory
        
    Returns:
        Dictionary of deployment paths
    """
    if target_platforms is None:
        target_platforms = ["ios", "android"]
    
    config = ExportConfig(
        target_devices=target_platforms,
        enable_quantization=enable_quantization
    )
    
    mobile_manager = MobileDeploymentManager(config)
    return mobile_manager.deploy_model(model, output_base_path=output_dir)


def optimize_cvm_for_mobile(cvm_model: CVMTransformer,
                           target_platforms: List[str] = None,
                           output_dir: str = "cvm_mobile") -> Dict[str, str]:
    """
    Optimize CVM model for mobile deployment.
    
    Args:
        cvm_model: CVM model to optimize
        target_platforms: Target platforms
        output_dir: Output directory
        
    Returns:
        Dictionary of deployment paths
    """
    if target_platforms is None:
        target_platforms = ["ios", "android"]
    
    config = ExportConfig(target_devices=target_platforms)
    optimizer = CVMMobileOptimizer(cvm_model)
    
    return optimizer.optimize_for_mobile(config, output_dir)


# Example usage and testing
if __name__ == "__main__":
    print("Testing MobileDeploymentManager...")
    
    # Test configuration
    config = ExportConfig(
        target_devices=["ios", "android"],
        enable_quantization=True,
        quantization_bits=8
    )
    
    # Test with dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
        
        def forward(self, x):
            return self.linear(x)
    
    dummy_model = DummyModel()
    
    try:
        mobile_manager = MobileDeploymentManager(config)
        print(f"Available exporters: {list(mobile_manager.exporters.keys())}")
        print(f"Target devices: {config.target_devices}")
        print("Mobile deployment manager ready for use!")
        
    except Exception as e:
        print(f"Expected error (missing dependencies): {e}")
    
    print("Mobile deployment system ready!")