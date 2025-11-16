#!/usr/bin/env python3
"""
Test script to identify and fix device handling issues in model distillation.
"""

import torch
import torch.nn as nn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_device_handling():
    """Test device handling in the distillation framework."""
    logger.info("Testing device handling...")

    # Check available devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # Test simple tensor operations
    try:
        test_tensor = torch.randn(2, 3, device=device)
        logger.info(f"Test tensor created on {device}: shape {test_tensor.shape}")

        # Test tensor operations
        result = test_tensor * 2
        logger.info("Basic tensor operations successful")

    except Exception as e:
        logger.error(f"Tensor operation failed: {e}")
        return False

    # Test model creation and forward pass
    try:
        from cvm_translator.cvm_transformer import CVMTransformer

        # Create a small test model
        model = CVMTransformer(
            vocab_size=1000, d_model=128, n_heads=4, n_layers=2, ff_dim=512
        ).to(device)

        logger.info(
            f"Test model created with {sum(p.numel() for p in model.parameters()):,} parameters"
        )

        # Test forward pass
        test_input = torch.randint(0, 1000, (2, 10), device=device)
        with torch.no_grad():
            output = model(test_input)

        logger.info(f"Model forward pass successful: output shape {output.shape}")

    except Exception as e:
        logger.error(f"Model test failed: {e}")
        return False

    # Test tokenizer device handling
    try:

        class SimpleTokenizer:
            def __init__(self, vocab_size=1000):
                self.vocab_size = vocab_size
                self.pad_token_id = 0

            def __call__(self, text, **kwargs):
                max_length = kwargs.get("max_length", 128)
                return {
                    "input_ids": torch.randint(1, 1000, (1, max_length)),
                    "attention_mask": torch.ones(1, max_length),
                }

        tokenizer = SimpleTokenizer()
        test_text = "This is a test sentence."
        tokens = tokenizer(test_text, max_length=10)

        # Test device placement
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        logger.info(
            f"Tokenizer test successful: input_ids shape {input_ids.shape}, device {input_ids.device}"
        )

    except Exception as e:
        logger.error(f"Tokenizer test failed: {e}")
        return False

    logger.info("All device handling tests passed!")
    return True


def test_distillation_dataset():
    """Test the DistillationDataset class for device issues."""
    logger.info("Testing DistillationDataset...")

    try:
        from cvm_translator.model_distillation import DistillationDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a proper teacher model (CVMTransformer)
        from cvm_translator.cvm_transformer import CVMTransformer

        teacher_model = CVMTransformer(
            vocab_size=100, d_model=64, n_heads=2, n_layers=2, ff_dim=256
        ).to(device)

        # Create a simple tokenizer
        class SimpleTokenizer:
            def __init__(self):
                self.vocab_size = 100
                self.pad_token_id = 0

            def __call__(self, text, **kwargs):
                max_length = kwargs.get("max_length", 5)
                return {
                    "input_ids": torch.randint(1, 100, (1, max_length)),
                    "attention_mask": torch.ones(1, max_length),
                }

        tokenizer = SimpleTokenizer()

        # Create dataset
        texts = ["Test sentence 1", "Test sentence 2", "Test sentence 3"]
        dataset = DistillationDataset(
            texts=texts,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            max_length=5,
            device=device,
        )

        logger.info(f"Dataset created with {len(dataset)} samples")

        # Test getting an item
        item = dataset[0]
        logger.info(f"Dataset item retrieved successfully")
        logger.info(
            f"Input IDs shape: {item['input_ids'].shape}, device: {item['input_ids'].device}"
        )
        logger.info(
            f"Teacher logits shape: {item['teacher_logits'].shape}, device: {item['teacher_logits'].device}"
        )

        # Verify all tensors are on the same device
        for key, value in item.items():
            if torch.is_tensor(value):
                # Compare device types (cuda vs cpu)
                if value.device.type != device.type:
                    logger.error(
                        f"Tensor {key} is on {value.device}, expected {device}"
                    )
                    return False

        logger.info("All tensors are on the correct device")

    except Exception as e:
        logger.error(f"DistillationDataset test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    logger.info("DistillationDataset test passed!")
    return True


if __name__ == "__main__":
    logger.info("Starting device handling tests...")

    success1 = test_device_handling()
    success2 = test_distillation_dataset()

    if success1 and success2:
        logger.info("✅ All tests passed! Device handling is working correctly.")
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
