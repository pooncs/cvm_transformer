#!/usr/bin/env python3
"""
Quick test of the complete model distillation framework.
"""

import torch
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_complete_distillation():
    """Test the complete distillation framework with a small number of iterations."""
    logger.info("Testing complete distillation framework...")
    
    try:
        from src.training.model_distillation import ModelDistiller, DistillationConfig
        
        # Create configuration for quick test
        config = DistillationConfig(
            num_iterations=100,  # Small number for testing
            validation_frequency=50,
            quality_threshold=0.8,  # Lower threshold for testing
            early_stopping_patience=2,
            temperature=6.0,
            alpha=0.8,
            quantization_aware_training=True,
            quantization_bits=8,
            batch_size=8,
            log_frequency=10,
            save_frequency=50,
            output_dir="test_distillation_output"
        )
        
        logger.info("Configuration created successfully")
        
        # Initialize distiller
        distiller = ModelDistiller(config)
        logger.info("Distiller initialized successfully")
        
        # Run a quick distillation test
        logger.info("Starting quick distillation test...")
        start_time = time.time()
        
        # Setup models and data
        distiller.load_teacher_model()
        distiller.create_student_model()
        distiller.prepare_training_data()
        distiller.setup_training()
        
        setup_time = time.time() - start_time
        logger.info(f"Setup completed in {setup_time:.2f} seconds")
        
        # Test a few training iterations
        logger.info("Testing training loop...")
        data_iterator = iter(distiller.train_dataloader)
        
        for iteration in range(1, 21):  # Test 20 iterations
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(distiller.train_dataloader)
                batch = next(data_iterator)
            
            # Training step
            losses = distiller.train_step(batch)
            
            if iteration % 5 == 0:
                logger.info(f"Iteration {iteration}: Loss={losses['total_loss']:.4f}")
        
        logger.info("Training loop test completed successfully")
        
        # Test validation
        logger.info("Testing validation...")
        validation_result = distiller.validate_model_enhanced()
        logger.info(f"Validation completed: Quality Score={validation_result.quality_score:.4f}")
        
        # Test report generation
        logger.info("Testing report generation...")
        report = distiller.generate_training_report()
        logger.info(f"Report generated: {len(report['training_history'])} metrics tracked")
        
        logger.info("✅ Complete distillation framework test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Complete distillation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting complete distillation framework test...")
    
    success = test_complete_distillation()
    
    if success:
        logger.info("🎉 All tests passed! The distillation framework is ready for 10,000 iterations.")
    else:
        logger.error("❌ Tests failed. Check the logs above for details.")