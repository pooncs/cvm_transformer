#!/usr/bin/env python3
"""
Export the advanced model to a clean format without scheduler dependencies
"""

import torch

def export_advanced_model():
    """Export the advanced model to a clean format"""
    
    print("Exporting advanced model...")
    
    try:
        # Load the full checkpoint
        checkpoint = torch.load('best_advanced_simple_model.pth', weights_only=False)
        
        # Extract just the model state dict
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            # If it's just the state dict, use it directly
            model_state = checkpoint
        
        # Save clean model state
        torch.save({
            'model_state_dict': model_state,
            'config': {
                'vocab_size': 1000,
                'd_model': 512,
                'nhead': 8,
                'num_encoder_layers': 12,
                'num_decoder_layers': 12,
                'dim_feedforward': 2048,
                'dropout': 0.1
            },
            'training_info': {
                'best_val_loss': checkpoint.get('val_loss', 'unknown'),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'model_type': 'advanced_with_lr_scheduling'
            }
        }, 'clean_advanced_model.pth')
        
        print("✅ Advanced model exported successfully to clean_advanced_model.pth")
        
        # Also save just the state dict for maximum compatibility
        torch.save(model_state, 'advanced_model_state_dict.pth')
        print("✅ Model state dict saved to advanced_model_state_dict.pth")
        
    except Exception as e:
        print(f"❌ Error exporting advanced model: {e}")
        print("Using enhanced model as fallback...")
        
        # Fall back to enhanced model
        checkpoint = torch.load('best_enhanced_model.pth', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
        
        torch.save({
            'model_state_dict': model_state,
            'config': {
                'vocab_size': 1000,
                'd_model': 512,
                'nhead': 8,
                'num_encoder_layers': 12,
                'num_decoder_layers': 12,
                'dim_feedforward': 2048,
                'dropout': 0.1
            },
            'training_info': {
                'model_type': 'enhanced_fallback'
            }
        }, 'clean_advanced_model.pth')
        
        print("✅ Enhanced model exported as fallback")

if __name__ == "__main__":
    export_advanced_model()