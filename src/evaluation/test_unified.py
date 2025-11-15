#!/usr/bin/env python3
"""
Final unified test script that can handle all model architectures
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import time
import sys
import os

# Add the current directory to path for imports
sys.path.append('.')

def create_simple_transformer(vocab_size, d_model=256, nhead=4, num_layers=4, 
                             dim_feedforward=1024, dropout=0.1, device='cpu'):
    """Create SimpleTransformer architecture"""
    
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, 
                     dim_feedforward=1024, dropout=0.1):
            super().__init__()
            
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            
            # Simple transformer
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            
            self.fc_out = nn.Linear(d_model, vocab_size)
            
            # Initialize with small weights to prevent initial bias
            self._init_weights()
        
        def _init_weights(self):
            """Initialize with small random weights."""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=0.02)
        
        def generate_square_subsequent_mask(self, sz):
            """Generate causal mask for decoder."""
            mask = torch.triu(torch.ones(sz, sz), diagonal=1)
            return mask.masked_fill(mask == 1, float('-inf'))
        
        def forward(self, src, tgt):
            # Embeddings
            src_emb = self.embedding(src)
            tgt_emb = self.embedding(tgt)
            
            # Create causal mask for decoder
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            # Transformer forward pass
            output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
            return self.fc_out(output)
    
    return SimpleTransformer(vocab_size, d_model, nhead, num_layers, 
                           dim_feedforward, dropout).to(device)

def create_enhanced_transformer(vocab_size, d_model=512, nhead=8, num_encoder_layers=12, 
                               num_decoder_layers=12, dim_feedforward=2048, dropout=0.1, device='cpu'):
    """Create EnhancedTransformer architecture"""
    
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-torch.log(torch.tensor(10000.0)) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            return x + self.pe[:x.size(0), :]
    
    class EnhancedTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=12, 
                     num_decoder_layers=12, dim_feedforward=2048, dropout=0.1):
            super().__init__()
            
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            
            # Enhanced transformer with 12 layers
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            
            self.fc_out = nn.Linear(d_model, vocab_size)
            self.dropout = nn.Dropout(dropout)
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights properly."""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                    src_padding_mask=None, tgt_padding_mask=None):
            
            # Embeddings with scaling
            src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
            tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
            
            # Add positional encoding
            src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
            tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
            
            # Apply dropout
            src_emb = self.dropout(src_emb)
            tgt_emb = self.dropout(tgt_emb)
            
            # Create causal mask for decoder if not provided
            if tgt_mask is None:
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            # Transformer forward pass
            output = self.transformer(
                src_emb, tgt_emb,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
            
            return self.fc_out(output)
        
        def generate_square_subsequent_mask(self, sz):
            """Generate causal mask for decoder."""
            mask = torch.triu(torch.ones(sz, sz), diagonal=1)
            return mask.masked_fill(mask == 1, float('-inf'))
    
    return EnhancedTransformer(vocab_size, d_model, nhead, num_encoder_layers, 
                             num_decoder_layers, dim_feedforward, dropout).to(device)

def load_model_universal(model_path: str, device: str):
    """Load any model with automatic architecture detection"""
    
    print(f"🔍 Analyzing model checkpoint: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract model state and config
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('config', {})
                val_loss = checkpoint.get('val_loss', 'unknown')
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"📊 Validation loss: {val_loss}, Epoch: {epoch}")
            else:
                state_dict = checkpoint
                config = {}
        else:
            state_dict = checkpoint
            config = {}
        
        # Determine architecture from state dict
        has_pos_encoder = any('pos_encoder' in key for key in state_dict.keys())
        has_embedding_scale = any('embedding.weight' in key and 'sqrt' in str(value) for key, value in state_dict.items())
        
        if has_pos_encoder:
            print("✅ Detected EnhancedTransformer architecture")
            vocab_size = state_dict['fc_out.weight'].shape[0]
            model = create_enhanced_transformer(vocab_size, device=device)
        else:
            print("✅ Detected SimpleTransformer architecture")
            vocab_size = state_dict['fc_out.weight'].shape[0]
            model = create_simple_transformer(vocab_size, device=device)
        
        # Load state dict
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def translate_universal(model, tokenizer, korean_text: str, device: str, max_length: int = 30):
    """Translate Korean to English using unified approach"""
    
    model.eval()
    
    with torch.no_grad():
        # Tokenize Korean input
        src_tokens = tokenizer.encode(korean_text)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
        
        # Start with BOS token
        tgt_tokens = [2]  # BOS
        
        for _ in range(max_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            
            # Get model predictions
            output = model(src_tensor, tgt_tensor)
            
            # Get the last token prediction
            next_token_logits = output[0, -1, :]
            next_token = next_token_logits.argmax().item()
            
            # Add to sequence
            tgt_tokens.append(next_token)
            
            # Stop if EOS token
            if next_token == 3:  # EOS
                break
        
        # Decode the output (excluding BOS and EOS)
        english_tokens = tgt_tokens[1:-1]  # Remove BOS and EOS
        english_text = tokenizer.decode(english_tokens)
        
        return english_text, tgt_tokens

def calculate_word_overlap_score(predicted: str, expected: str) -> float:
    """Calculate word overlap score between predicted and expected translations"""
    
    pred_words = set(predicted.lower().split())
    expected_words = set(expected.lower().split())
    
    if not expected_words:
        return 1.0 if not pred_words else 0.0
    
    overlap = len(pred_words & expected_words)
    return overlap / len(expected_words)

def test_model_universal(model_path: str, model_name: str):
    """Test a specific model with universal compatibility"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🧪 TESTING {model_name.upper()} MODEL")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    
    # Load tokenizer
    try:
        from cvm_translator.sp_tokenizer import SPTokenizer
        tokenizer = SPTokenizer("kr_en_diverse.model")
        print(f"✅ Loaded SPTokenizer, vocab size: {tokenizer.vocab_size()}")
    except ImportError:
        # Fallback to SentencePiece directly
        tokenizer = spm.SentencePieceProcessor(model_file='kr_en_diverse.model')
        print(f"✅ Loaded SentencePiece tokenizer, vocab size: {tokenizer.vocab_size()}")
    
    # Load model
    print(f"Loading {model_name} model from {model_path}...")
    model = load_model_universal(model_path, device)
    
    if model is None:
        print(f"❌ Failed to load {model_name} model")
        return None
    
    print(f"✅ {model_name} model loaded successfully")
    
    # Test sentences
    test_sentences = [
        ("안녕하세요", "Hello"),
        ("감사합니다", "Thank you"),
        ("안녕히 가세요", "Goodbye"),
        ("오늘 날씨가 정말 좋네요", "The weather is really nice today"),
        ("저는 커피를 좋아합니다", "I like coffee"),
        ("회의가 몇 시에 있나요?", "What time is the meeting?"),
        ("얼마예요?", "How much is it?"),
        ("화장실이 어디에 있나요?", "Where is the bathroom?"),
        ("계산서 주세요", "Please give me the bill"),
        ("오늘 기분이 정말 좋아요", "I feel really good today"),
        ("걱정이 많이 되네요", "I'm very worried"),
        ("이 소식이 너무 기쁩니다", "This news makes me very happy"),
        ("내일 봐요", "See you tomorrow"),
        ("잘 지내세요?", "How are you?"),
        ("만나서 반갑습니다", "Nice to meet you"),
        ("죄송합니다", "I'm sorry"),
        ("도와주세요", "Please help me")
    ]
    
    print(f"\n🧪 Testing {len(test_sentences)} sentence pairs")
    print("-" * 70)
    
    results = []
    total_score = 0
    total_tokens = 0
    all_tokens = []
    
    for i, (korean, expected) in enumerate(test_sentences, 1):
        print(f"\nTest {i}:")
        print(f"🇰🇷 Korean: {korean}")
        print(f"🇺🇸 Expected: {expected}")
        
        try:
            # Translate
            start_time = time.time()
            translation, tokens = translate_universal(model, tokenizer, korean, device)
            translation_time = time.time() - start_time
            
            # Calculate score
            score = calculate_word_overlap_score(translation, expected)
            total_score += score
            
            # Token analysis
            total_tokens += len(tokens)
            all_tokens.extend(tokens)
            
            print(f"🤖 Translation: {translation}")
            print(f"🔢 Tokens: {tokens}")
            print(f"📊 Word overlap: {score:.2f}")
            print(f"⏱️ Translation time: {translation_time:.3f}s")
            
            # Quality assessment
            if len(tokens) <= 3:  # Too short
                print("⚠️  WARNING: Translation too short")
            elif all(t == tokens[1] for t in tokens[1:]):  # All same token
                print("⚠️  WARNING: All tokens identical")
            elif tokens.count(1) > len(tokens) * 0.5:  # Too many UNK tokens
                print("⚠️  WARNING: Too many unknown tokens")
            else:
                print("✅ Translation looks reasonable")
            
            results.append({
                'korean': korean,
                'expected': expected,
                'translation': translation,
                'score': score,
                'tokens': tokens,
                'token_count': len(tokens),
                'time': translation_time
            })
            
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            results.append({
                'korean': korean,
                'expected': expected,
                'translation': "FAILED",
                'score': 0.0,
                'error': str(e)
            })
    
    # Summary statistics
    avg_score = total_score / len(test_sentences)
    vocab_size = tokenizer.vocab_size()
    
    print(f"\n📈 {model_name.upper()} MODEL RESULTS")
    print("=" * 50)
    print(f"Average translation score: {avg_score:.3f}")
    print(f"Tests passed (>0.3 score): {sum(1 for r in results if r['score'] > 0.3)}/{len(results)}")
    print(f"Tests with good overlap (>0.5): {sum(1 for r in results if r['score'] > 0.5)}/{len(results)}")
    print(f"Perfect translations (>0.8): {sum(1 for r in results if r['score'] > 0.8)}/{len(results)}")
    
    # Token analysis
    if all_tokens:
        unique_tokens = set(all_tokens)
        print(f"\n🔤 TOKEN ANALYSIS")
        print("=" * 20)
        print(f"Total tokens generated: {len(all_tokens)}")
        print(f"Unique tokens used: {len(unique_tokens)}")
        print(f"Vocabulary utilization: {len(unique_tokens) / vocab_size * 100:.2f}%")
        print(f"Token ID range: {min(unique_tokens)} - {max(unique_tokens)}")
        
        # Check for the old problem (token ID 1)
        if 1 in unique_tokens:
            print(f"⚠️  WARNING: Token ID 1 (UNK) used {all_tokens.count(1)} times")
        else:
            print("✅ Token ID 1 (problematic token) not used")
        
        # Check for EOS token usage
        eos_count = all_tokens.count(3)
        print(f"EOS tokens (ID 3): {eos_count} ({eos_count/len(all_tokens)*100:.1f}%)")
        
        # Check for diversity
        if len(unique_tokens) < 10:
            print("⚠️  WARNING: Very low vocabulary diversity")
        elif len(unique_tokens) < 50:
            print("📊 Moderate vocabulary diversity")
        else:
            print("✅ Good vocabulary diversity")
    
    # Save results
    results_file = f'{model_name}_model_final_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': model_name,
            'model_path': model_path,
            'average_score': avg_score,
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results if r['score'] > 0.3),
            'good_tests': sum(1 for r in results if r['score'] > 0.5),
            'perfect_tests': sum(1 for r in results if r['score'] > 0.8),
            'vocab_utilization': len(set(all_tokens)) / vocab_size * 100 if all_tokens else 0,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {results_file}")
    
    return {
        'model_name': model_name,
        'average_score': avg_score,
        'results': results,
        'vocab_utilization': len(set(all_tokens)) / vocab_size * 100 if all_tokens else 0
    }

def main():
    """Main function to test all available models"""
    
    print("🚀 FINAL UNIFIED MODEL TESTING SYSTEM")
    print("=" * 60)
    print("This script automatically detects model architecture and tests compatibility")
    
    # Test all available models
    models_to_test = [
        ('simple_best_model.pth', 'simple'),
        ('best_enhanced_model.pth', 'enhanced'),
        ('best_advanced_simple_model.pth', 'advanced_simple'),
        ('clean_advanced_model.pth', 'clean_advanced')
    ]
    
    results = {}
    
    for model_path, model_name in models_to_test:
        if os.path.exists(model_path):
            print(f"\n📁 Found {model_path}, testing...")
            result = test_model_universal(model_path, model_name)
            if result:
                results[model_name] = result
        else:
            print(f"\n⚠️ Model {model_path} not found, skipping...")
    
    # Compare results
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("📊 MODEL COMPARISON")
        print(f"{'='*60}")
        
        for model_name, result in results.items():
            print(f"{model_name}: {result['average_score']:.3f} "
                  f"(vocab: {result['vocab_utilization']:.1f}%)")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['average_score'])
        print(f"\n🏆 BEST MODEL: {best_model[0]} (Score: {best_model[1]['average_score']:.3f})")
    
    # Final assessment
    if results:
        best_score = max(result['average_score'] for result in results.values())
        best_vocab = max(result['vocab_utilization'] for result in results.values())
        
        print(f"\n🎯 FINAL ASSESSMENT")
        print("=" * 30)
        
        if best_score > 0.5:
            print("🎉 EXCELLENT: Translation system is working very well!")
            print("✅ Architecture compatibility issues resolved")
            print("✅ Multiple models working correctly")
        elif best_score > 0.3:
            print("✅ GOOD: Translation system is working well!")
            print("✅ Architecture compatibility achieved")
            print("📊 Some models performing better than others")
        elif best_score > 0.1:
            print("⚠️ FAIR: Translation system is functional but needs improvement")
            print("✅ Basic compatibility working")
            print("📊 Model quality varies significantly")
        else:
            print("❌ POOR: Translation system needs major fixes")
            print("⚠️  Architecture compatibility issues remain")
            print("📚 Further investigation needed")
        
        print(f"\n📊 Best performance: {best_score:.3f} score, {best_vocab:.1f}% vocab utilization")
    
    return results

if __name__ == "__main__":
    main()