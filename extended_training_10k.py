#!/usr/bin/env python3
"""
Extended training script for 10,000 iterations with comprehensive monitoring.
"""

import torch
import time
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from cvm_translator.cvm_transformer import CVMTransformer
from cvm_translator.kd_losses import compute_loss
from collections import defaultdict


class BiTextDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.tokenizer.encode(src)[:self.max_len]
        tgt_ids = self.tokenizer.encode(tgt)[:self.max_len]
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long)
        }


def collate(batch):
    src_ids = torch.nn.utils.rnn.pad_sequence([b["src_ids"] for b in batch], batch_first=True, padding_value=0)
    tgt_ids = torch.nn.utils.rnn.pad_sequence([b["tgt_ids"] for b in batch], batch_first=True, padding_value=0)
    return {"src_ids": src_ids, "tgt_ids": tgt_ids}


def train_epoch_extended(student, teacher, loader, optimizer, device, epoch, total_epochs):
    student.train()
    teacher.eval()
    total_loss = 0
    batch_losses = []
    
    for batch_idx, batch in enumerate(loader):
        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)
        optimizer.zero_grad()
        
        # Forward pass - CVMTransformer only returns logits
        s_logits = student(src)
        with torch.no_grad():
            t_logits = teacher(src)
        
        # Compute loss - use logits directly for knowledge distillation
        # Simple KD loss: MSE between student and teacher logits + cross-entropy with target
        kd_loss = torch.nn.functional.mse_loss(s_logits, t_logits)
        
        # For cross-entropy, we need to handle the sequence dimension properly
        # Reshape to [batch_size * seq_len, vocab_size] for CE loss
        batch_size, seq_len, vocab_size = s_logits.shape
        s_logits_flat = s_logits.view(-1, vocab_size)
        tgt_flat = tgt.view(-1)
        
        ce_loss = torch.nn.functional.cross_entropy(
            s_logits_flat, 
            tgt_flat, 
            ignore_index=0
        )
        loss = 0.7 * ce_loss + 0.3 * kd_loss  # Weighted combination
        
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        
        # Progress logging every 50 batches
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx:3d}/{len(loader):3d} | Loss: {batch_loss:.6f}")
    
    avg_loss = total_loss / len(loader)
    return avg_loss, batch_losses


def test_translation_quality(student, tokenizer, device):
    """Test translation quality on sample sentences."""
    student.eval()
    test_sentences = [
        ("안녕하세요", "Hello"),
        ("오늘 날씨 좋네요", "Today weather is nice"),
        ("실시간 번역", "real-time translation"),
        ("CVM 알고리즘", "CVM algorithm"),
        ("한국어 영어", "Korean English"),
        ("감사합니다", "Thank you"),
        ("어디에 가세요?", "Where are you going?"),
        ("이것은 테스트입니다", "This is a test"),
    ]
    
    results = []
    with torch.no_grad():
        for korean, expected in test_sentences:
            # Tokenize input
            src_ids = tokenizer.encode(korean)
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
            
            # Generate translation
            logits = student(src_tensor)
            predicted_ids = torch.argmax(logits[0], dim=-1).cpu().numpy()
            
            # Decode output (simplified)
            predicted_words = [tokenizer.decode([idx]) if idx < len(tokenizer.vocab) else "<UNK>" 
                             for idx in predicted_ids[:len(src_ids)]]
            predicted_text = " ".join(predicted_words)
            
            results.append({
                "input": korean,
                "expected": expected,
                "predicted": predicted_text,
                "length_match": len(predicted_ids) == len(src_ids)
            })
    
    return results


def main():
    print("🚀 CVM TRANSFORMER - EXTENDED TRAINING (10,000 ITERATIONS)")
    print("=" * 70)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 32000
    d_model = 768
    n_layers_student = 6
    n_layers_teacher = 12
    core_capacity_student = 64
    core_capacity_teacher = 256
    max_len = 128
    batch_size = 16
    learning_rate = 1e-4
    total_epochs = 50  # Will result in ~10,000 iterations with current dataset size
    
    print(f"📊 Configuration:")
    print(f"   Device: {device}")
    print(f"   Vocab Size: {vocab_size}")
    print(f"   Model Dim: {d_model}")
    print(f"   Student Layers: {n_layers_student} (capacity: {core_capacity_student})")
    print(f"   Teacher Layers: {n_layers_teacher} (capacity: {core_capacity_teacher})")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Total Epochs: {total_epochs}")
    print()
    
    # Initialize models
    print("🔄 Initializing models...")
    student = CVMTransformer(vocab_size, d_model=d_model, n_layers=n_layers_student, 
                          core_capacity=core_capacity_student).to(device)
    teacher = CVMTransformer(vocab_size, d_model=d_model, n_layers=n_layers_teacher, 
                            core_capacity=core_capacity_teacher).to(device)
    
    # Initialize teacher with better weights (simulated pre-training)
    for param in teacher.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
    
    # Create enhanced dataset
    print("📚 Creating enhanced training dataset...")
    base_pairs = [
        ("안녕하세요", "Hello"),
        ("오늘 날씨 좋네요", "Today weather is nice"),
        ("실시간 번역", "real-time translation"),
        ("CVM 알고리즘", "CVM algorithm"),
        ("한국어 영어", "Korean English"),
        ("감사합니다", "Thank you"),
        ("어디에 가세요?", "Where are you going?"),
        ("이것은 테스트입니다", "This is a test"),
        ("좋은 아침입니다", "Good morning"),
        ("안녕히 가세요", "Goodbye"),
        ("네, 알겠습니다", "Yes, I understand"),
        ("아니요, 괜찮습니다", "No, it's okay"),
        ("도와주세요", "Help me"),
        ("얼마예요?", "How much is it?"),
        ("어디 있어요?", "Where is it?"),
        ("지금 몇 시예요?", "What time is it now?"),
        ("배고파요", "I'm hungry"),
        ("목말라요", "I'm thirsty"),
        ("피곤해요", "I'm tired"),
        ("행복해요", "I'm happy"),
    ]
    
    # Expand dataset by repeating and varying
    pairs = []
    for i in range(200):  # 4000 total samples
        pairs.extend(base_pairs)
    
    # Simple tokenizer simulation
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {}
            self.reverse_vocab = {}
            self._build_vocab()
        
        def _build_vocab(self):
            # Simple character-level vocabulary
            chars = set()
            for ko, en in base_pairs:
                chars.update(ko + en)
            
            special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
            for token in special_tokens:
                self.vocab[token] = len(self.vocab)
            
            for char in sorted(chars):
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
            
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        def encode(self, text):
            return [self.vocab.get(char, self.vocab["<unk>"]) for char in text]
        
        def decode(self, ids):
            return "".join([self.reverse_vocab.get(id, "<unk>") for id in ids])
    
    tokenizer = SimpleTokenizer()
    
    # Create dataset and loader
    dataset = BiTextDataset(pairs, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Number of batches: {len(loader)} per epoch")
    print(f"   Total iterations: ~{len(loader) * total_epochs}")
    print()
    
    # Training setup
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training metrics
    metrics = {
        'epoch_losses': [],
        'learning_rates': [],
        'batch_losses': defaultdict(list),
        'translation_quality': [],
        'training_times': []
    }
    
    # Training loop
    print("🏃 Starting extended training...")
    start_time = time.time()
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        print(f"\n📈 Epoch {epoch+1:2d}/{total_epochs}")
        print("-" * 40)
        
        # Training
        avg_loss, batch_losses = train_epoch_extended(student, teacher, loader, optimizer, device, 
                                                     epoch, total_epochs)
        
        # Record metrics
        metrics['epoch_losses'].append(avg_loss)
        metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
        metrics['batch_losses'][epoch] = batch_losses
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Translation quality check every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\n🧪 Testing translation quality...")
            quality_results = test_translation_quality(student, tokenizer, device)
            metrics['translation_quality'].append({
                'epoch': epoch + 1,
                'results': quality_results
            })
            
            # Show sample translations
            print("   Sample translations:")
            for i, result in enumerate(quality_results[:3]):
                print(f"   {i+1}. '{result['input']}' → '{result['predicted']}' (expected: '{result['expected']}')")
        
        # Timing
        epoch_time = time.time() - epoch_start
        metrics['training_times'].append(epoch_time)
        
        print(f"   Average Loss: {avg_loss:.6f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"   Epoch Time: {epoch_time:.2f}s")
        print(f"   Total Time: {time.time() - start_time:.1f}s")
        
        # Early stopping check
        if epoch > 10 and avg_loss < 0.01:
            print(f"\n✅ Early stopping at epoch {epoch+1} - loss converged")
            break
    
    total_training_time = time.time() - start_time
    
    # Final results
    print("\n" + "=" * 70)
    print("🎯 TRAINING COMPLETED!")
    print("=" * 70)
    
    print(f"\n📊 FINAL METRICS:")
    print(f"   Total Epochs: {len(metrics['epoch_losses'])}")
    print(f"   Total Training Time: {total_training_time:.1f}s")
    print(f"   Average Epoch Time: {sum(metrics['training_times'])/len(metrics['training_times']):.2f}s")
    print(f"   Final Loss: {metrics['epoch_losses'][-1]:.6f}")
    print(f"   Initial Loss: {metrics['epoch_losses'][0]:.6f}")
    print(f"   Loss Reduction: {((metrics['epoch_losses'][0] - metrics['epoch_losses'][-1]) / metrics['epoch_losses'][0] * 100):.1f}%")
    
    print(f"\n🚀 PERFORMANCE ANALYSIS:")
    print(f"   Training Speed: {len(loader) * len(metrics['epoch_losses']) / total_training_time:.1f} iterations/second")
    print(f"   Total Iterations: {len(loader) * len(metrics['epoch_losses'])}")
    print(f"   Samples Processed: {len(dataset) * len(metrics['epoch_losses'])}")
    
    # Save results
    results = {
        'config': {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_layers_student': n_layers_student,
            'n_layers_teacher': n_layers_teacher,
            'core_capacity_student': core_capacity_student,
            'core_capacity_teacher': core_capacity_teacher,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'total_epochs': len(metrics['epoch_losses']),
            'device': str(device)
        },
        'metrics': metrics,
        'total_training_time': total_training_time,
        'final_loss': metrics['epoch_losses'][-1],
        'initial_loss': metrics['epoch_losses'][0]
    }
    
    with open('training_10k_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Plot training curve
    plt.figure(figsize=(12, 8))
    
    # Loss curve
    plt.subplot(2, 2, 1)
    plt.plot(metrics['epoch_losses'])
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Learning rate
    plt.subplot(2, 2, 2)
    plt.plot(metrics['learning_rates'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    # Training time
    plt.subplot(2, 2, 3)
    plt.plot(metrics['training_times'])
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    # Loss reduction rate
    plt.subplot(2, 2, 4)
    if len(metrics['epoch_losses']) > 1:
        loss_diff = [metrics['epoch_losses'][i] - metrics['epoch_losses'][i+1] 
                    for i in range(len(metrics['epoch_losses'])-1)]
        plt.plot(loss_diff)
        plt.title('Loss Reduction Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_10k_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Results saved to:")
    print(f"   - training_10k_results.json (detailed metrics)")
    print(f"   - training_10k_analysis.png (training curves)")
    
    return results


if __name__ == "__main__":
    results = main()