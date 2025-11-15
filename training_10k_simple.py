#!/usr/bin/env python3
"""
Simplified training script for 10,000 iterations with proper debugging.
"""

import torch
import time
import json
from torch.utils.data import Dataset, DataLoader
from cvm_translator.cvm_transformer import CVMTransformer
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


def train_epoch_simple(student, teacher, loader, optimizer, device, epoch):
    student.train()
    teacher.eval()
    total_loss = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(loader):
        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)
        optimizer.zero_grad()
        
        # Forward pass
        s_logits = student(src)
        with torch.no_grad():
            t_logits = teacher(src)
        
        # Simple KD loss - MSE between logits
        kd_loss = torch.nn.functional.mse_loss(s_logits, t_logits)
        
        loss = kd_loss  # Start with just KD loss for simplicity
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        batch_count += 1
        
        # Progress logging every 50 batches
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx:3d}/{len(loader):3d} | Loss: {batch_loss:.6f}")
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss


def main():
    print("🚀 CVM TRANSFORMER - SIMPLIFIED TRAINING (10,000+ ITERATIONS)")
    print("=" * 70)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 32000
    d_model = 768
    n_layers_student = 6
    n_layers_teacher = 12
    core_capacity_student = 64
    core_capacity_teacher = 256
    max_len = 64  # Reduced for faster training
    batch_size = 32  # Increased for better throughput
    learning_rate = 1e-4
    total_epochs = 40  # Will result in ~10,000 iterations
    
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
    
    # Initialize teacher with better weights
    for param in teacher.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
    
    # Create dataset
    print("📚 Creating training dataset...")
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
    
    # Expand dataset significantly for 10k+ iterations
    pairs = []
    for i in range(500):  # 10,000 total samples
        pairs.extend(base_pairs)
    
    # Simple tokenizer
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
            self._build_vocab()
        
        def _build_vocab(self):
            chars = set()
            for ko, en in base_pairs:
                chars.update(ko + en)
            
            for char in sorted(chars):
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        
        def encode(self, text):
            return [self.vocab.get(char, self.vocab["<unk>"]) for char in text]
        
        def decode(self, ids):
            return "".join([list(self.vocab.keys())[id] if id < len(self.vocab) else "<unk>" 
                           for id in ids if id < len(self.vocab)])
    
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5)
    
    # Training metrics
    epoch_losses = []
    training_times = []
    
    # Training loop
    print("🏃 Starting extended training...")
    start_time = time.time()
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        print(f"\n📈 Epoch {epoch+1:2d}/{total_epochs}")
        print("-" * 40)
        
        # Training
        avg_loss = train_epoch_simple(student, teacher, loader, optimizer, device, epoch)
        
        # Record metrics
        epoch_losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Timing
        epoch_time = time.time() - epoch_start
        training_times.append(epoch_time)
        
        print(f"   Average Loss: {avg_loss:.6f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"   Epoch Time: {epoch_time:.2f}s")
        print(f"   Total Time: {time.time() - start_time:.1f}s")
        
        # Early stopping if loss converges
        if epoch > 15 and avg_loss < 0.001:
            print(f"\n✅ Early stopping at epoch {epoch+1} - loss converged")
            break
    
    total_training_time = time.time() - start_time
    
    # Final results
    print("\n" + "=" * 70)
    print("🎯 TRAINING COMPLETED!")
    print("=" * 70)
    
    print(f"\n📊 FINAL METRICS:")
    print(f"   Total Epochs: {len(epoch_losses)}")
    print(f"   Total Training Time: {total_training_time:.1f}s")
    print(f"   Average Epoch Time: {sum(training_times)/len(training_times):.2f}s")
    print(f"   Final Loss: {epoch_losses[-1]:.6f}")
    print(f"   Initial Loss: {epoch_losses[0]:.6f}")
    print(f"   Loss Reduction: {((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.1f}%")
    
    total_iterations = len(loader) * len(epoch_losses)
    print(f"\n🚀 PERFORMANCE ANALYSIS:")
    print(f"   Training Speed: {total_iterations / total_training_time:.1f} iterations/second")
    print(f"   Total Iterations: {total_iterations}")
    print(f"   Samples Processed: {len(dataset) * len(epoch_losses)}")
    print(f"   Throughput: {len(dataset) * len(epoch_losses) / total_training_time:.1f} samples/second")
    
    # Test final model
    print(f"\n🧪 FINAL MODEL TEST:")
    student.eval()
    test_sentences = [
        ("안녕하세요", "Hello"),
        ("감사합니다", "Thank you"),
        ("실시간 번역", "real-time translation"),
    ]
    
    with torch.no_grad():
        for korean, expected in test_sentences:
            src_ids = tokenizer.encode(korean)
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
            logits = student(src_tensor)
            predicted_ids = torch.argmax(logits[0], dim=-1).cpu().numpy()
            predicted_text = tokenizer.decode(predicted_ids[:len(src_ids)])
            print(f"   '{korean}' → '{predicted_text}' (expected: '{expected}')")
    
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
            'total_epochs': len(epoch_losses),
            'max_len': max_len,
            'device': str(device)
        },
        'metrics': {
            'epoch_losses': epoch_losses,
            'training_times': training_times,
            'total_iterations': total_iterations,
            'total_training_time': total_training_time
        },
        'final_model_test': {
            'test_sentences': test_sentences,
            'results': 'see console output'
        }
    }
    
    with open('training_10k_simple_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: training_10k_simple_results.json")
    
    return results


if __name__ == "__main__":
    results = main()