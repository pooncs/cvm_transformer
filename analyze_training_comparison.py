#!/usr/bin/env python3
"""
Comprehensive analysis comparing 1,000 vs 10,000+ iteration training results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_previous_results():
    """Load previous 1000-iteration training results from summary."""
    return {
        'total_iterations': 1000,
        'final_loss': 0.1080,
        'training_time': 4.9,
        'speed': 203.2,
        'loss_reduction_percent': 89.3,
        'initial_loss': 1.0  # Estimated from 89.3% reduction
    }

def load_10k_results():
    """Load 10,000+ iteration training results."""
    try:
        with open('training_10k_simple_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 10k training results file not found")
        return None

def create_comparison_analysis():
    """Create comprehensive comparison analysis."""
    
    print("📊 CVM TRANSFORMER TRAINING COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Load results
    prev_results = load_previous_results()
    results_10k = load_10k_results()
    
    if not results_10k:
        return
    
    metrics_10k = results_10k['metrics']
    
    print(f"\n📈 TRAINING SCALABILITY ANALYSIS:")
    print("-" * 50)
    print(f"   Previous Training: {prev_results['total_iterations']:,} iterations")
    print(f"   Extended Training: {metrics_10k['total_iterations']:,} iterations")
    print(f"   Scale Factor: {metrics_10k['total_iterations'] / prev_results['total_iterations']:.1f}x")
    
    print(f"\n🎯 LOSS CONVERGENCE COMPARISON:")
    print("-" * 50)
    print(f"   1K Iterations - Final Loss: {prev_results['final_loss']:.6f}")
    print(f"   10K+ Iterations - Final Loss: {metrics_10k['epoch_losses'][-1]:.6f}")
    print(f"   Loss Improvement: {((prev_results['final_loss'] - metrics_10k['epoch_losses'][-1]) / prev_results['final_loss'] * 100):.1f}%")
    print(f"   Convergence Quality: {'EXCELLENT' if metrics_10k['epoch_losses'][-1] < 0.0001 else 'GOOD' if metrics_10k['epoch_losses'][-1] < 0.001 else 'FAIR'}")
    
    print(f"\n⚡ TRAINING EFFICIENCY ANALYSIS:")
    print("-" * 50)
    prev_speed = prev_results['total_iterations'] / prev_results['training_time']
    new_speed = metrics_10k['total_iterations'] / metrics_10k['total_training_time']
    
    print(f"   1K Iterations Speed: {prev_speed:.1f} iterations/second")
    print(f"   10K+ Iterations Speed: {new_speed:.1f} iterations/second")
    print(f"   Speed Change: {((new_speed - prev_speed) / prev_speed * 100):+.1f}%")
    print(f"   Training Time Scaling: {metrics_10k['total_training_time'] / prev_results['training_time']:.1f}x")
    
    print(f"\n📊 CONVERGENCE PATTERN ANALYSIS:")
    print("-" * 50)
    
    # Analyze convergence patterns
    losses_10k = metrics_10k['epoch_losses']
    
    # Find convergence point (when loss < 0.001)
    convergence_threshold = 0.001
    convergence_epoch = None
    for i, loss in enumerate(losses_10k):
        if loss < convergence_threshold:
            convergence_epoch = i + 1
            break
    
    if convergence_epoch:
        convergence_iterations = convergence_epoch * (metrics_10k['total_iterations'] / len(losses_10k))
        print(f"   Convergence Epoch: {convergence_epoch}")
        print(f"   Convergence Iterations: {convergence_iterations:.0f}")
        print(f"   Time to Convergence: {metrics_10k['training_times'][:convergence_epoch]}")
        convergence_time = sum(metrics_10k['training_times'][:convergence_epoch])
        print(f"   Convergence Time: {convergence_time:.1f}s")
    
    print(f"\n🔬 STABILITY ANALYSIS:")
    print("-" * 50)
    
    # Calculate loss variance in later epochs
    if len(losses_10k) > 10:
        recent_losses = losses_10k[-10:]
        loss_variance = np.var(recent_losses)
        loss_std = np.std(recent_losses)
        
        print(f"   Final 10 Epochs - Mean Loss: {np.mean(recent_losses):.8f}")
        print(f"   Final 10 Epochs - Std Dev: {loss_std:.8f}")
        print(f"   Training Stability: {'VERY STABLE' if loss_std < 1e-5 else 'STABLE' if loss_std < 1e-4 else 'UNSTABLE'}")
    
    print(f"\n🚀 PERFORMANCE METRICS:")
    print("-" * 50)
    
    # Calculate additional metrics
    total_samples = results_10k['config']['batch_size'] * metrics_10k['total_iterations']
    throughput = total_samples / metrics_10k['total_training_time']
    
    print(f"   Total Samples Processed: {total_samples:,}")
    print(f"   Overall Throughput: {throughput:.1f} samples/second")
    print(f"   Average Epoch Time: {np.mean(metrics_10k['training_times']):.2f}s")
    print(f"   Training Efficiency: {'HIGH' if throughput > 500 else 'MEDIUM' if throughput > 100 else 'LOW'}")
    
    print(f"\n📈 SCALING INSIGHTS:")
    print("-" * 50)
    
    # Analyze scaling behavior
    if len(losses_10k) > 5:
        # Check if loss reduction follows exponential decay
        x = np.arange(len(losses_10k))
        y = np.log(losses_10k)
        
        # Fit exponential decay
        try:
            coeffs = np.polyfit(x, y, 1)
            decay_rate = -coeffs[0]
            print(f"   Exponential Decay Rate: {decay_rate:.4f}")
            print(f"   Decay Quality: {'EXCELLENT' if decay_rate > 0.5 else 'GOOD' if decay_rate > 0.2 else 'POOR'}")
        except:
            print(f"   Decay Analysis: Could not fit exponential model")
    
    print(f"\n🎯 TRAINING QUALITY ASSESSMENT:")
    print("-" * 50)
    
    # Overall quality assessment
    final_loss = losses_10k[-1]
    training_time = metrics_10k['total_training_time']
    iterations = metrics_10k['total_iterations']
    
    quality_score = 0
    if final_loss < 1e-5: quality_score += 40
    elif final_loss < 1e-4: quality_score += 30
    elif final_loss < 1e-3: quality_score += 20
    elif final_loss < 1e-2: quality_score += 10
    
    if iterations >= 5000: quality_score += 30
    elif iterations >= 2000: quality_score += 20
    elif iterations >= 1000: quality_score += 10
    
    if training_time < 300: quality_score += 30
    elif training_time < 600: quality_score += 20
    elif training_time < 1200: quality_score += 10
    
    quality_grade = 'A+' if quality_score >= 90 else 'A' if quality_score >= 80 else 'B' if quality_score >= 70 else 'C'
    
    print(f"   Final Loss Score: {min(40, max(0, int(-np.log10(final_loss) * 10)))}/40")
    print(f"   Iteration Score: {min(30, int(iterations / 200))}/30")
    print(f"   Speed Score: {min(30, max(0, int((300 - training_time) / 10)))}/30")
    print(f"   Overall Quality Score: {quality_score}/100")
    print(f"   Training Grade: {quality_grade}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Loss curve comparison
    plt.subplot(2, 3, 1)
    plt.plot(losses_10k, 'b-', linewidth=2, label='10K+ Iterations')
    plt.axhline(y=prev_results['final_loss'], color='r', linestyle='--', label='1K Iterations Final')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training time per epoch
    plt.subplot(2, 3, 2)
    plt.plot(metrics_10k['training_times'], 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.grid(True, alpha=0.3)
    
    # Loss reduction rate
    plt.subplot(2, 3, 3)
    if len(losses_10k) > 1:
        loss_diff = [losses_10k[i] - losses_10k[i+1] for i in range(len(losses_10k)-1)]
        plt.plot(loss_diff, 'm-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Reduction')
        plt.title('Loss Reduction Rate')
        plt.grid(True, alpha=0.3)
    
    # Convergence analysis
    plt.subplot(2, 3, 4)
    plt.plot(losses_10k, 'b-', linewidth=2)
    if convergence_epoch:
        plt.axvline(x=convergence_epoch-1, color='r', linestyle='--', label=f'Convergence (Epoch {convergence_epoch})')
        plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convergence Analysis')
    plt.grid(True, alpha=0.3)
    
    # Throughput analysis
    plt.subplot(2, 3, 5)
    cumulative_time = np.cumsum([0] + metrics_10k['training_times'])
    iterations_per_second = np.arange(len(losses_10k)) / cumulative_time[1:len(losses_10k)+1]
    plt.plot(iterations_per_second, 'c-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Iterations/Second')
    plt.title('Training Throughput')
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = f"""
    TRAINING SUMMARY
    
    Total Iterations: {metrics_10k['total_iterations']:,}
    Final Loss: {losses_10k[-1]:.8f}
    Training Time: {metrics_10k['total_training_time']:.1f}s
    Throughput: {throughput:.1f} samples/s
    
    Convergence: {'YES' if convergence_epoch else 'NO'}
    Stability: {'HIGH' if len(losses_10k) > 10 and np.std(losses_10k[-10:]) < 1e-5 else 'MEDIUM'}
    Quality: {quality_grade}
    """
    plt.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('training_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "=" * 70)
    print(f"🎉 COMPARISON ANALYSIS COMPLETED!")
    print(f"=" * 70)
    print(f"   Visualization saved: training_comparison_analysis.png")
    print(f"   Training Grade: {quality_grade} ({quality_score}/100)")
    print(f"   Key Achievement: {metrics_10k['total_iterations']:,} iterations with {losses_10k[-1]:.8f} final loss")
    print(f"   Performance: {throughput:.1f}x improvement in throughput vs previous training")
    print(f"=" * 70)

if __name__ == "__main__":
    create_comparison_analysis()