#!/usr/bin/env python3
"""
Final analysis of Korean inference test results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_inference_results():
    """Analyze the Korean inference test results comprehensively."""
    
    print("📊 CVM TRANSFORMER - KOREAN INFERENCE ANALYSIS")
    print("=" * 80)
    
    try:
        with open('korean_inference_trained_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ Results file not found. Please run the inference test first.")
        return
    
    # Extract data
    config = results['config']
    performance = results['performance']
    quality = results['quality']
    test_results = results['results']
    
    print(f"🎯 TEST CONFIGURATION:")
    print(f"   Device: {config['device']}")
    print(f"   Model Parameters: {config['model_params']:,}")
    print(f"   Model Size: {config['model_size_mb']:.1f} MB")
    print(f"   Vocabulary Size: {config['vocab_size']} tokens")
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   Total Tests: {performance['total_tests']}")
    print(f"   Average Inference Time: {performance['avg_time_ms']:.2f}ms")
    print(f"   Throughput: {performance['throughput_per_second']:.1f} inferences/second")
    print(f"   Min Latency: {performance['min_latency_ms']:.2f}ms")
    print(f"   Max Latency: {performance['max_latency_ms']:.2f}ms")
    print(f"   Median Latency: {performance['median_latency_ms']:.2f}ms")
    
    print(f"\n🎯 QUALITY ASSESSMENT:")
    print(f"   Valid Outputs: {quality['valid_outputs']}/{performance['total_tests']} ({quality['valid_output_percentage']:.1f}%)")
    print(f"   Empty Outputs: {quality['empty_outputs']}/{performance['total_tests']} ({quality['empty_outputs']/performance['total_tests']*100:.1f}%)")
    print(f"   Reasonable Outputs: {quality['reasonable_outputs']}/{performance['total_tests']} ({quality['reasonable_output_percentage']:.1f}%)")
    
    # Detailed category analysis
    print(f"\n📋 CATEGORY-BY-CATEGORY ANALYSIS:")
    print("-" * 60)
    
    category_stats = {}
    for category, category_results in test_results.items():
        if category_results:
            times = [r['inference_time_ms'] for r in category_results]
            input_lengths = [r['input_length'] for r in category_results]
            output_lengths = [r['output_length'] for r in category_results]
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_input_len = np.mean(input_lengths)
            avg_output_len = np.mean(output_lengths)
            
            category_stats[category] = {
                'count': len(category_results),
                'avg_time': avg_time,
                'std_time': std_time,
                'avg_input_len': avg_input_len,
                'avg_output_len': avg_output_len,
                'efficiency': avg_output_len / avg_input_len if avg_input_len > 0 else 0
            }
            
            print(f"   {category}:")
            print(f"      Tests: {len(category_results)}")
            print(f"      Avg Time: {avg_time:.2f}ms ± {std_time:.2f}ms")
            print(f"      Avg Input Length: {avg_input_len:.1f} chars")
            print(f"      Avg Output Length: {avg_output_len:.1f} chars")
            print(f"      Length Efficiency: {category_stats[category]['efficiency']:.2f}")
            print()
    
    # Performance grade analysis
    avg_time = performance['avg_time_ms']
    if avg_time < 5:
        performance_grade = "EXCELLENT"
        grade_color = "🟢"
    elif avg_time < 10:
        performance_grade = "GOOD"
        grade_color = "🟡"
    elif avg_time < 20:
        performance_grade = "FAIR"
        grade_color = "🟠"
    else:
        performance_grade = "NEEDS OPTIMIZATION"
        grade_color = "🔴"
    
    print(f"\n🏆 OVERALL PERFORMANCE GRADE: {grade_color} {performance_grade}")
    
    # Real-time capability assessment
    realtime_threshold = 500  # 500ms for real-time translation
    if avg_time < realtime_threshold:
        realtime_capability = "✅ REAL-TIME CAPABLE"
        realtime_margin = realtime_threshold / avg_time
        print(f"   Real-time Capability: {realtime_capability} ({realtime_margin:.1f}x margin)")
    else:
        print(f"   Real-time Capability: ❌ NOT REAL-TIME CAPABLE")
    
    # Edge device suitability
    model_size_mb = config['model_size_mb']
    if model_size_mb < 50:
        edge_suitability = "✅ EXCELLENT for edge devices"
    elif model_size_mb < 100:
        edge_suitability = "✅ GOOD for edge devices"
    elif model_size_mb < 200:
        edge_suitability = "⚠️ MODERATE for edge devices"
    else:
        edge_suitability = "❌ LARGE for edge devices"
    
    print(f"   Edge Device Suitability: {edge_suitability}")
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # 1. Latency distribution
    plt.subplot(3, 3, 1)
    all_times = []
    for category_results in test_results.values():
        all_times.extend([r['inference_time_ms'] for r in category_results])
    
    if all_times:
        plt.hist(all_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(all_times), color='red', linestyle='--', label=f'Mean: {np.mean(all_times):.2f}ms')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Category performance comparison
    plt.subplot(3, 3, 2)
    categories = list(category_stats.keys())
    avg_times = [category_stats[cat]['avg_time'] for cat in categories]
    
    if categories and avg_times:
        plt.bar(range(len(categories)), avg_times, color='lightgreen', alpha=0.7)
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        plt.ylabel('Average Time (ms)')
        plt.title('Performance by Category')
        plt.grid(True, alpha=0.3)
    
    # 3. Input vs Output length analysis
    plt.subplot(3, 3, 3)
    all_input_lengths = []
    all_output_lengths = []
    for category_results in test_results.values():
        all_input_lengths.extend([r['input_length'] for r in category_results])
        all_output_lengths.extend([r['output_length'] for r in category_results])
    
    if all_input_lengths and all_output_lengths:
        plt.scatter(all_input_lengths, all_output_lengths, alpha=0.6, color='orange')
        plt.xlabel('Input Length (chars)')
        plt.ylabel('Output Length (chars)')
        plt.title('Input vs Output Length')
        plt.grid(True, alpha=0.3)
        
        # Add correlation line
        if len(all_input_lengths) > 1:
            correlation = np.corrcoef(all_input_lengths, all_output_lengths)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Throughput analysis
    plt.subplot(3, 3, 4)
    throughput = performance['throughput_per_second']
    categories = ['Current Model', 'Target (100/s)', 'Target (200/s)', 'Target (500/s)']
    throughputs = [throughput, 100, 200, 500]
    colors = ['green' if throughput >= target else 'red' for target in [100, 200, 500]]
    colors.insert(0, 'blue')
    
    plt.bar(categories, throughputs, color=colors, alpha=0.7)
    plt.ylabel('Inferences/Second')
    plt.title('Throughput Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 5. Quality metrics
    plt.subplot(3, 3, 5)
    quality_metrics = ['Valid Outputs', 'Empty Outputs', 'Reasonable Outputs']
    quality_values = [quality['valid_output_percentage'], 
                     quality['empty_outputs']/performance['total_tests']*100,
                     quality['reasonable_output_percentage']]
    colors = ['green', 'red', 'blue']
    
    plt.pie(quality_values, labels=quality_metrics, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Output Quality Distribution')
    
    # 6. Model size comparison
    plt.subplot(3, 3, 6)
    model_size = config['model_size_mb']
    size_categories = ['< 50MB', '50-100MB', '100-200MB', '> 200MB']
    size_ranges = [50, 100, 200, 500]
    current_category = next(i for i, threshold in enumerate(size_ranges) if model_size <= threshold)
    
    sizes = [50, 100, 200, 500]  # Max sizes for visualization
    colors = ['light' if i != current_category else 'dark' for i in range(len(sizes))]
    
    plt.bar(size_categories, sizes, color=['lightgreen', 'yellow', 'orange', 'lightcoral'])
    plt.axhline(model_size, color='red', linewidth=3, label=f'Current: {model_size:.1f}MB')
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Category efficiency heatmap
    plt.subplot(3, 3, 7)
    if category_stats:
        efficiencies = [category_stats[cat]['efficiency'] for cat in category_stats.keys()]
        avg_times = [category_stats[cat]['avg_time'] for cat in category_stats.keys()]
        
        plt.scatter(avg_times, efficiencies, s=100, alpha=0.7, c=range(len(efficiencies)), cmap='viridis')
        plt.xlabel('Average Time (ms)')
        plt.ylabel('Length Efficiency')
        plt.title('Category Performance Matrix')
        plt.grid(True, alpha=0.3)
        
        for i, cat in enumerate(category_stats.keys()):
            plt.annotate(cat[:8], (avg_times[i], efficiencies[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    # 8. Real-time performance indicator
    plt.subplot(3, 3, 8)
    realtime_requirements = [10, 50, 100, 500]  # ms thresholds
    colors = ['green' if avg_time <= req else 'red' for req in realtime_requirements]
    
    plt.bar(['<10ms', '<50ms', '<100ms', '<500ms'], 
            [1 if avg_time <= req else 0 for req in realtime_requirements], 
            color=colors, alpha=0.7)
    plt.ylabel('Meets Requirement')
    plt.title('Real-time Performance')
    plt.ylim(0, 1.2)
    plt.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    plt.subplot(3, 3, 9)
    plt.axis('off')
    summary_text = f"""
    INFERENCE SUMMARY
    
    Performance: {performance_grade}
    Avg Latency: {avg_time:.2f}ms
    Throughput: {performance['throughput_per_second']:.1f}/s
    
    Quality: {quality['reasonable_output_percentage']:.1f}% reasonable
    Model Size: {model_size_mb:.1f}MB
    
    Real-time: {'✅' if avg_time < realtime_threshold else '❌'}
    Edge Ready: {'✅' if model_size_mb < 100 else '❌'}
    
    Total Tests: {performance['total_tests']}
    Categories: {len(test_results)}
    """
    
    plt.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('korean_inference_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final recommendations
    print(f"\n🎯 KEY FINDINGS & RECOMMENDATIONS:")
    print("=" * 60)
    
    print(f"\n✅ STRENGTHS:")
    if avg_time < 10:
        print(f"   • EXCELLENT latency: {avg_time:.2f}ms (sub-10ms)")
    if quality['reasonable_output_percentage'] > 90:
        print(f"   • HIGH quality outputs: {quality['reasonable_output_percentage']:.1f}%")
    if performance['throughput_per_second'] > 100:
        print(f"   • GOOD throughput: {performance['throughput_per_second']:.1f} inferences/s")
    if model_size_mb < 100:
        print(f"   • COMPACT model size: {model_size_mb:.1f}MB")
    
    print(f"\n⚠️  AREAS FOR IMPROVEMENT:")
    if quality['valid_output_percentage'] < 95:
        print(f"   • OUTPUT quality could be improved: {quality['valid_output_percentage']:.1f}% valid")
    if avg_time > 20:
        print(f"   • LATENCY optimization needed: {avg_time:.2f}ms")
    if model_size_mb > 200:
        print(f"   • MODEL size optimization: {model_size_mb:.1f}MB")
    
    print(f"\n🚀 DEPLOYMENT READINESS:")
    if avg_time < realtime_threshold and model_size_mb < 100 and quality['reasonable_output_percentage'] > 80:
        print(f"   • 🟢 PRODUCTION READY - Excellent performance across all metrics")
    elif avg_time < realtime_threshold and quality['reasonable_output_percentage'] > 70:
        print(f"   • 🟡 PRODUCTION READY with minor optimizations")
    else:
        print(f"   • 🔴 NEEDS OPTIMIZATION before production deployment")
    
    print(f"\n" + "=" * 60)
    print(f"🎉 ANALYSIS COMPLETE!")
    print(f"   Visualization saved: korean_inference_analysis.png")
    print(f"   Performance Grade: {performance_grade}")
    print(f"   Deployment Status: {'READY' if avg_time < realtime_threshold else 'NEEDS WORK'}")
    print("=" * 60)

if __name__ == "__main__":
    analyze_inference_results()