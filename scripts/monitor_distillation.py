#!/usr/bin/env python3
"""
Monitor the distillation process and generate progress report.
"""

import json
import torch
import os
import time
from datetime import datetime

def monitor_distillation_progress():
    """Monitor the current distillation progress."""
    
    output_dir = "distillation_output"
    
    if not os.path.exists(output_dir):
        print("❌ No distillation output directory found")
        return
    
    # Check for checkpoint files
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(checkpoints_dir):
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
        print(f"📁 Found {len(checkpoints)} checkpoint files:")
        for checkpoint in sorted(checkpoints):
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
            print(f"   {checkpoint}: {file_size:.1f} MB (modified: {mod_time.strftime('%H:%M:%S')})")
    
    # Check for reports
    reports = []
    if os.path.exists(os.path.join(output_dir, "training_report.json")):
        reports.append("training_report.json")
    if os.path.exists(os.path.join(output_dir, "distillation_summary.txt")):
        reports.append("distillation_summary.txt")
    
    print(f"\n📊 Found {len(reports)} report files: {', '.join(reports)}")
    
    # Try to load and display training report if available
    training_report_path = os.path.join(output_dir, "training_report.json")
    if os.path.exists(training_report_path):
        try:
            with open(training_report_path, 'r') as f:
                report = json.load(f)
            
            print("\n📈 Current Training Status:")
            
            # Training summary
            if 'training_summary' in report:
                summary = report['training_summary']
                print(f"   Total Iterations: {summary.get('total_iterations', 0):,} / {summary.get('target_iterations', 10000):,}")
                print(f"   Training Success: {'✅ YES' if summary.get('training_success', False) else '❌ NO'}")
                print(f"   Early Stop Triggered: {'Yes' if summary.get('early_stop_triggered', False) else 'No'}")
            
            # Quality assurance
            if 'quality_assurance' in report:
                qa = report['quality_assurance']
                print(f"\n🔍 Quality Assurance:")
                print(f"   Final Quality Score: {qa.get('final_quality_score', 0):.4f}")
                print(f"   Best Quality Score: {qa.get('best_quality_score', 0):.4f}")
                print(f"   Quality Retention: {qa.get('quality_retention_percentage', 0):.1f}%")
                print(f"   Meets 90% Requirement: {'✅ YES' if qa.get('meets_90_percent_requirement', False) else '❌ NO'}")
            
            # Model analysis
            if 'model_analysis' in report:
                analysis = report['model_analysis']
                print(f"\n🤖 Model Analysis:")
                print(f"   Student Parameters: {analysis.get('student_model_parameters', 0):,}")
                print(f"   Compression Ratio: {analysis.get('compression_ratio', 0):.1f}x")
                print(f"   Model Size: {analysis.get('model_size_mb', 0):.1f} MB")
                print(f"   Quantization Applied: {'Yes' if analysis.get('quantization_applied', False) else 'No'}")
            
        except Exception as e:
            print(f"⚠️  Could not read training report: {e}")
    
    # Check if process is still running
    print(f"\n⏱️  Current Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Estimate completion time if possible
    if os.path.exists(checkpoints_dir):
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith('checkpoint_')]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            if 'checkpoint_' in latest_checkpoint:
                try:
                    iteration = int(latest_checkpoint.replace('checkpoint_', '').replace('.pt', ''))
                    progress = iteration / 10000 * 100
                    print(f"   Progress: {progress:.1f}% ({iteration:,} / 10,000 iterations)")
                    
                    # Rough time estimation (this is very approximate)
                    if len(checkpoints) > 1:
                        # Get time difference between first and last checkpoint
                        first_checkpoint = sorted(checkpoints)[0]
                        last_checkpoint = sorted(checkpoints)[-1]
                        
                        first_time = os.path.getmtime(os.path.join(checkpoints_dir, first_checkpoint))
                        last_time = os.path.getmtime(os.path.join(checkpoints_dir, last_checkpoint))
                        
                        time_diff = last_time - first_time
                        iterations_diff = iteration - 1000  # Assuming first checkpoint is at 1000
                        
                        if iterations_diff > 0 and time_diff > 0:
                            iterations_per_second = iterations_diff / time_diff
                            remaining_iterations = 10000 - iteration
                            remaining_time = remaining_iterations / iterations_per_second
                            
                            hours = remaining_time / 3600
                            print(f"   Estimated Time Remaining: {hours:.1f} hours")
                            
                except ValueError:
                    pass

if __name__ == "__main__":
    print("🔍 Monitoring Distillation Process")
    print("=" * 50)
    monitor_distillation_progress()
    print("=" * 50)