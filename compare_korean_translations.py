#!/usr/bin/env python3
"""
Comprehensive comparison of Korean translation results with expected translations.
This script analyzes the quality of translations and provides detailed comparison.
"""

import json
import re
from difflib import SequenceMatcher

def load_results():
    """Load the Korean inference test results."""
    try:
        with open('korean_inference_trained_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Results file not found. Please run the inference test first.")
        return None

def calculate_similarity(predicted, expected):
    """Calculate similarity between predicted and expected translations."""
    if not predicted or not expected:
        return 0.0
    
    # Normalize both strings
    pred_normalized = predicted.lower().strip()
    exp_normalized = expected.lower().strip()
    
    # Calculate similarity using SequenceMatcher
    similarity = SequenceMatcher(None, pred_normalized, exp_normalized).ratio()
    
    return similarity

def analyze_translation_quality(predicted, expected, korean_input):
    """Analyze translation quality with multiple metrics."""
    
    # Basic metrics
    pred_words = predicted.split()
    exp_words = expected.split()
    
    # Word-level similarity
    word_matches = 0
    for exp_word in exp_words:
        if exp_word.lower() in predicted.lower():
            word_matches += 1
    
    word_precision = word_matches / len(pred_words) if pred_words else 0
    word_recall = word_matches / len(exp_words) if exp_words else 0
    word_f1 = 2 * (word_precision * word_recall) / (word_precision + word_recall) if (word_precision + word_recall) > 0 else 0
    
    # Character-level similarity
    char_similarity = calculate_similarity(predicted, expected)
    
    # Length ratio (how close are the lengths)
    length_ratio = min(len(predicted), len(expected)) / max(len(predicted), len(expected)) if max(len(predicted), len(expected)) > 0 else 0
    
    # Semantic analysis (basic heuristics)
    semantic_score = 0
    
    # Check for key semantic elements
    if korean_input and predicted:
        # Basic semantic mapping heuristics
        korean_meanings = {
            '안녕': ['hello', 'hi', 'greetings'],
            '감사': ['thank', 'thanks', 'grateful'],
            '안녕히': ['goodbye', 'farewell'],
            '날씨': ['weather'],
            '번역': ['translation', 'translate'],
            '시간': ['time', 'hour'],
            '어디': ['where'],
            '얼마': ['how much', 'price'],
            '도와': ['help', 'assist'],
            '배고파': ['hungry'],
            '목말라': ['thirsty'],
            '피곤': ['tired'],
            '행복': ['happy', 'happiness'],
            '슬퍼': ['sad'],
            '컴퓨터': ['computer'],
            '이해': ['understand'],
            '정확한': ['accurate', 'correct'],
            '결과': ['result'],
            '프로그램': ['program'],
            '알고리즘': ['algorithm'],
            '한국어': ['korean'],
            '영어': ['english']
        }
        
        # Check if key Korean concepts appear in predicted translation
        for korean_key, english_equivalents in korean_meanings.items():
            if korean_key in korean_input:
                for eng_word in english_equivalents:
                    if eng_word in predicted.lower():
                        semantic_score += 1
                        break
    
    semantic_score = min(semantic_score / max(1, len(exp_words) // 2), 1.0)  # Normalize
    
    return {
        'char_similarity': char_similarity,
        'word_precision': word_precision,
        'word_recall': word_recall,
        'word_f1': word_f1,
        'length_ratio': length_ratio,
        'semantic_score': semantic_score,
        'overall_quality': (char_similarity * 0.4 + word_f1 * 0.3 + semantic_score * 0.3)
    }

def categorize_translation_quality(quality_score):
    """Categorize translation quality."""
    if quality_score >= 0.8:
        return "EXCELLENT"
    elif quality_score >= 0.6:
        return "GOOD"
    elif quality_score >= 0.4:
        return "FAIR"
    elif quality_score >= 0.2:
        return "POOR"
    else:
        return "VERY POOR"

def main():
    """Main analysis function."""
    
    print("🔍 KOREAN TRANSLATION QUALITY ANALYSIS")
    print("=" * 80)
    
    results = load_results()
    if not results:
        return
    
    test_results = results['results']
    
    print("📊 ANALYZING TRANSLATION QUALITY")
    print("=" * 80)
    
    all_quality_scores = []
    category_scores = {}
    
    for category, category_results in test_results.items():
        print(f"\n📋 {category.upper()}")
        print("-" * 60)
        
        category_scores[category] = []
        
        for i, result in enumerate(category_results, 1):
            korean_input = result['input']
            predicted = result['predicted']
            expected = result['expected']
            inference_time = result['inference_time_ms']
            
            # Skip empty inputs
            if not korean_input:
                continue
            
            # Analyze translation quality
            quality_metrics = analyze_translation_quality(predicted, expected, korean_input)
            quality_score = quality_metrics['overall_quality']
            all_quality_scores.append(quality_score)
            category_scores[category].append(quality_score)
            
            # Display detailed analysis
            print(f"  Test {i}: '{korean_input}'")
            print(f"    Predicted: '{predicted}'")
            print(f"    Expected:  '{expected}'")
            print(f"    Time: {inference_time:.2f}ms")
            print(f"    Quality Metrics:")
            print(f"      • Character Similarity: {quality_metrics['char_similarity']:.3f}")
            print(f"      • Word F1-Score: {quality_metrics['word_f1']:.3f}")
            print(f"      • Semantic Score: {quality_metrics['semantic_score']:.3f}")
            print(f"      • Length Ratio: {quality_metrics['length_ratio']:.3f}")
            print(f"      • OVERALL QUALITY: {quality_score:.3f} ({categorize_translation_quality(quality_score)})")
            print()
        
        # Category summary
        if category_scores[category]:
            cat_avg = sum(category_scores[category]) / len(category_scores[category])
            print(f"  📊 Category Average Quality: {cat_avg:.3f} ({categorize_translation_quality(cat_avg)})")
            print()
    
    # Overall analysis
    print("\n" + "=" * 80)
    print("🎯 OVERALL TRANSLATION QUALITY ANALYSIS")
    print("=" * 80)
    
    if all_quality_scores:
        avg_quality = sum(all_quality_scores) / len(all_quality_scores)
        min_quality = min(all_quality_scores)
        max_quality = max(all_quality_scores)
        
        # Quality distribution
        excellent = sum(1 for score in all_quality_scores if score >= 0.8)
        good = sum(1 for score in all_quality_scores if 0.6 <= score < 0.8)
        fair = sum(1 for score in all_quality_scores if 0.4 <= score < 0.6)
        poor = sum(1 for score in all_quality_scores if 0.2 <= score < 0.4)
        very_poor = sum(1 for score in all_quality_scores if score < 0.2)
        
        total_tests = len(all_quality_scores)
        
        print(f"📈 QUALITY METRICS:")
        print(f"   Average Quality Score: {avg_quality:.3f}")
        print(f"   Quality Range: {min_quality:.3f} - {max_quality:.3f}")
        print(f"   Overall Grade: {categorize_translation_quality(avg_quality)}")
        print()
        
        print(f"📊 QUALITY DISTRIBUTION:")
        print(f"   EXCELLENT (≥0.8): {excellent}/{total_tests} ({excellent/total_tests*100:.1f}%)")
        print(f"   GOOD (0.6-0.8): {good}/{total_tests} ({good/total_tests*100:.1f}%)")
        print(f"   FAIR (0.4-0.6): {fair}/{total_tests} ({fair/total_tests*100:.1f}%)")
        print(f"   POOR (0.2-0.4): {poor}/{total_tests} ({poor/total_tests*100:.1f}%)")
        print(f"   VERY POOR (<0.2): {very_poor}/{total_tests} ({very_poor/total_tests*100:.1f}%)")
        print()
        
        # Category ranking
        print(f"🏆 CATEGORY RANKING (by average quality):")
        sorted_categories = sorted(category_scores.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0, reverse=True)
        for i, (category, scores) in enumerate(sorted_categories, 1):
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"   {i}. {category}: {avg_score:.3f} ({categorize_translation_quality(avg_score)})")
        print()
        
        # Performance insights
        print(f"🔍 KEY INSIGHTS:")
        if avg_quality >= 0.8:
            print(f"   ✅ EXCELLENT overall translation quality")
        elif avg_quality >= 0.6:
            print(f"   ✅ GOOD overall translation quality")
        elif avg_quality >= 0.4:
            print(f"   ⚠️ FAIR translation quality - room for improvement")
        else:
            print(f"   ❌ POOR translation quality - significant improvement needed")
        
        if excellent + good > total_tests * 0.7:
            print(f"   ✅ High percentage of good/excellent translations")
        else:
            print(f"   ⚠️ Significant portion of translations need improvement")
        
        print()
        
        # Recommendations
        print(f"💡 RECOMMENDATIONS:")
        if avg_quality < 0.6:
            print(f"   • Consider increasing training data or epochs")
            print(f"   • Review vocabulary coverage for Korean-English pairs")
            print(f"   • Implement attention mechanisms or better alignment")
        
        if very_poor > total_tests * 0.1:
            print(f"   • Address systematic translation failures")
            print(f"   • Check for data quality issues in training set")
        
        print(f"   • Monitor performance on longer, more complex sentences")
        print(f"   • Consider domain-specific training data")
        
        print()
        print("=" * 80)
        print(f"🎉 ANALYSIS COMPLETE!")
        print(f"   Overall Quality: {avg_quality:.3f} ({categorize_translation_quality(avg_quality)})")
        print(f"   Tests Analyzed: {total_tests}")
        print("=" * 80)
    
    else:
        print("❌ No valid test results found for analysis.")

if __name__ == "__main__":
    main()