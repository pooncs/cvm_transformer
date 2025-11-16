#!/usr/bin/env python3
"""
Enhanced Validation Protocol for Model Distillation
Implements comprehensive validation with quality degradation detection and rollback capability
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time

try:
    from sacrebleu import corpus_bleu, sentence_bleu
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score

    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    logging.warning(
        "Some metrics libraries not available. Install with: pip install sacrebleu rouge-score bert-score"
    )

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation protocol"""

    validation_frequency: int = 1000  # Validate every N iterations
    quality_threshold: float = 0.9  # 90% quality retention requirement
    degradation_threshold: float = 0.05  # 5% degradation triggers rollback
    rollback_checkpoints: int = 3  # Number of checkpoints to keep for rollback
    min_improvement_threshold: float = 0.01  # Minimum improvement to continue training
    early_stopping_patience: int = 5  # Stop if no improvement for N validations
    save_best_only: bool = True  # Only save models that improve validation metrics
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "bleu": 0.4,
            "rouge_l": 0.3,
            "ter": 0.2,
            "bert_score": 0.1,
        }
    )


@dataclass
class ValidationResult:
    """Results from a validation checkpoint"""

    iteration: int
    timestamp: float
    metrics: Dict[str, float]
    quality_score: float
    teacher_metrics: Optional[Dict[str, float]] = None
    comparison_ratios: Optional[Dict[str, float]] = None
    degradation_detected: bool = False
    rollback_recommended: bool = False
    error_analysis: Optional[Dict] = None


class TranslationQualityEvaluator:
    """Comprehensive translation quality evaluation with multiple metrics"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.rouge_scorer = (
            rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            if HAS_METRICS
            else None
        )

    def compute_comprehensive_metrics(
        self, hypotheses: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute comprehensive translation quality metrics"""
        if not HAS_METRICS:
            return self._compute_fallback_metrics(hypotheses, references)

        results = {}

        # BLEU scores
        try:
            bleu_score = corpus_bleu(hypotheses, [references]).score
            results["bleu"] = bleu_score / 100.0  # Normalize to 0-1
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            results["bleu"] = 0.0

        # ROUGE scores
        try:
            rouge_scores = []
            for hyp, ref in zip(hypotheses, references):
                scores = self.rouge_scorer.score(ref, hyp)
                rouge_scores.append(scores)

            rouge1 = np.mean([s["rouge1"].fmeasure for s in rouge_scores])
            rouge2 = np.mean([s["rouge2"].fmeasure for s in rouge_scores])
            rouge_l = np.mean([s["rougeL"].fmeasure for s in rouge_scores])

            results["rouge1"] = rouge1
            results["rouge2"] = rouge2
            results["rouge_l"] = rouge_l
        except Exception as e:
            logger.warning(f"ROUGE computation failed: {e}")
            results.update({"rouge1": 0.0, "rouge2": 0.0, "rouge_l": 0.0})

        # TER scores (Translation Error Rate)
        try:
            ter_scores = []
            for hyp, ref in zip(hypotheses, references):
                # Simple TER approximation
                ter_score = self._compute_ter(hyp, ref)
                ter_scores.append(ter_score)
            results["ter"] = 1.0 - np.mean(ter_scores)  # Convert to accuracy (1 - TER)
        except Exception as e:
            logger.warning(f"TER computation failed: {e}")
            results["ter"] = 0.0

        # BERTScore
        try:
            P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)
            results["bert_score"] = float(F1.mean())
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            results["bert_score"] = 0.0

        return results

    def _compute_fallback_metrics(
        self, hypotheses: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Fallback metrics when external libraries are not available"""
        results = {}

        # Simple word overlap (BLEU-like)
        bleu_scores = []
        for hyp, ref in zip(hypotheses, references):
            hyp_tokens = hyp.lower().split()
            ref_tokens = ref.lower().split()

            if len(ref_tokens) == 0:
                bleu_scores.append(0.0)
                continue

            # Simple precision
            matches = sum(1 for token in hyp_tokens if token in ref_tokens)
            precision = matches / len(hyp_tokens) if hyp_tokens else 0.0

            # Simple recall
            recall = matches / len(ref_tokens) if ref_tokens else 0.0

            # F1-like score
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            bleu_scores.append(f1)

        results["bleu"] = np.mean(bleu_scores)
        results["rouge1"] = results["bleu"]  # Use same for fallback
        results["rouge2"] = results["bleu"] * 0.8  # Slightly lower
        results["rouge_l"] = results["bleu"] * 0.9
        results["ter"] = 1.0 - results["bleu"]  # Inverse relationship
        results["bert_score"] = results["bleu"] * 0.95  # Slightly optimistic

        return results

    def _compute_ter(self, hypothesis: str, reference: str) -> float:
        """Simple Translation Error Rate approximation"""
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()

        if len(ref_tokens) == 0:
            return 1.0

        # Count edits needed (simplified)
        edits = 0
        max_len = max(len(hyp_tokens), len(ref_tokens))

        for i in range(max_len):
            if i >= len(hyp_tokens):
                edits += 1  # Deletion
            elif i >= len(ref_tokens):
                edits += 1  # Insertion
            elif hyp_tokens[i] != ref_tokens[i]:
                edits += 1  # Substitution

        return edits / len(ref_tokens)


class QualityDegradationDetector:
    """Detects quality degradation in training progress"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.history = deque(maxlen=config.rollback_checkpoints)
        self.trend_window = 3  # Number of points to consider for trend

    def detect_degradation(
        self, current_result: ValidationResult, history: List[ValidationResult]
    ) -> Tuple[bool, str]:
        """Detect if quality is degrading and provide reason"""
        if not history:
            return False, "Insufficient history"

        # Check against quality threshold
        if current_result.quality_score < self.config.quality_threshold:
            return (
                True,
                f"Quality score {current_result.quality_score:.3f} below threshold {self.config.quality_threshold}",
            )

        # Check for degradation trend
        if len(history) >= self.trend_window:
            recent_scores = [h.quality_score for h in history[-self.trend_window :]]
            current_score = current_result.quality_score

            # Calculate trend
            if len(recent_scores) >= 2:
                trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                if trend < -self.config.degradation_threshold:
                    return True, f"Degrading trend detected: slope {trend:.4f}"

        # Check individual metrics
        if current_result.comparison_ratios:
            for metric, ratio in current_result.comparison_ratios.items():
                if ratio < self.config.quality_threshold:
                    return True, f"Metric {metric} ratio {ratio:.3f} below threshold"

        # Check for sudden drops
        if history:
            previous_best = max(h.quality_score for h in history)
            drop = previous_best - current_result.quality_score
            if drop > self.config.degradation_threshold * 2:
                return True, f"Sudden quality drop: {drop:.3f}"

        return False, "No degradation detected"

    def should_rollback(
        self, current_result: ValidationResult, history: List[ValidationResult]
    ) -> bool:
        """Determine if rollback is recommended"""
        degradation_detected, reason = self.detect_degradation(current_result, history)

        if degradation_detected:
            logger.warning(f"Quality degradation detected: {reason}")

            # Check if we have better checkpoints to rollback to
            if history:
                best_quality = max(h.quality_score for h in history)
                if (
                    best_quality
                    > current_result.quality_score + self.config.degradation_threshold
                ):
                    return True

        return False

    def get_best_checkpoint(self, history: List[ValidationResult]) -> Optional[int]:
        """Get the iteration number of the best checkpoint"""
        if not history:
            return None

        best_result = max(history, key=lambda x: x.quality_score)
        return best_result.iteration


class DistillationValidator:
    """Main validation orchestrator for model distillation"""

    def __init__(
        self, config: ValidationConfig, teacher_model: Optional[nn.Module] = None
    ):
        self.config = config
        self.teacher_model = teacher_model
        self.evaluator = TranslationQualityEvaluator(config)
        self.degradation_detector = QualityDegradationDetector(config)
        self.validation_history = []
        self.best_quality = 0.0
        self.best_iteration = 0
        self.patience_counter = 0

    def validate_distillation(
        self,
        student_model: nn.Module,
        validation_data: List[Tuple[str, str]],
        iteration: int,
        device: str = "cpu",
    ) -> ValidationResult:
        """Perform comprehensive validation of student model"""
        logger.info(f"🔍 Validation checkpoint at iteration {iteration}")

        start_time = time.time()

        # Generate translations
        student_translations = []
        teacher_translations = []
        references = []

        student_model.eval()
        if self.teacher_model:
            self.teacher_model.eval()

        with torch.no_grad():
            for source, target in validation_data:
                # Student translation
                student_output = self._generate_translation(
                    student_model, source, device
                )
                student_translations.append(student_output)

                # Teacher translation (if available)
                if self.teacher_model:
                    teacher_output = self._generate_translation(
                        self.teacher_model, source, device
                    )
                    teacher_translations.append(teacher_output)

                references.append(target)

        # Compute metrics
        student_metrics = self.evaluator.compute_comprehensive_metrics(
            student_translations, references
        )

        teacher_metrics = None
        comparison_ratios = None

        if self.teacher_model and teacher_translations:
            teacher_metrics = self.evaluator.compute_comprehensive_metrics(
                teacher_translations, references
            )
            comparison_ratios = self._compute_comparison_ratios(
                student_metrics, teacher_metrics
            )

        # Compute overall quality score
        quality_score = self._compute_quality_score(student_metrics, comparison_ratios)

        # Error analysis
        error_analysis = self._perform_error_analysis(
            student_translations, references, student_metrics
        )

        # Create validation result
        validation_result = ValidationResult(
            iteration=iteration,
            timestamp=time.time(),
            metrics=student_metrics,
            quality_score=quality_score,
            teacher_metrics=teacher_metrics,
            comparison_ratios=comparison_ratios,
            error_analysis=error_analysis,
        )

        # Check for degradation
        degradation_detected, degradation_reason = (
            self.degradation_detector.detect_degradation(
                validation_result, self.validation_history
            )
        )
        validation_result.degradation_detected = degradation_detected

        rollback_recommended = self.degradation_detector.should_rollback(
            validation_result, self.validation_history
        )
        validation_result.rollback_recommended = rollback_recommended

        # Update history and tracking
        self.validation_history.append(validation_result)

        if quality_score > self.best_quality:
            self.best_quality = quality_score
            self.best_iteration = iteration
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Log results
        self._log_validation_results(
            validation_result, degradation_reason, rollback_recommended
        )

        logger.info(f"✅ Validation completed in {time.time() - start_time:.2f}s")

        return validation_result

    def _generate_translation(self, model: nn.Module, source: str, device: str) -> str:
        """Generate translation using model"""
        # This is a simplified version - in practice, you'd use proper decoding
        try:
            # Simulate translation generation
            # In real implementation, this would use beam search or other decoding
            if hasattr(model, "generate"):
                return model.generate(source)
            else:
                # Fallback for models without generate method
                return f"Translated: {source}"
        except Exception as e:
            logger.warning(f"Translation generation failed: {e}")
            return source  # Return source as fallback

    def _compute_comparison_ratios(
        self, student_metrics: Dict[str, float], teacher_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute ratios between student and teacher metrics"""
        ratios = {}
        for metric in student_metrics:
            if metric in teacher_metrics and teacher_metrics[metric] > 0:
                ratios[metric] = student_metrics[metric] / teacher_metrics[metric]
            else:
                ratios[metric] = 1.0
        return ratios

    def _compute_quality_score(
        self,
        metrics: Dict[str, float],
        comparison_ratios: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute overall quality score"""
        if comparison_ratios:
            # Use comparison ratios if available
            weighted_score = sum(
                self.config.metric_weights.get(metric, 0.1) * ratio
                for metric, ratio in comparison_ratios.items()
            )
        else:
            # Use absolute metrics
            weighted_score = sum(
                self.config.metric_weights.get(metric, 0.1) * metrics[metric]
                for metric in metrics
            )

        return weighted_score

    def _perform_error_analysis(
        self, hypotheses: List[str], references: List[str], metrics: Dict[str, float]
    ) -> Dict:
        """Perform detailed error analysis"""
        analysis = {
            "total_samples": len(hypotheses),
            "average_length_ratio": np.mean(
                [
                    len(h.split()) / len(r.split()) if r.split() else 0
                    for h, r in zip(hypotheses, references)
                ]
            ),
            "length_variance": np.var([len(h.split()) for h in hypotheses]),
            "lowest_scoring_samples": [],
            "common_issues": [],
        }

        # Find lowest scoring samples
        sample_scores = []
        for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
            if HAS_METRICS:
                try:
                    score = sentence_bleu(hyp, [ref]).score / 100.0
                except:
                    score = 0.0
            else:
                # Fallback score
                hyp_tokens = hyp.lower().split()
                ref_tokens = ref.lower().split()
                matches = sum(1 for token in hyp_tokens if token in ref_tokens)
                score = matches / len(ref_tokens) if ref_tokens else 0.0

            sample_scores.append((i, score, hyp, ref))

        # Get worst samples
        sample_scores.sort(key=lambda x: x[1])
        analysis["lowest_scoring_samples"] = [
            {"index": idx, "score": score, "hypothesis": hyp, "reference": ref}
            for idx, score, hyp, ref in sample_scores[:3]
        ]

        return analysis

    def _log_validation_results(
        self,
        result: ValidationResult,
        degradation_reason: str,
        rollback_recommended: bool,
    ):
        """Log validation results"""
        logger.info(f"📊 Validation Results (Iteration {result.iteration}):")
        logger.info(f"   Quality Score: {result.quality_score:.4f}")
        logger.info(
            f"   Best Quality: {self.best_quality:.4f} (Iteration {self.best_iteration})"
        )

        for metric, value in result.metrics.items():
            logger.info(f"   {metric.upper()}: {value:.4f}")

        if result.comparison_ratios:
            logger.info("   Comparison with Teacher:")
            for metric, ratio in result.comparison_ratios.items():
                logger.info(f"     {metric}: {ratio:.3f}x")

        if result.degradation_detected:
            logger.warning(f"   ⚠️  Degradation detected: {degradation_reason}")

        if result.rollback_recommended:
            logger.error(f"   🔄 Rollback recommended!")

        if self.patience_counter > 0:
            logger.info(
                f"   Patience counter: {self.patience_counter}/{self.config.early_stopping_patience}"
            )

    def should_stop_training(self) -> Tuple[bool, str]:
        """Determine if training should stop early"""
        if self.patience_counter >= self.config.early_stopping_patience:
            return (
                True,
                f"Early stopping: no improvement for {self.patience_counter} validations",
            )

        if self.validation_history:
            recent_results = self.validation_history[-3:]
            if all(r.degradation_detected for r in recent_results):
                return True, "Consistent degradation detected"

        return False, ""

    def save_validation_history(self, filepath: str):
        """Save validation history to file"""
        history_data = []
        for result in self.validation_history:
            history_data.append(
                {
                    "iteration": result.iteration,
                    "timestamp": result.timestamp,
                    "metrics": result.metrics,
                    "quality_score": result.quality_score,
                    "teacher_metrics": result.teacher_metrics,
                    "comparison_ratios": result.comparison_ratios,
                    "degradation_detected": result.degradation_detected,
                    "rollback_recommended": result.rollback_recommended,
                }
            )

        with open(filepath, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"💾 Validation history saved to {filepath}")

    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        if not self.validation_history:
            return "No validation history available"

        report_lines = [
            "=" * 60,
            "MODEL DISTILLATION VALIDATION REPORT",
            "=" * 60,
            "",
            f"Total Validations: {len(self.validation_history)}",
            f"Best Quality Score: {self.best_quality:.4f} (Iteration {self.best_iteration})",
            f"Final Quality Score: {self.validation_history[-1].quality_score:.4f}",
            "",
            "Validation History:",
            "-" * 40,
        ]

        for result in self.validation_history:
            status = "🟢" if not result.degradation_detected else "🔴"
            report_lines.append(
                f"{status} Iteration {result.iteration:5d}: "
                f"Quality={result.quality_score:.4f} "
                f"BLEU={result.metrics.get('bleu', 0):.3f}"
            )

        # Add summary statistics
        quality_scores = [r.quality_score for r in self.validation_history]
        report_lines.extend(
            [
                "",
                "Summary Statistics:",
                f"  Mean Quality: {np.mean(quality_scores):.4f}",
                f"  Std Quality: {np.std(quality_scores):.4f}",
                f"  Min Quality: {np.min(quality_scores):.4f}",
                f"  Max Quality: {np.max(quality_scores):.4f}",
                "",
                f"Quality Retention: {(self.validation_history[-1].quality_score / self.best_quality * 100):.1f}%",
            ]
        )

        return "\n".join(report_lines)


if __name__ == "__main__":
    # Example usage
    config = ValidationConfig()
    validator = DistillationValidator(config)

    # Create sample validation data
    sample_data = [
        ("Hello world", "Hola mundo"),
        ("How are you?", "¿Cómo estás?"),
        ("Good morning", "Buenos días"),
    ]

    # Create a dummy model for testing
    class DummyModel(nn.Module):
        def forward(self, x):
            return x

    dummy_model = DummyModel()

    # Run validation
    result = validator.validate_distillation(dummy_model, sample_data, 1000)
    print(f"Quality Score: {result.quality_score:.4f}")
    print(f"Degradation Detected: {result.degradation_detected}")
    print(f"Rollback Recommended: {result.rollback_recommended}")
