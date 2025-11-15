import torch
import numpy as np
from typing import List, Union, Optional
import re
from collections import Counter
import math


class BLEUScore:
    """BLEU score calculation for translation evaluation."""
    
    def __init__(self, max_n: int = 4, weights: Optional[List[float]] = None):
        """
        Initialize BLEU scorer.
        
        Args:
            max_n: Maximum n-gram order (default: 4)
            weights: Weights for each n-gram order (default: uniform)
        """
        self.max_n = max_n
        if weights is None:
            self.weights = [1.0 / max_n] * max_n
        else:
            self.weights = weights
    
    def _get_ngrams(self, text: str, n: int) -> Counter:
        """Extract n-grams from text."""
        tokens = text.lower().split()
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    def _modified_precision(self, reference: str, hypothesis: str, n: int) -> float:
        """Calculate modified n-gram precision."""
        ref_ngrams = self._get_ngrams(reference, n)
        hyp_ngrams = self._get_ngrams(hypothesis, n)
        
        if len(hyp_ngrams) == 0:
            return 0.0
        
        matches = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        return matches / len(hyp_ngrams)
    
    def _brevity_penalty(self, reference: str, hypothesis: str) -> float:
        """Calculate brevity penalty."""
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        if len(hyp_tokens) >= len(ref_tokens):
            return 1.0
        else:
            return math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    
    def __call__(self, references: List[str], hypotheses: List[str]) -> float:
        """
        Calculate BLEU score.
        
        Args:
            references: List of reference translations
            hypotheses: List of hypothesis translations
            
        Returns:
            BLEU score (0-1)
        """
        if len(references) != len(hypotheses):
            raise ValueError("Number of references and hypotheses must match")
        
        if len(references) == 0:
            return 0.0
        
        scores = []
        for ref, hyp in zip(references, hypotheses):
            # Calculate n-gram precisions
            precisions = []
            for n in range(1, self.max_n + 1):
                precision = self._modified_precision(ref, hyp, n)
                precisions.append(precision)
            
            # Calculate geometric mean of precisions
            if all(p > 0 for p in precisions):
                geo_mean = math.exp(sum(w * math.log(p) for w, p in zip(self.weights, precisions)))
            else:
                geo_mean = 0.0
            
            # Apply brevity penalty
            bp = self._brevity_penalty(ref, hyp)
            
            bleu_score = bp * geo_mean
            scores.append(bleu_score)
        
        return np.mean(scores)


class ExactMatchScore:
    """Exact match accuracy calculation."""
    
    def __init__(self, case_sensitive: bool = False, normalize: bool = True):
        """
        Initialize exact match scorer.
        
        Args:
            case_sensitive: Whether comparison is case-sensitive
            normalize: Whether to normalize text before comparison
        """
        self.case_sensitive = case_sensitive
        self.normalize = normalize
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not self.case_sensitive:
            text = text.lower()
        
        if self.normalize:
            # Remove extra whitespace and punctuation
            text = re.sub(r'\s+', ' ', text.strip())
            text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def __call__(self, reference: str, hypothesis: str) -> float:
        """
        Calculate exact match score.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        ref_norm = self._normalize_text(reference)
        hyp_norm = self._normalize_text(hypothesis)
        
        return 1.0 if ref_norm == hyp_norm else 0.0


class SemanticSimilarity:
    """Semantic similarity calculation using cosine similarity."""
    
    def __init__(self, embedding_dim: int = 300):
        """
        Initialize semantic similarity scorer.
        
        Args:
            embedding_dim: Dimension of word embeddings
        """
        self.embedding_dim = embedding_dim
        # Simple word-based embedding (in practice, use pre-trained embeddings)
        self.word_embeddings = {}
    
    def _get_word_embedding(self, word: str) -> np.ndarray:
        """Get word embedding (simplified implementation)."""
        if word not in self.word_embeddings:
            # Generate random embedding for demonstration
            # In practice, use pre-trained embeddings like Word2Vec, GloVe, or BERT
            np.random.seed(hash(word) % 2**32)
            self.word_embeddings[word] = np.random.randn(self.embedding_dim)
        
        return self.word_embeddings[word]
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        words = text.lower().split()
        if not words:
            return np.zeros(self.embedding_dim)
        
        embeddings = [self._get_word_embedding(word) for word in words]
        return np.mean(embeddings, axis=0)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def __call__(self, reference: str, hypothesis: str) -> float:
        """
        Calculate semantic similarity.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            Cosine similarity score (0-1)
        """
        ref_embedding = self._text_to_embedding(reference)
        hyp_embedding = self._text_to_embedding(hypothesis)
        
        return self._cosine_similarity(ref_embedding, hyp_embedding)


class CharacterErrorRate:
    """Character Error Rate (CER) calculation."""
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize CER calculator.
        
        Args:
            case_sensitive: Whether comparison is case-sensitive
        """
        self.case_sensitive = case_sensitive
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not self.case_sensitive:
            text = text.lower()
        return text.strip()
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def __call__(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            CER score (0-1, lower is better)
        """
        ref_norm = self._normalize_text(reference)
        hyp_norm = self._normalize_text(hypothesis)
        
        if len(ref_norm) == 0:
            return 1.0 if len(hyp_norm) > 0 else 0.0
        
        distance = self._levenshtein_distance(ref_norm, hyp_norm)
        return distance / len(ref_norm)


class WordErrorRate:
    """Word Error Rate (WER) calculation."""
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize WER calculator.
        
        Args:
            case_sensitive: Whether comparison is case-sensitive
        """
        self.case_sensitive = case_sensitive
    
    def _normalize_text(self, text: str) -> List[str]:
        """Normalize and tokenize text."""
        if not self.case_sensitive:
            text = text.lower()
        return text.strip().split()
    
    def _levenshtein_distance_words(self, ref_words: List[str], hyp_words: List[str]) -> int:
        """Calculate Levenshtein distance between word sequences."""
        if len(ref_words) < len(hyp_words):
            return self._levenshtein_distance_words(hyp_words, ref_words)
        
        if len(hyp_words) == 0:
            return len(ref_words)
        
        previous_row = list(range(len(hyp_words) + 1))
        for i, ref_word in enumerate(ref_words):
            current_row = [i + 1]
            for j, hyp_word in enumerate(hyp_words):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (ref_word != hyp_word)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def __call__(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            WER score (0-1, lower is better)
        """
        ref_words = self._normalize_text(reference)
        hyp_words = self._normalize_text(hypothesis)
        
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        
        distance = self._levenshtein_distance_words(ref_words, hyp_words)
        return distance / len(ref_words)


class METEORScore:
    """METEOR score calculation."""
    
    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5):
        """
        Initialize METEOR scorer.
        
        Args:
            alpha: Weight for recall vs precision
            beta: Weight for penalty calculation
            gamma: Weight for fragmentation penalty
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def _get_alignments(self, ref_words: List[str], hyp_words: List[str]) -> List[tuple]:
        """Find word alignments between reference and hypothesis."""
        alignments = []
        ref_matched = set()
        hyp_matched = set()
        
        # Exact matches first
        for i, ref_word in enumerate(ref_words):
            if i in ref_matched:
                continue
            for j, hyp_word in enumerate(hyp_words):
                if j in hyp_matched:
                    continue
                if ref_word == hyp_word:
                    alignments.append((i, j))
                    ref_matched.add(i)
                    hyp_matched.add(j)
                    break
        
        return alignments
    
    def _calculate_fragmentation(self, alignments: List[tuple], ref_length: int) -> float:
        """Calculate fragmentation penalty."""
        if not alignments:
            return 1.0
        
        # Sort alignments by reference position
        sorted_alignments = sorted(alignments, key=lambda x: x[0])
        
        # Count chunks (consecutive aligned words)
        chunks = 1
        for i in range(1, len(sorted_alignments)):
            if sorted_alignments[i][0] != sorted_alignments[i-1][0] + 1:
                chunks += 1
        
        return chunks / len(alignments)
    
    def __call__(self, reference: str, hypothesis: str) -> float:
        """
        Calculate METEOR score.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            METEOR score (0-1)
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if len(ref_words) == 0 or len(hyp_words) == 0:
            return 0.0
        
        # Find alignments
        alignments = self._get_alignments(ref_words, hyp_words)
        
        if not alignments:
            return 0.0
        
        # Calculate precision and recall
        precision = len(alignments) / len(hyp_words)
        recall = len(alignments) / len(ref_words)
        
        # Calculate F-mean
        if precision == 0 or recall == 0:
            f_mean = 0.0
        else:
            f_mean = (precision * recall) / (self.alpha * recall + (1 - self.alpha) * precision)
        
        # Calculate fragmentation penalty
        fragmentation = self._calculate_fragmentation(alignments, len(ref_words))
        penalty = self.gamma * fragmentation ** self.beta
        
        # Final METEOR score
        return f_mean * (1 - penalty)


class TranslationMetrics:
    """Comprehensive translation metrics calculator."""
    
    def __init__(self):
        """Initialize all metric calculators."""
        self.bleu = BLEUScore()
        self.exact_match = ExactMatchScore()
        self.semantic_sim = SemanticSimilarity()
        self.cer = CharacterErrorRate()
        self.wer = WordErrorRate()
        self.meteor = METEORScore()
    
    def calculate_all(self, reference: str, hypothesis: str) -> dict:
        """
        Calculate all metrics for a single translation pair.
        
        Args:
            reference: Reference translation
            hypothesis: Hypothesis translation
            
        Returns:
            Dictionary with all metric scores
        """
        return {
            'bleu': self.bleu([reference], [hypothesis]),
            'exact_match': self.exact_match(reference, hypothesis),
            'semantic_similarity': self.semantic_sim(reference, hypothesis),
            'cer': self.cer(reference, hypothesis),
            'wer': self.wer(reference, hypothesis),
            'meteor': self.meteor(reference, hypothesis)
        }
    
    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> dict:
        """
        Calculate all metrics for a batch of translation pairs.
        
        Args:
            references: List of reference translations
            hypotheses: List of hypothesis translations
            
        Returns:
            Dictionary with average metric scores
        """
        if len(references) != len(hypotheses):
            raise ValueError("Number of references and hypotheses must match")
        
        all_scores = []
        for ref, hyp in zip(references, hypotheses):
            scores = self.calculate_all(ref, hyp)
            all_scores.append(scores)
        
        # Calculate averages
        avg_scores = {}
        for metric in ['bleu', 'exact_match', 'semantic_similarity', 'cer', 'wer', 'meteor']:
            avg_scores[f'avg_{metric}'] = np.mean([scores[metric] for scores in all_scores])
        
        return avg_scores


def compute_translation_accuracy(predictions: List[str], references: List[str], threshold: float = 0.8) -> float:
    """
    Compute translation accuracy based on BLEU score threshold.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations  
        threshold: BLEU score threshold for considering translation accurate
        
    Returns:
        Accuracy score (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    if len(predictions) == 0:
        return 0.0
    
    bleu_scorer = BLEUScore()
    accurate_count = 0
    
    for pred, ref in zip(predictions, references):
        bleu_score = bleu_scorer([ref], [pred])
        if bleu_score >= threshold:
            accurate_count += 1
    
    return accurate_count / len(predictions)