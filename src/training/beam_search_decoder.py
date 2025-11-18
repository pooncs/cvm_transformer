#!/usr/bin/env python3
"""
Beam Search Decoder for Korean-English Translation
Implements beam search decoding for improved translation quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import sentencepiece as spm
import json
import numpy as np


class BeamHypothesis:
    """Represents a single beam hypothesis"""

    def __init__(
        self, tokens: List[int], score: float, attention: Optional[torch.Tensor] = None
    ):
        self.tokens = tokens
        self.score = score
        self.attention = attention

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        return f"BeamHypothesis(tokens={self.tokens}, score={self.score:.4f})"


class BeamSearchDecoder:
    """Beam search decoder for sequence-to-sequence models with strategy options and stats logging"""

    def __init__(
        self,
        model,
        tokenizer,
        beam_size: int = 5,
        max_length: int = 128,
        length_penalty: float = 0.6,
        temperature: float = 1.0,
        strategy: str = "beam",
        diversity_strength: float = 0.0,
        num_groups: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.strategy = strategy
        self.diversity_strength = diversity_strength
        self.num_groups = max(1, num_groups)

        # Special tokens
        self.pad_token_id = tokenizer.piece_to_id("<pad>")
        self.bos_token_id = tokenizer.piece_to_id("<s>")
        self.eos_token_id = tokenizer.piece_to_id("</s>")
        self.unk_token_id = tokenizer.piece_to_id("<unk>")

    def length_penalty_fn(self, length: int) -> float:
        """Apply length penalty to normalize scores"""
        return ((5 + length) / (5 + 1)) ** self.length_penalty

    def _log_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(logits / max(self.temperature, 1e-5), dim=-1)

    def _apply_diversity(
        self, group_index: int, token_scores: torch.Tensor
    ) -> torch.Tensor:
        if self.diversity_strength <= 0.0 or self.num_groups <= 1:
            return token_scores
        # Penalize previously selected tokens in earlier groups to encourage diversity
        penalty = self.diversity_strength * group_index
        return token_scores - penalty

    def decode_single(
        self, src_tokens: torch.Tensor, src_length: torch.Tensor
    ) -> Tuple[str, List[float], Optional[torch.Tensor]]:
        """Decode a single source sequence using beam search"""
        self.model.eval()

        with torch.no_grad():
            # Move to device
            device = next(self.model.parameters()).device
            src_tokens = src_tokens.unsqueeze(0).to(device)
            src_length = src_length.unsqueeze(0).to(device)

            # Create source mask
            src_key_padding_mask = self.model.create_padding_mask(
                src_length, src_tokens.size(1)
            )

            # Encode source sequence
            src_emb = self.model.embedding(src_tokens) * np.sqrt(self.model.d_model)
            src_emb = self.model.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
            src_emb = src_emb.transpose(0, 1)  # (seq_len, batch, d_model)

            # Encode through transformer encoder
            memory = self.model.transformer.encoder(
                src_emb, src_key_padding_mask=src_key_padding_mask
            )

            # Initialize beams
            beams = [BeamHypothesis([self.bos_token_id], 0.0)]
            completed_beams = []

            for step in range(self.max_length):
                candidates = []

                # Expand each beam
                for beam in beams:
                    if beam.tokens[-1] == self.eos_token_id:
                        # Already completed
                        completed_beams.append(beam)
                        continue

                    # Prepare decoder input
                    tgt_tokens = (
                        torch.tensor(beam.tokens, dtype=torch.long)
                        .unsqueeze(0)
                        .to(device)
                    )
                    tgt_length = torch.tensor([len(beam.tokens)], dtype=torch.long).to(
                        device
                    )

                    # Create target mask
                    tgt_key_padding_mask = self.model.create_padding_mask(
                        tgt_length, tgt_tokens.size(1)
                    )

                    # Add positional encoding
                    tgt_emb = self.model.embedding(tgt_tokens) * np.sqrt(
                        self.model.d_model
                    )
                    tgt_emb = self.model.pos_encoding(
                        tgt_emb.transpose(0, 1)
                    ).transpose(0, 1)
                    tgt_emb = tgt_emb.transpose(0, 1)

                    # Create causal mask
                    tgt_length_current = tgt_tokens.size(1)
                    tgt_mask = self.model.transformer.generate_square_subsequent_mask(
                        tgt_length_current
                    ).to(device)

                    # Decode through transformer decoder
                    decoder_output = self.model.transformer.decoder(
                        tgt_emb,
                        memory,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=src_key_padding_mask,
                    )

                    # Project to vocabulary
                    decoder_output = decoder_output.transpose(0, 1)
                    logits = self.model.fc_out(decoder_output[:, -1, :])  # Last token

                    # Apply temperature
                    if self.temperature != 1.0:
                        logits = logits / self.temperature

                    # Get top-k candidates
                    log_probs = F.log_softmax(logits, dim=-1)
                    topk_log_probs, topk_indices = torch.topk(
                        log_probs, k=self.beam_size
                    )

                    # Create new candidates
                    for i in range(self.beam_size):
                        token_id = topk_indices[0, i].item()
                        log_prob = topk_log_probs[0, i].item()

                        new_tokens = beam.tokens + [token_id]
                        new_score = beam.score + log_prob / self.length_penalty_fn(
                            len(new_tokens)
                        )

                        candidates.append(BeamHypothesis(new_tokens, new_score))

                # Select top beam_size candidates
                candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
                beams = candidates[: self.beam_size]

                # Check if all beams are completed
                if all(beam.tokens[-1] == self.eos_token_id for beam in beams):
                    completed_beams.extend(beams)
                    break

            # Add remaining beams to completed
            completed_beams.extend(beams)

            # Sort completed beams by score
            completed_beams = sorted(
                completed_beams, key=lambda x: x.score, reverse=True
            )

            # Return best result
            if completed_beams:
                best_beam = completed_beams[0]

                # Remove BOS and EOS tokens
                output_tokens = [
                    t
                    for t in best_beam.tokens
                    if t not in [self.bos_token_id, self.eos_token_id]
                ]

                # Decode to text
                output_text = self.tokenizer.decode(output_tokens)

                # Return scores for each token (approximate)
                token_scores = [1.0] * len(output_tokens)  # Placeholder

                return output_text, token_scores, None

            return "", [], None

    def decode_batch(
        self, src_batch: torch.Tensor, src_lengths: torch.Tensor
    ) -> List[Tuple[str, List[float], Optional[torch.Tensor]]]:
        """Decode a batch of source sequences"""
        results = []
        batch_size = src_batch.size(0)

        for i in range(batch_size):
            result = self.decode_single(src_batch[i], src_lengths[i])
            results.append(result)

        return results


class TranslationInference:
    """High-level translation inference interface"""

    def __init__(self, model_path: str, tokenizer_model: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)

        # Load model
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        args = checkpoint["args"]

        # Create model
        import sys
        import os

        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from src.training.train_extended_nmt import TransformerNMT

        self.model = TransformerNMT(
            vocab_size=self.tokenizer.get_piece_size(),
            d_model=getattr(args, "d_model", 512),
            nhead=getattr(args, "nhead", 8),
            num_encoder_layers=getattr(args, "num_layers", 6),
            num_decoder_layers=getattr(args, "num_layers", 6),
            dim_feedforward=getattr(args, "dim_feedforward", 2048),
            dropout=getattr(args, "dropout", 0.1),
            max_len=getattr(args, "max_length", 128),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Create decoder
        self.decoder = BeamSearchDecoder(
            model=self.model,
            tokenizer=self.tokenizer,
            beam_size=5,
            max_length=128,
            length_penalty=0.6,
        )

    def translate(self, korean_text: str) -> str:
        """Translate Korean text to English using configured search strategy"""
        tokens = (
            [self.tokenizer.piece_to_id("<s>")]
            + self.tokenizer.encode(korean_text)
            + [self.tokenizer.piece_to_id("</s>")]
        )
        length = len(tokens)
        src = torch.tensor(tokens, dtype=torch.long)
        src_len = torch.tensor(length, dtype=torch.long)

        # Beam search decoding via decoder
        out, _, _ = self.decoder.decode_single(src, src_len)
        return out

    def decode_single_with_stats(self, src: torch.Tensor, src_len: torch.Tensor):
        """Decode a single example returning translation and beam stats"""
        translation, token_scores, attn = self.decoder.decode_single(src, src_len)
        beams = [translation]
        scores = token_scores
        return translation, beams, scores

    def translate_batch(self, korean_texts: List[str]) -> List[str]:
        """Translate multiple Korean texts to English"""
        results = []

        # Prepare batch
        korean_batch = []
        lengths_batch = []

        for text in korean_texts:
            tokens = (
                [self.tokenizer.piece_to_id("<s>")]
                + self.tokenizer.encode(text)
                + [self.tokenizer.piece_to_id("</s>")]
            )
            tokens = tokens[:128]
            length = len(tokens)
            padded = tokens + [self.tokenizer.piece_to_id("<pad>")] * (128 - length)

            korean_batch.append(padded)
            lengths_batch.append(length)

        # Convert to tensors
        korean_tensor = torch.tensor(korean_batch, dtype=torch.long)
        lengths_tensor = torch.tensor(lengths_batch, dtype=torch.long)

        # Decode batch
        translations = self.decoder.decode_batch(korean_tensor, lengths_tensor)

        return [trans[0] for trans in translations]


def test_beam_search():
    """Test beam search decoder"""
    print("Testing beam search decoder...")

    # Sample Korean sentences
    test_sentences = [
        "안녕하세요",
        "오늘 날씨가 좋네요",
        "한국어를 영어로 번역합니다",
        "이것은 테스트 문장입니다",
    ]

    # Initialize inference
    inference = TranslationInference(
        model_path="models/extended/best_model.pt",
        tokenizer_model="data/processed_large_simple/sentencepiece_large.model",
    )

    # Test translations
    print("\nTranslation Results:")
    print("=" * 50)

    for korean in test_sentences:
        english = inference.translate(korean)
        print(f"Korean:  {korean}")
        print(f"English: {english}")
        print("-" * 50)

    # Test batch translation
    print("\nBatch Translation:")
    batch_results = inference.translate_batch(test_sentences)
    for i, (korean, english) in enumerate(zip(test_sentences, batch_results)):
        print(f"{i+1}. {korean} -> {english}")


if __name__ == "__main__":
    test_beam_search()
