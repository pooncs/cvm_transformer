import os
import urllib.request
import tarfile
import gzip
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import librosa
import soundfile as sf
from PIL import Image
import pytesseract

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultimodalCorpusConfig:
    """Configuration for multimodal corpus preparation."""

    vocab_size: int = 16000  # Target vocabulary size (8k-16k range)
    min_sentence_length: int = 3
    max_sentence_length: int = 512
    audio_sample_rate: int = 16000
    audio_min_duration: float = 1.0  # Minimum audio duration in seconds
    audio_max_duration: float = 30.0  # Maximum audio duration in seconds
    image_min_width: int = 100
    image_min_height: int = 100
    enable_multilingual: bool = True
    enable_code_switching: bool = True  # Allow mixed language sentences
    quality_threshold: float = 0.7  # Quality score threshold for filtering


class MultimodalCorpusPreparer:
    """
    Enhanced corpus preparer supporting multimodal data (text, audio, images)
    and larger vocabulary sizes for improved SLM quality.
    """

    def __init__(self, config: MultimodalCorpusConfig = None):
        self.config = config or MultimodalCorpusConfig()
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        # Multilingual corpus sources
        self.text_sources = {
            "opus": {
                "ko-en": [
                    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/ko-en.txt.zip",
                    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-ko.txt.zip",
                ],
                "ja-en": [
                    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/ja-en.txt.zip",
                    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-ja.txt.zip",
                ],
                "zh-en": [
                    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/zh-en.txt.zip",
                    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-zh.txt.zip",
                ],
            },
            "aihub": {
                "ko-en": "data/aihub",  # Manual download required
                "ja-en": "data/aihub_ja",  # Manual download required
                "zh-en": "data/aihub_zh",  # Manual download required
            },
            "common_crawl": {
                "multilingual": [
                    "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-23/segments/1685223402037.99/",
                    "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-50/segments/1701810402037.99/",
                ]
            },
        }

        # Audio corpus sources
        self.audio_sources = {
            "common_voice": {
                "ko": "https://voice.mozilla.org/en/datasets",
                "en": "https://voice.mozilla.org/en/datasets",
                "ja": "https://voice.mozilla.org/en/datasets",
                "zh": "https://voice.mozilla.org/en/datasets",
            },
            "voxlingua107": {
                "url": "https://bark.phon.ioc.ee/voxlingua107/",
                "languages": [
                    "ko",
                    "en",
                    "ja",
                    "zh",
                    "es",
                    "fr",
                    "de",
                    "ru",
                    "ar",
                    "hi",
                ],
            },
            "librispeech": {"en": "http://www.openslr.org/12/"},
        }

        # Image corpus sources (OCR training)
        self.image_sources = {
            "mjsynth": {
                "url": "http://www.robots.ox.ac.uk/~vgg/data/text/",
                "description": "Synthetic text images for OCR training",
            },
            "coco_text": {
                "url": "https://bgshih.github.io/cocotext/",
                "description": "Natural scene text images",
            },
            "icdar2019": {
                "url": "https://rrc.cvc.uab.es/?ch=15",
                "description": "Multilingual scene text detection",
            },
        }

    def download_multilingual_corpora(self) -> Dict[str, List[str]]:
        """Download multilingual text corpora."""
        downloaded_files = {}

        # Download OPUS corpora
        for lang_pair, urls in self.text_sources["opus"].items():
            downloaded_files[f"opus_{lang_pair}"] = []

            for url in urls:
                try:
                    filename = url.split("/")[-1]
                    filepath = self.data_dir / filename

                    if not filepath.exists():
                        logger.info(f"Downloading {url}...")
                        urllib.request.urlretrieve(url, filepath)

                    # Extract if zip
                    if filename.endswith(".zip"):
                        import zipfile

                        extract_dir = self.data_dir / f"opus_{lang_pair}"
                        extract_dir.mkdir(exist_ok=True)

                        with zipfile.ZipFile(filepath, "r") as zip_ref:
                            zip_ref.extractall(extract_dir)

                        # Find extracted files
                        for split in ["train", "dev", "test"]:
                            for lang_file in extract_dir.glob(f"*.{split}.*"):
                                downloaded_files[f"opus_{lang_pair}"].append(
                                    str(lang_file)
                                )

                        logger.info(f"Extracted {filename} to {extract_dir}")

                except Exception as e:
                    logger.warning(f"Failed to download {url}: {e}")

        return downloaded_files

    def download_audio_corpora(self) -> Dict[str, List[str]]:
        """Download audio corpora for speech recognition and language detection."""
        audio_files = {}

        # Common Voice
        cv_dir = self.data_dir / "common_voice"
        cv_dir.mkdir(exist_ok=True)

        logger.info("Common Voice download requires manual download from:")
        logger.info("https://commonvoice.mozilla.org/en/datasets")
        logger.info(f"Place downloaded files in: {cv_dir}")

        # VoxLingua107
        vox_dir = self.data_dir / "voxlingua107"
        vox_dir.mkdir(exist_ok=True)

        logger.info("VoxLingua107 download requires manual download from:")
        logger.info("https://bark.phon.ioc.ee/voxlingua107/")
        logger.info(f"Place downloaded files in: {vox_dir}")

        return audio_files

    def prepare_multilingual_text_corpus(
        self, corpus_files: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Prepare clean multilingual text corpus with quality filtering.

        Args:
            corpus_files: Dictionary of corpus file paths by source

        Returns:
            Dictionary of processed text lines by language
        """
        language_corpora = {}

        for source_name, file_paths in corpus_files.items():
            for file_path in file_paths:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    # Extract language from filename
                    if ".ko" in file_path:
                        lang = "ko"
                    elif ".en" in file_path:
                        lang = "en"
                    elif ".ja" in file_path:
                        lang = "ja"
                    elif ".zh" in file_path:
                        lang = "zh"
                    else:
                        continue

                    # Process lines
                    processed_lines = []
                    for line in lines:
                        line = line.strip()
                        if self._is_valid_text_line(line):
                            processed_lines.append(line)

                    if lang not in language_corpora:
                        language_corpora[lang] = []
                    language_corpora[lang].extend(processed_lines)

                    logger.info(
                        f"Processed {len(processed_lines)} lines from {file_path}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")

        # Remove duplicates and balance corpus
        for lang in language_corpora:
            # Remove duplicates
            language_corpora[lang] = list(set(language_corpora[lang]))

            # Sort by length for better training
            language_corpora[lang].sort(key=len)

            # Apply length filtering
            filtered_lines = []
            for line in language_corpora[lang]:
                if (
                    len(line.split()) >= self.config.min_sentence_length
                    and len(line.split()) <= self.config.max_sentence_length
                ):
                    filtered_lines.append(line)

            language_corpora[lang] = filtered_lines
            logger.info(f"Final corpus for {lang}: {len(filtered_lines)} lines")

        return language_corpora

    def _is_valid_text_line(self, line: str) -> bool:
        """
        Validate text line for quality and appropriateness.

        Args:
            line: Text line to validate

        Returns:
            True if line is valid
        """
        if not line or len(line.strip()) < 10:
            return False

        # Check for excessive punctuation
        if line.count("!") > 3 or line.count("?") > 3:
            return False

        # Check for excessive capitalization
        if sum(1 for c in line if c.isupper()) / len(line) > 0.5:
            return False

        # Check for URLs, emails, phone numbers
        if any(
            pattern in line for pattern in ["http://", "https://", "@", "tel:", "fax:"]
        ):
            return False

        # Check character encoding issues
        try:
            line.encode("utf-8")
        except UnicodeEncodeError:
            return False

        return True

    def prepare_audio_corpus(
        self, audio_files: Dict[str, List[str]]
    ) -> Dict[str, List[Dict]]:
        """
        Prepare audio corpus with segmentation and quality filtering.

        Args:
            audio_files: Dictionary of audio file paths

        Returns:
            Dictionary of processed audio segments by language
        """
        audio_corpora = {}

        for source_name, file_paths in audio_files.items():
            for file_path in file_paths:
                try:
                    # Load audio
                    audio, sr = librosa.load(
                        file_path, sr=self.config.audio_sample_rate
                    )

                    # Extract language from path or metadata
                    lang = self._extract_language_from_path(file_path)
                    if lang is None:
                        continue

                    # Segment audio
                    segments = self._segment_audio(audio, sr)

                    if lang not in audio_corpora:
                        audio_corpora[lang] = []

                    for segment in segments:
                        audio_corpora[lang].append(
                            {
                                "audio": segment["audio"],
                                "duration": segment["duration"],
                                "source_file": file_path,
                                "start_time": segment["start_time"],
                                "end_time": segment["end_time"],
                            }
                        )

                    logger.info(f"Processed {len(segments)} segments from {file_path}")

                except Exception as e:
                    logger.warning(f"Failed to process audio {file_path}: {e}")

        return audio_corpora

    def _segment_audio(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """
        Segment audio into appropriate chunks for training.

        Args:
            audio: Audio data
            sample_rate: Sample rate

        Returns:
            List of audio segments
        """
        segments = []

        # Use voice activity detection for segmentation
        try:
            import webrtcvad

            vad = webrtcvad.Vad(2)  # Aggressiveness level 2

            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)

            # Frame size for VAD (30ms)
            frame_duration = 0.03  # 30ms
            frame_size = int(sample_rate * frame_duration)

            # Segment audio
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i : i + frame_size]

                if len(frame) == frame_size:
                    is_speech = vad.is_speech(frame.tobytes(), sample_rate)

                    if is_speech:
                        # Extract segment around speech
                        start_idx = max(0, i - int(0.5 * sample_rate))  # 0.5s padding
                        end_idx = min(
                            len(audio), i + frame_size + int(2.0 * sample_rate)
                        )  # 2s padding

                        segment_audio = audio[start_idx:end_idx]
                        duration = len(segment_audio) / sample_rate

                        if (
                            self.config.audio_min_duration
                            <= duration
                            <= self.config.audio_max_duration
                        ):
                            segments.append(
                                {
                                    "audio": segment_audio,
                                    "duration": duration,
                                    "start_time": start_idx / sample_rate,
                                    "end_time": end_idx / sample_rate,
                                }
                            )

        except ImportError:
            # Fallback to simple time-based segmentation
            segment_duration = 5.0  # 5-second segments
            segment_samples = int(segment_duration * sample_rate)

            for i in range(0, len(audio) - segment_samples, segment_samples):
                segment_audio = audio[i : i + segment_samples]
                duration = len(segment_audio) / sample_rate

                segments.append(
                    {
                        "audio": segment_audio,
                        "duration": duration,
                        "start_time": i / sample_rate,
                        "end_time": (i + segment_samples) / sample_rate,
                    }
                )

        return segments

    def _extract_language_from_path(self, file_path: str) -> Optional[str]:
        """Extract language code from file path."""
        path_lower = file_path.lower()

        language_patterns = {
            "ko": ["korean", "ko_", "_ko", "/ko/", "common_voice/ko"],
            "en": ["english", "en_", "_en", "/en/", "common_voice/en"],
            "ja": ["japanese", "ja_", "_ja", "/ja/", "common_voice/ja"],
            "zh": ["chinese", "zh_", "_zh", "/zh/", "common_voice/zh"],
            "es": ["spanish", "es_", "_es", "/es/"],
            "fr": ["french", "fr_", "_fr", "/fr/"],
            "de": ["german", "de_", "_de", "/de/"],
            "ru": ["russian", "ru_", "_ru", "/ru/"],
            "ar": ["arabic", "ar_", "_ar", "/ar/"],
            "hi": ["hindi", "hi_", "_hi", "/hi/"],
        }

        for lang, patterns in language_patterns.items():
            if any(pattern in path_lower for pattern in patterns):
                return lang

        return None

    def prepare_image_corpus(
        self, image_files: Dict[str, List[str]]
    ) -> Dict[str, List[Dict]]:
        """
        Prepare image corpus with OCR text extraction and quality filtering.

        Args:
            image_files: Dictionary of image file paths

        Returns:
            Dictionary of processed images with extracted text by language
        """
        image_corpora = {}

        for source_name, file_paths in image_files.items():
            for file_path in file_paths:
                try:
                    # Load and preprocess image
                    image = Image.open(file_path)

                    # Resize if needed
                    if (
                        image.width < self.config.image_min_width
                        or image.height < self.config.image_min_height
                    ):
                        continue

                    # Extract text using OCR
                    extracted_text = pytesseract.image_to_string(image)

                    if not extracted_text.strip():
                        continue

                    # Detect language of extracted text
                    try:
                        from .language_detector import LanguageDetector

                        detector = LanguageDetector()
                        detection_result = detector.detect_text_language(extracted_text)

                        if detection_result is None:
                            continue

                        lang = detection_result.language

                        if lang not in image_corpora:
                            image_corpora[lang] = []

                        image_corpora[lang].append(
                            {
                                "image_path": file_path,
                                "extracted_text": extracted_text.strip(),
                                "ocr_confidence": detection_result.confidence,
                                "image_size": (image.width, image.height),
                                "detection_metadata": detection_result.metadata,
                            }
                        )

                        logger.info(
                            f"Processed image {file_path}: {len(extracted_text)} chars, lang: {lang}"
                        )

                    except ImportError:
                        logger.warning(
                            "LanguageDetector not available for image corpus preparation"
                        )
                        continue

                except Exception as e:
                    logger.warning(f"Failed to process image {file_path}: {e}")

        return image_corpora

    def create_unified_corpus(
        self,
        text_corpora: Dict[str, List[str]],
        audio_corpora: Dict[str, List[Dict]],
        image_corpora: Dict[str, List[Dict]],
    ) -> Dict[str, Dict]:
        """
        Create unified multimodal corpus with aligned data.

        Args:
            text_corpora: Text corpus by language
            audio_corpora: Audio corpus by language
            image_corpora: Image corpus by language

        Returns:
            Unified corpus with multimodal alignment
        """
        unified_corpus = {}

        all_languages = (
            set(text_corpora.keys())
            | set(audio_corpora.keys())
            | set(image_corpora.keys())
        )

        for lang in all_languages:
            unified_corpus[lang] = {
                "text": text_corpora.get(lang, []),
                "audio": audio_corpora.get(lang, []),
                "image": image_corpora.get(lang, []),
                "statistics": {
                    "text_samples": len(text_corpora.get(lang, [])),
                    "audio_samples": len(audio_corpora.get(lang, [])),
                    "image_samples": len(image_corpora.get(lang, [])),
                    "total_samples": (
                        len(text_corpora.get(lang, []))
                        + len(audio_corpora.get(lang, []))
                        + len(image_corpora.get(lang, []))
                    ),
                },
            }

            logger.info(
                f"Unified corpus for {lang}: {unified_corpus[lang]['statistics']}"
            )

        return unified_corpus

    def train_large_vocabulary_tokenizer(
        self, unified_corpus: Dict[str, Dict], vocab_size: int = None
    ) -> str:
        """
        Train a large vocabulary tokenizer using multimodal corpus.

        Args:
            unified_corpus: Unified multimodal corpus
            vocab_size: Target vocabulary size (uses config if None)

        Returns:
            Path to trained tokenizer model
        """
        if vocab_size is None:
            vocab_size = self.config.vocab_size

        # Collect all text data
        all_text_lines = []

        for lang, corpus_data in unified_corpus.items():
            # Add text corpus
            all_text_lines.extend(corpus_data["text"])

            # Add OCR text from images
            for image_data in corpus_data["image"]:
                all_text_lines.append(image_data["extracted_text"])

            # Add transcribed text from audio (if available)
            # This would require ASR - for now we'll skip

        # Remove duplicates and shuffle
        all_text_lines = list(set(all_text_lines))
        np.random.shuffle(all_text_lines)

        logger.info(
            f"Training tokenizer with {len(all_text_lines)} unique lines, vocab_size={vocab_size}"
        )

        # Write combined corpus
        combined_corpus_path = self.data_dir / "multilingual_corpus.txt"
        with open(combined_corpus_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text_lines))

        # Train SentencePiece tokenizer
        try:
            from .sp_tokenizer import train_spm

            model_prefix = f"multilingual_slm_v{vocab_size}"
            model_path = train_spm(
                [str(combined_corpus_path)],
                prefix=model_prefix,
                vocab_size=vocab_size,
                character_coverage=0.995,  # High coverage for multilingual
                model_type="unigram",  # Better for multilingual
            )

            logger.info(f"Large vocabulary tokenizer trained: {model_path}")
            return model_path

        except ImportError:
            logger.error("SentencePiece tokenizer not available")
            return None

    def save_corpus_metadata(
        self, unified_corpus: Dict[str, Dict], tokenizer_path: str
    ) -> str:
        """
        Save corpus metadata and statistics.

        Args:
            unified_corpus: Unified multimodal corpus
            tokenizer_path: Path to trained tokenizer

        Returns:
            Path to metadata file
        """
        metadata = {
            "config": self.config.__dict__,
            "corpus_statistics": {},
            "tokenizer_info": {
                "model_path": tokenizer_path,
                "vocab_size": self.config.vocab_size,
            },
            "languages": list(unified_corpus.keys()),
            "creation_timestamp": str(np.datetime64("now")),
        }

        # Add detailed statistics
        for lang, corpus_data in unified_corpus.items():
            metadata["corpus_statistics"][lang] = corpus_data["statistics"]

        # Save metadata
        metadata_path = self.data_dir / "corpus_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Corpus metadata saved: {metadata_path}")
        return str(metadata_path)

    def run_full_preparation(self) -> Dict[str, str]:
        """
        Run complete multimodal corpus preparation pipeline.

        Returns:
            Dictionary with paths to generated files
        """
        logger.info("Starting multimodal corpus preparation...")

        # Step 1: Download corpora
        logger.info("Step 1: Downloading multilingual text corpora...")
        text_files = self.download_multilingual_corpora()

        logger.info("Step 2: Downloading audio corpora...")
        audio_files = self.download_audio_corpora()

        logger.info("Step 3: Preparing text corpus...")
        text_corpora = self.prepare_multilingual_text_corpus(text_files)

        logger.info("Step 4: Preparing audio corpus...")
        audio_corpora = self.prepare_audio_corpus(audio_files)

        logger.info("Step 5: Preparing image corpus...")
        # Note: Image corpus requires manual download first
        image_corpora = {}  # Empty for now, would be populated from downloaded images

        logger.info("Step 6: Creating unified corpus...")
        unified_corpus = self.create_unified_corpus(
            text_corpora, audio_corpora, image_corpora
        )

        logger.info("Step 7: Training large vocabulary tokenizer...")
        tokenizer_path = self.train_large_vocabulary_tokenizer(unified_corpus)

        logger.info("Step 8: Saving corpus metadata...")
        metadata_path = self.save_corpus_metadata(unified_corpus, tokenizer_path)

        results = {
            "tokenizer_model": tokenizer_path,
            "corpus_metadata": metadata_path,
            "data_directory": str(self.data_dir),
            "languages": list(unified_corpus.keys()),
            "total_samples": sum(
                lang_data["statistics"]["total_samples"]
                for lang_data in unified_corpus.values()
            ),
        }

        logger.info("Multimodal corpus preparation completed!")
        logger.info(f"Results: {results}")

        return results


def prepare_large_corpus(vocab_size: int = 16000) -> str:
    """
    Convenience function to prepare large vocabulary corpus.

    Args:
        vocab_size: Target vocabulary size

    Returns:
        Path to trained tokenizer model
    """
    config = MultimodalCorpusConfig(vocab_size=vocab_size)
    preparer = MultimodalCorpusPreparer(config)

    results = preparer.run_full_preparation()
    return results["tokenizer_model"]


if __name__ == "__main__":
    # Example usage
    print("Preparing multimodal corpus with 16k vocabulary...")

    # Create preparer with enhanced configuration
    config = MultimodalCorpusConfig(
        vocab_size=16000,
        min_sentence_length=5,
        max_sentence_length=256,
        enable_multilingual=True,
        enable_code_switching=True,
        quality_threshold=0.8,
    )

    preparer = MultimodalCorpusPreparer(config)
    results = preparer.run_full_preparation()

    print(f"Corpus preparation completed!")
    print(f"Tokenizer model: {results['tokenizer_model']}")
    print(f"Metadata: {results['corpus_metadata']}")
    print(f"Languages: {results['languages']}")
    print(f"Total samples: {results['total_samples']}")
