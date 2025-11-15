import os
import urllib.request
import tarfile
import gzip
from pathlib import Path

def download_opus_ko_en():
    """Download OPUS Korean-English parallel corpus"""
    urls = [
        "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/ko-en.txt.zip",
        "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-ko.txt.zip"
    ]
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for url in urls:
        filename = url.split('/')[-1]
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, filepath)
            
        # Extract if zip
        if filename.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Extracted {filename}")

def download_ai_hub_ko_en():
    """Download AI Hub Korean-English parallel corpus (if available)"""
    # AI Hub requires registration, so we'll use a placeholder
    # Users can manually download and place files in data/aihub/
    aihub_dir = Path("data/aihub")
    aihub_dir.mkdir(exist_ok=True)
    
    print("AI Hub corpus download requires manual registration at:")
    print("https://www.aihub.or.kr/aihub-data/natural-language/about")
    print("Place downloaded files in data/aihub/")

def prepare_corpus_files():
    """Prepare clean corpus files for SentencePiece training"""
    data_dir = Path("data")
    kr_lines = []
    en_lines = []
    
    # Process OPUS files
    opus_dir = data_dir / "OpenSubtitles"
    if opus_dir.exists():
        for split in ["train", "dev", "test"]:
            kr_file = opus_dir / f"ko-en.{split}.ko"
            en_file = opus_dir / f"ko-en.{split}.en"
            
            if kr_file.exists() and en_file.exists():
                with open(kr_file, 'r', encoding='utf-8') as f:
                    kr_lines.extend(f.readlines())
                with open(en_file, 'r', encoding='utf-8') as f:
                    en_lines.extend(f.readlines())
    
    # Process AI Hub files if available
    aihub_dir = data_dir / "aihub"
    if aihub_dir.exists():
        for file in aihub_dir.glob("*.ko"):
            with open(file, 'r', encoding='utf-8') as f:
                kr_lines.extend(f.readlines())
        for file in aihub_dir.glob("*.en"):
            with open(file, 'r', encoding='utf-8') as f:
                en_lines.extend(f.readlines())
    
    # Clean and deduplicate
    kr_lines = [line.strip() for line in kr_lines if line.strip()]
    en_lines = [line.strip() for line in en_lines if line.strip()]
    
    # Ensure parallel corpus alignment
    min_len = min(len(kr_lines), len(en_lines))
    kr_lines = kr_lines[:min_len]
    en_lines = en_lines[:min_len]
    
    # Write final corpus files
    with open(data_dir / "kr_corpus.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(kr_lines))
    
    with open(data_dir / "en_corpus.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(en_lines))
    
    print(f"Prepared corpus: {len(kr_lines)} Korean lines, {len(en_lines)} English lines")
    return [str(data_dir / "kr_corpus.txt"), str(data_dir / "en_corpus.txt")]

def train_production_tokenizer(vocab_size=32000):
    """Train a production-quality SentencePiece tokenizer"""
    from sp_tokenizer import train_spm
    
    corpus_files = prepare_corpus_files()
    
    print(f"Training SentencePiece tokenizer with vocab_size={vocab_size}...")
    train_spm(corpus_files, prefix="kr_en_prod", vocab_size=vocab_size)
    
    print("Production tokenizer trained: kr_en_prod.model")
    return "kr_en_prod.model"

if __name__ == "__main__":
    # Download corpora
    download_opus_ko_en()
    download_ai_hub_ko_en()
    
    # Train production tokenizer
    train_production_tokenizer(vocab_size=32000)