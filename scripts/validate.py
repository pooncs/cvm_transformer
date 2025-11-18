import argparse
import yaml
import subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/extended/best_model.pt')
    p.add_argument('--tokenizer', default='data/processed_large_simple/sentencepiece_large.model')
    p.add_argument('--data', default='data/raw/korean_english_large_test.json')
    p.add_argument('--beam', type=int, default=10)
    p.add_argument('--lenp', type=float, default=0.8)
    args = p.parse_args()
    subprocess.run([
        'python','src/evaluation/final_validation.py',
        '--model-path', args.model,
        '--tokenizer-model', args.tokenizer,
        '--test-data', args.data,
        '--search-strategy','beam',
        '--beam-size', str(args.beam),
        '--length-penalty', str(args.lenp),
        '--max-samples','1000'
    ], check=True)

if __name__ == '__main__':
    main()
