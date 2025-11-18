import json
import sys
from pathlib import Path
import numpy as np

def main():
    inp = sys.argv[1] if len(sys.argv)>1 else 'data/raw/korean_english_large.json'
    outdir = Path('validation_sets')
    outdir.mkdir(parents=True, exist_ok=True)
    data = json.load(Path(inp).open('r', encoding='utf-8'))
    np.random.seed(42)
    np.random.shuffle(data)
    holdout = data[:1000]
    with (outdir/'holdout_seed42.json').open('w', encoding='utf-8') as f:
        json.dump(holdout, f, ensure_ascii=False, indent=2)
    print('created holdout_seed42.json with', len(holdout))

if __name__ == '__main__':
    main()
