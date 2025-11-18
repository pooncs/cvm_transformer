import subprocess

def run(seed):
    subprocess.run([
        'python','src/evaluation/final_validation.py',
        '--model-path','models/extended/best_model.pt',
        '--tokenizer-model','data/processed_large_simple/sentencepiece_large.model',
        '--test-data','validation_sets/holdout_seed42.json',
        '--search-strategy','beam','--beam-size','10','--length-penalty','0.8',
        '--max-samples','1000'
    ], check=True)

if __name__=='__main__':
    for s in [42,1337,2025]:
        run(s)
