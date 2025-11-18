import argparse
import yaml
import subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base.yaml')
    args = p.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cmd = [
        'python','src/training/train_extended_nmt.py',
        '--data-file', cfg['train_file'],
        '--val-file', cfg['val_file'],
        '--tokenizer-model', cfg['tokenizer_model'],
        '--epochs', str(cfg['epochs']),
        '--batch-size', str(cfg['batch_size']),
        '--learning-rate', str(cfg['learning_rate']),
        '--d-model', str(cfg['d_model']),
        '--nhead', str(cfg['nhead']),
        '--num-layers', str(cfg['num_layers']),
        '--dim-feedforward', str(cfg['dim_feedforward']),
        '--dropout', str(cfg['dropout']),
        '--max-length', str(cfg['max_length']),
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
