import shutil
from pathlib import Path

def main():
    for d in ['__pycache__','wandb']: 
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)
    print('cleanup done')

if __name__=='__main__':
    main()
