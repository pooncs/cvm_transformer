# Repository Structure for CVM Transformer Translation System

## Project Organization Plan

### Core Directories
```
cvm_transformer/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   ├── training/                 # Training scripts and configs
│   ├── data/                     # Data processing utilities
│   ├── evaluation/               # Testing and evaluation
│   └── deployment/               # Production deployment
├── data/                         # Dataset files
│   ├── raw/                      # Original datasets
│   ├── processed/                # Processed datasets
│   └── multilingual/             # Multilingual extensions
├── models/                       # Saved model checkpoints
│   ├── checkpoints/              # Training checkpoints
│   ├── production/               # Production-ready models
│   └── experiments/              # Experimental models
├── tests/                        # Test suites
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── comprehensive/            # Comprehensive evaluation
├── configs/                      # Configuration files
│   ├── training/                 # Training configurations
│   ├── model/                    # Model architectures
│   └── deployment/               # Deployment settings
├── docs/                         # Documentation
│   ├── api/                      # API documentation
│   ├── tutorials/                # Usage tutorials
│   └── reports/                  # Analysis reports
└── scripts/                      # Utility scripts
    ├── setup/                    # Setup scripts
    ├── maintenance/              # Maintenance scripts
    └── analysis/                 # Analysis tools
```

### Current File Organization Plan

#### Move to src/models/
- cvm_translator/cvm_transformer.py → src/models/cvm_transformer.py
- cvm_translator/sp_tokenizer.py → src/models/tokenizers.py
- cvm_translator/slm_model.py → src/models/slm_model.py

#### Move to src/training/
- scripts/train.py → src/training/train_basic.py
- scripts/train_enhanced.py → src/training/train_enhanced.py
- scripts/train_advanced_simple.py → src/training/train_advanced.py
- scripts/create_production_model.py → src/training/export_model.py

#### Move to src/evaluation/
- scripts/comprehensive_test.py → src/evaluation/comprehensive_test.py
- scripts/test_production_model.py → src/evaluation/test_production.py
- scripts/test_final_unified.py → src/evaluation/test_unified.py

#### Move to models/
- simple_best_model.pth → models/checkpoints/simple_best.pth
- production_model.pth → models/production/korean_english_translator.pth
- production_config.json → models/production/config.json

#### Move to data/
- data/kr.txt → data/raw/korean_sentences.txt
- data/en.txt → data/raw/english_sentences.txt
- data/kr_diverse.txt → data/processed/korean_diverse.txt
- data/en_diverse.txt → data/processed/english_diverse.txt
- kr_en_diverse.model → data/tokenizers/korean_english.model
- kr_en_diverse.vocab → data/tokenizers/korean_english.vocab

#### Move to tests/comprehensive/
- Create new comprehensive test suite with multimedia support
- Include Korean word lists, images, and audio clips
- Design automated evaluation pipeline

### Implementation Steps
1. Create directory structure
2. Move and organize existing files
3. Update import paths
4. Create comprehensive test suite
5. Optimize hyperparameters
6. Run full training
7. Comprehensive evaluation