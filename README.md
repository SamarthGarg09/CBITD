# Concept-Based Interpretability for Toxicity Detection

This repository contains implementations for analyzing toxicity detection models using concept-based interpretability techniques. The code is organized into two separate experiments for the Civil Comments and HateXplain datasets.

## Repository Structure

```
.
├── Artifacts/                    # Output directory for both experiments
├── backup-patches/               # Backup files
├── CivilComments-EXP/           # Civil Comments experiment
├── HateXplain-EXP/              # HateXplain experiment
├── .gitignore
└── README.md
```

## Dataset Specific Instructions

### Civil Comments Experiment
Navigate to CivilComments-EXP directory:
```bash
cd CivilComments-EXP
```

1. Train models:
```bash
python target_model.py    # Train toxicity detection model
python concept_model.py   # Train concept prediction model
```

2. Run analysis:
```bash
python cg_apply_automated.py   # Run concept gradient analysis
python tcav_apply.py          # Run TCAV analysis
python get_stats.py           # Generate statistics
```

Concepts analyzed:
- Obscene
- Threat
- Sexual Explicit
- Insult
- Identity Attack

### HateXplain Experiment
Navigate to HateXplain-EXP directory:
```bash
cd HateXplain-EXP
```

1. Train models:
```bash
python target_model.py    # Train hate speech detection model
python concept_model.py   # Train concept prediction model
```

2. Run analysis:
```bash
python cg_apply_automated.py   # Run concept gradient analysis
python tcav_apply.py          # Run TCAV analysis
python get_stats.py           # Generate statistics
```

Concepts analyzed:
- Race
- Religion
- Gender

## Setup Requirements

Install dependencies:
```bash
pip install torch transformers datasets pandas numpy matplotlib seaborn wordcloud tqdm captum
```

## Data Organization

For each experiment, prepare your data in the following structure:
```
dataset/
    ├── train.csv
    ├── dev.csv
    └── test.csv
```

## Model Architecture

Both experiments use:
- Base Architecture: RoBERTa
- Target Model: Binary classification (toxic/non-toxic or hate/normal)
- Concept Model: Multi-label concept classification

## Output Structure

Results are saved in the Artifacts directory:
```
Artifacts/
    ├── CivilComments/
    │   ├── word_clouds/        # Word cloud visualizations
    │   ├── plots/              # Concept attribution plots
    │   └── csv_dumps/          # Analysis results
    │
    └── HateXplain/
        ├── word_clouds/        # Word cloud visualizations
        └─ plots/              # Concept attribution plots
```
