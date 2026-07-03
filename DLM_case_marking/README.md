# Quantifying How Case Marking Reduces DLM Pressure via Predictability

## Overview

This project investigates how **case marking** in languages like Hindi, Japanese, German, Korean, and Turkish compensates for violations of **Dependency Length Minimization (DLM)**. We use LLM-based **surprisal** as a measure of processing cost and show that case markers maintain predictability across long syntactic dependencies.

## Hypothesis

Case-marking languages violate DLM more freely (longer dependencies) but case markers **compensate by maintaining predictability** — the slope of surprisal vs. dependency length is flatter in case-marking languages than in non-case-marking controls.

## Languages

| Case-Marking | Non-Case-Marking |
|-------------|-----------------|
| Hindi | English |
| Japanese | Mandarin Chinese |
| German | Indonesian |
| Korean | Vietnamese |
| Turkish | French |

## Analyses

1. **Core**: Surprisal × Dependency Length regression — compare slopes across language types
2. **Ablation**: Remove case markers → measure information loss (delta surprisal)
3. **Argument vs. All Deps**: Case marking effect should be strongest for argument dependencies
4. **UID**: Uniform Information Density — case marking → smoother surprisal distribution
5. **Critical Region**: Surprisal at dependency "reunion points"

## Setup

```bash
pip install -r requirements.txt
python download_data.py
```

## Usage

```bash
# Step 1: Parse UD data and extract features
python -m src.conllu_parser

# Step 2: Compute surprisal (run on GPU — Kaggle/Colab recommended)
python -m src.surprisal_calculator

# Step 3: Run analyses
python -m analysis.core_analysis
python -m analysis.uid_analysis
python -m analysis.critical_region_analysis

# Step 4: Generate visualizations
python -m analysis.visualizations
```

## Data

Uses [Universal Dependencies](https://universaldependencies.org/) treebanks (CoNLL-U format).
Surprisal computed using [XGLM-564M](https://huggingface.co/facebook/xglm-564M).
