
# NeuroFold AI

**A machine-learning research prototype for analysing short peptide sequences.**

## Overview

NeuroFold AI is a Python-based project that extracts physicochemical features from six-amino-acid peptide sequences and trains machine-learning classifiers on a labelled peptide dataset.

The current prototype includes:
- Dataset preparation from FASTA files
- Peptide feature extraction
- Logistic Regression and Random Forest classification
- Model evaluation using accuracy, balanced accuracy, confusion matrices, and ROC-AUC
- A saved model for experimental predictions on new six-amino-acid sequences

## Research Scope

The current model is limited to six-amino-acid sequences in the supplied dataset. Its biological labels and dataset provenance must be verified against the original source before interpreting predictions as amyloidogenic or non-amyloidogenic.

## Methods

The feature extraction script calculates:
- Sequence length
- Molecular weight
- Isoelectric point
- GRAVY (average hydropathicity)
- Aromaticity
- Amino-acid composition

The current classifiers are Logistic Regression and Random Forest.

## Preliminary Results

A random train/test split produced a preliminary ROC-AUC of 0.859 for Logistic Regression.

This result is not an independent final performance estimate. Similar peptide sequences may appear in both the training and test sets, potentially making performance appear better than it would be on genuinely novel sequences.

## Project Files

- `prepare_dataset.py` — converts FASTA records into a CSV dataset
- `extract_features.py` — calculates peptide features
- `train_model.py` — trains, evaluates, and saves the model
- `predict_peptide.py` — accepts a six-amino-acid sequence for experimental prediction
- `check_sequences.py` — checks sequence validity
- `check_similarity.py` — examines sequence similarity in the dataset
- `check_split_similarity.py` — examines similarity across a random train/test split
- `requirements.txt` — lists required Python libraries

## Installation

Install Python, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Run these commands from the project folder in the following order:

```bash
python prepare_dataset.py
python extract_features.py
python train_model.py
python predict_peptide.py
```

The training script creates `neurofold_model.joblib`, which is used by the prediction script.

## Limitations

- The model is experimental and has not been independently validated.
- Performance may be inflated by similarities between training and test sequences.
- Predictions are not experimentally confirmed results.
- The model is not intended for clinical diagnosis or medical decision-making.
- Dataset provenance, label definitions, and reuse permissions should be verified.

## Future Work

- Verify the dataset source and label definitions
- Implement similarity-aware evaluation
- Compare against simple baseline models
- Test generalisation on an independent dataset
- Improve documentation and reproducibility

## Disclaimer

NeuroFold AI is an educational and research-support prototype. Predictions should not be treated as definitive evidence of peptide aggregation or used for clinical decisions.
