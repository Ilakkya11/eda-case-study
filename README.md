# Exploratory Data Analysis and Visualization: Breast Cancer Diagnosis

## Overview
This undergraduate case study applies exploratory data analysis, statistical visualization, preprocessing, and a feed-forward deep learning model to the Wisconsin Breast Cancer Diagnostic dataset. It classifies samples as malignant or benign from 30 numeric measurements computed from digitized fine-needle aspirate images.

> Educational demonstration only. This model is not a medical device and must not be used for diagnosis or clinical decisions.

## Dataset
- Source: UCI Machine Learning Repository, Wisconsin Diagnostic Breast Cancer (WDBC)
- Dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- DOI: https://doi.org/10.24432/C5DW2B
- Size: 569 rows, 30 numeric predictor features, plus binary target.
- Target mapping: 0 = malignant; 1 = benign.

## Results
- Test accuracy: 95.35%
- Test precision (benign): 94.64%
- Test recall (benign): 98.15%
- Test F1-score (benign): 96.36%
- Test ROC-AUC (benign): 0.9902
- Split: 398 train / 85 validation / 86 test samples.

## Seven stages of visualization
This project uses Ben Fry's framework: Acquire, Parse, Filter, Mine, Represent, Refine, Interact. The report explains how each stage is applied. The static report is not interactive; the notebook can be rerun and modified.

## Project structure
- `src/main.py` — reproducible EDA, preprocessing, MLP model training, metrics, and visualization generation.
- `notebooks/eda_case_study.ipynb` — notebook entry point.
- `data/README.md` — dataset source and access notes.
- `requirements.txt` — Python dependencies.
- `reports/` — case study report (PDF and DOCX).
- `figures/` — generated EDA and model evaluation charts.
- `results/` — metric tables and run summaries.
- `models/` — trained model artifact.

## Run locally
Python 3.10+ recommended:
```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python src/main.py
```

The dataset is bundled with scikit-learn, so no separate CSV download is required. The script generates figures and results. See the report for methodology, limitations, and references.