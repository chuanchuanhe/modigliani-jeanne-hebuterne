# Comparing Machine Learning Classifiers for Breast Cancer Diagnosis

This project compares three machine learning classifiers for breast cancer diagnosis using the Breast Cancer Wisconsin Diagnostic dataset. The goal is to evaluate how traditional machine learning models and a simple neural network perform on a binary medical classification task.

The classification task is to predict whether a tumor is:

- **Malignant**
- **Benign**

Because this is a medical diagnosis problem, performance is evaluated using not only accuracy, but also **precision**, **recall**, and **F1-score**. Recall is especially important in this context because false negatives may mean missing malignant cases.

## Repository Contents

| File | Description |
|---|---|
| `ML_BC.py` | Main Python script for preprocessing, training, cross-validation, evaluation, and plotting |
| `Report_ML_BC.pdf` | Full written project report |
| `Poster_ML_BC.pdf` | Project poster |
| `README.md` | Project overview and instructions |
| `data.csv` | Project dataset: Breast Cancer Wisconsin Diagnostic dataset |

## Project Overview

This project compares three classification models:

1. **K-Nearest Neighbors**
2. **Logistic Regression**
3. **Neural Network**

The models are evaluated using stratified 5-fold cross-validation. Stratification helps preserve the malignant/benign class distribution in each fold.

## Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset.

The dataset contains numerical features computed from digitized images of fine needle aspirate samples of breast masses. Each sample is labeled as either malignant or benign.

The target column is:

```text
diagnosis
