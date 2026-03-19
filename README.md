# Efficient Tagging of Technical Debt in Issue Trackers

This repository contains the **replication package** for a Master's thesis focused on the efficient identification of **Technical Debt (TD)** in issue trackers using **Knowledge Distillation** techniques.

---

## 📌 Project Objective

The goal of this work is to investigate whether a lightweight model based on **CatBoost** can match or outperform Transformer-based models (specifically DistilBERT) in the task of Technical Debt classification.

Particular attention is given to:
- computational efficiency
- inference latency
- environmental sustainability

---

## 📊 Datasets

This project relies on both **original datasets** and **processed datasets** published on Hugging Face.

### 🔹 Original Dataset (Training Source)

- [`karths/binary-10IQR-TD`](https://huggingface.co/datasets/karths/binary-10IQR-TD)  
  Original dataset used as the primary source for training and experimentation.

---

### 🔹 Processed and Derived Datasets

The following datasets have been **cleaned, enriched, or restructured** for the purposes of this thesis:

- [`OscarSaci/train_catboost`](http://huggingface.co/datasets/OscarSaci/train_catboost/blob/main/risultati_merged_mixed_enriched.csv)  
  Processed dataset used for training the CatBoost model after preprocessing and feature engineering.

---

### 🔹 Evaluation and Test Datasets

Additional datasets used for **evaluation and generalization testing**:

- [`ds_jira`](https://huggingface.co/datasets/OscarSaci/train_catboost/blob/main/jira_TD_dataset.csv)  
  Dataset derived from Jira issue trackers used to test model robustness on real-world data.

- [`2024_10IQR_technical_debt`](https://huggingface.co/datasets/OscarSaci/train_catboost/blob/main/2024_10IQR_technical%20debt.csv)  
  Additional dataset used for validation and benchmarking.

> ⚠️ Note: Some datasets are included in this repository or available via Hugging Face, depending on size and preprocessing requirements.

---

## 🛠️ Requirements and Technologies

The project is developed in **Python 3.10** and relies on the following main libraries:

- **CatBoost** – Training of the student model based on gradient boosting  
- **HuggingFace Transformers** – Implementation of the teacher model (DistilBERT)  
- **spaCy** – Text preprocessing and linguistic normalization  
- **CodeCarbon** – Monitoring energy consumption and CO₂ emissions  
- **SHAP** – Feature interpretability analysis  
