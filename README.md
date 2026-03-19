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

## 📂 Repository Structure

All files are organized in the root directory for simplicity. Below is a description of the main components of the replication package.

---

### 📊 Datasets and Preprocessing

- `train_ds_cart.ipynb`  
  Generates saliency maps from the original dataset using the Transformer model. These saliency scores are later used to train CatBoost.

- `extract_saliency.ipynb`  
  Performs advanced processing and aggregation of saliency values for the training dataset.

---

### 🤖 Models

- `catboost_td_distilled.cbm`  
- `catboost_text_tokenaggr.cbm`  
- `catboost_td_cheap.cbm`  
- `catboost_td_full.cbm`  

  Pretrained CatBoost models ready for inference.

- `CB-Text.ipynb*`
- `CB__Text_tokenAggr.ipynb*`
- `CB_Cheap.ipynb*`
- `CB_Full.ipynb*`    
  Model-specific artifacts and intermediate outputs generated during training.

---

### 📈 Training Results

- `plot_catboost*`
- `plot_catboos_text_tokenaggr*`
- `plot_catboos_chep*`
- `plot_catboos_full*`   
  Contain training metrics and visualizations for the four CatBoost variants:
  - distilled  
  - text-token aggregation  
  - cheap  
  - full  

---

### ⚡ Carbon Footprint Analysis

- `carbon_catboost/`  
  Energy consumption and CO₂ emissions data collected during CatBoost inference.

- `carbon_transformer/`  
  Energy consumption and CO₂ emissions data collected during Transformer inference.

- `carbon_plots/`  
  Comparative analysis and visualizations of carbon footprint between the two models.

- `generate_carbon_plots.py`  
  Script used to generate carbon footprint comparison plots.

---

### 📉 Evaluation and Statistics

- `Model-performance.ipynb`  
  Compares CatBoost and Transformer models in terms of classification performance.

- `binary_classification_train_TD.ipynb`  
  Computes inference comparisons between CatBoost and transformer.

---

## 🛠️ Requirements and Technologies

The project is developed in **Python 3.10** and relies on the following main libraries:

- **CatBoost** – Training of the student model based on gradient boosting  
- **HuggingFace Transformers** – Implementation of the teacher model (DistilBERT)  
- **spaCy** – Text preprocessing and linguistic normalization  
- **CodeCarbon** – Monitoring energy consumption and CO₂ emissions  
- **SHAP** – Feature interpretability analysis  

---

### 🧠 Summary

This repository includes:
- data preprocessing and saliency extraction pipelines  
- trained CatBoost models  
- training and evaluation results  
- carbon footprint analysis and visualizations  

All components required to reproduce the experiments are provided.
