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

## 🛠️ Requirements and Technologies

The project is developed in **Python 3.10** and relies on the following main libraries:

- **CatBoost** – Training of the student model based on gradient boosting  
- **HuggingFace Transformers** – Implementation of the teacher model (DistilBERT)  
- **spaCy** – Text preprocessing and linguistic normalization  
- **CodeCarbon** – Monitoring energy consumption and CO₂ emissions  
- **SHAP** – Feature interpretability analysis  
