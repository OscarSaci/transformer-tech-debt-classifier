# Efficient Tagging of Technical Debt in Issue Trackers

Questo repository contiene il **replication package** relativo alla tesi di Laurea Magistrale, focalizzata sull'identificazione efficiente del **Technical Debt (TD)** negli issue tracker attraverso tecniche di **Knowledge Distillation**.

---

## 📌 Obiettivo del Progetto

L'obiettivo del lavoro è valutare se un modello leggero basato su **CatBoost** sia in grado di eguagliare o superare le prestazioni di modelli basati su Transformer (in particolare DistilBERT) nel task di classificazione del Technical Debt.

Particolare attenzione è dedicata a:
- efficienza computazionale
- latenza in inferenza
- sostenibilità ambientale

---

## 🛠️ Requisiti e Tecnologie

Il progetto è sviluppato in **Python 3.10** e utilizza le seguenti librerie principali:

- **CatBoost** – Addestramento del modello student basato su gradient boosting  
- **HuggingFace Transformers** – Implementazione del modello teacher (DistilBERT)  
- **spaCy** – Preprocessing e normalizzazione del testo  
- **CodeCarbon** – Monitoraggio del consumo energetico e delle emissioni di CO₂  
- **SHAP** – Analisi di interpretabilità delle feature  

Per installare le dipendenze:

```bash
pip install -r requirements.txt
