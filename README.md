# CS 171 Final Project

# 🧊 Iceberg vs Ship Detection CNN (CS171 Intro to ML)

**Final Results**: 77.5% test accuracy (+19.5% over 58% baseline) on satellite imagery classification

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanbancan/CS171-Intro-to-ML-Project/blob/main/notebooks/Full_Run.ipynb)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🎯 Project Overview
Classify 75×75px satellite images as **icebergs** or **ships** for maritime safety. Dataset: 1,604 train / 8,424 test images.

**Baseline**: 58% test accuracy → **Final**: 77.5% test accuracy (F1: 0.76)

## 📁 Repository Structure
CS171-Intro-to-ML-Project/
├── README.md                 
├── notebooks/
│   ├── 00_eda_victoria.ipynb
│   ├── 01_data_preprocessing_sania.ipynb  ← **SANIA**
│   └── 03_model_experiments_sania.ipynb   ← **SANIA**
├── src/
│   ├── data_preparation.py              ← **SANIA**
│   ├── train_sania.py                   ← **SANIA**
│   ├── baseline_victoria.py
│   └── model_architectures.py
├── data/                 ← sample_data.csv (10 images)
├── results/              ← accuracy_plots.png, experiment_log.csv
└── requirements.txt



## 🔧 My Contributions (Sania Bandekar)
1. **Data Pipeline** (`02_data_preprocessing_sania.ipynb`): Normalization, augmentation (rotation/flip/brightness), class balancing
2. **Best CNN Architecture** (`sania_cnn_v2.py`): 4-layer (64→128→256→512 filters), 77.5% test accuracy
3. **Evaluation Framework** (`04_model_evaluation_sania.ipynb`): 5 architectures compared, F1-metrics, CV
4. **Hyperparameter Tracking** (`hyperparameter_log_sania.csv`): 23 experiments (batch_size, dropout, epochs)

## 📊 Key Results
| Model | Test Acc | Test F1 | Train Acc | Val Acc |
|-------|----------|---------|-----------|---------|
| Logistic Baseline | 58.2% | 0.57 | - | - |
| Victoria CNN v1 (3-layer) | 68.4% | 0.67 | 89% | 65% |
| **Sania CNN v2 (4-layer)** | **77.5%** | **0.76** | 94% | 73% |

## 🛠️ Technical Decisions
Architecture: Conv2D(64) → MaxPool → Conv2D(128) → Dropout(0.2) →
Conv2D(256) → Conv2D(512) → GlobalAvgPool → Dense(1, sigmoid)

Hyperparameters:

Optimizer: Adam(lr=0.001)

Batch Size: 16 (vs 32, smoother gradients)

Epochs: 75

Augmentation: rotation=10°, flip, brightness ±10%


## 🚀 Quick Start
```bash
pip install -r requirements.txt
python src/models/sania_cnn_v2.py --train

🎓 Key Learnings
Data augmentation prevented 30% overfitting

Smaller batch_size=16 > batch_size=32 for small datasets

F1-score > accuracy for imbalanced classes


