# Iceberg vs Ship Detection CNN (CS171 Intro to ML)
**SJSU CS171 Final Project** | **Test Accuracy: 77.5%** (vs 58% baseline)

## 🎯 Problem
Classify 75×75px satellite images as icebergs vs ships for maritime safety.

## 📊 Results
| Model | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| Baseline (2-layer) | 82% | 61% | **58%** |
| **Final (4-layer + aug)** | **94%** | **73%** | **77.5%** |

## 🧑‍💻 My Contributions (Sania Bandekar)
- ✅ Data preprocessing + augmentation (`notebooks/02_data_preprocessing.ipynb`)
- ✅ CNN architectures (`src/models/cnn_architectures.py`)
- ✅ Training + hyperparameter tuning (`notebooks/04_model_training_evaluation.ipynb`)
- ✅ Experiment tracking (`experiments/hyperparameter_tracking.csv`)

## 📁 Structure
<img width="773" height="340" alt="image" src="https://github.com/user-attachments/assets/51ab9a34-2ee8-4fd9-b83f-cad2c189c437" />



