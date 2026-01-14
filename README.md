# Iceberg vs Ship Detection CNN - Sania's Contributions
**CS171 Intro to ML, San José State University**  
*Collaborators: Sania Bandekar (ML Pipeline), Victoria Lee (Data Exploration)*

## 🎯 Business Problem
Classify satellite images (75x75px) as icebergs vs ships for maritime safety.  
**Dataset**: 1,604 train / 8,424 test images  
**Baseline**: 58% test accuracy → **My final model**: **77.5% test accuracy**

## 🛠 My Contributions (LinkedIn REACH Application)
| Task | My Files | Key Decisions |
|------|----------|---------------|
| Data Prep | `src/data_preprocessing_sania.py` | Normalization, 3x augmentation |
| Model Design | `notebooks/my_experiments_sania.ipynb` | 4-layer CNN (64→512 filters) |
| Training | `src/train_sania.py` | Batch=16, dropout=0.2, 75 epochs |
| Evaluation | `results/sania_model_comparison.csv` | +19.3% over baseline |

## 📊 Results
| Model                   | Test Acc | Improvement |
| ----------------------- | -------- | ----------- |
| Random Classifier       | 50%      |             |
| Victoria's Baseline CNN | 58%      |             |
| Sania's Final CNN       | 77.5%    |             |

[Full experiment log](results/sania_model_comparison.csv)

## 🚀 Run My Pipeline
```bash
pip install -r requirements.txt
python src/data_preprocessing_sania.py
python src/train_sania.py
python src/evaluate_sania.py


📈 Key Learnings for LinkedIn REACH
1.Systematic hyperparameter tuning (+19% lift)

2.Overfitting mitigation (dropout + augmentation)

3.Production evaluation (confusion matrices, F1 scores)

4.Reproducible experiments (seeded, logged results)



