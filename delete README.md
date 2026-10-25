# Road Accident Severity Prediction

Multi-class classification (3 severity classes) of road accidents using classical ML
(Logistic Regression, Random Forest, XGBoost, LightGBM) and deep learning (DNN, LSTM, GRU).

## Important finding about the dataset

`data/augmented_accident_dataset.csv` (9,097 rows) contains only **2,999 unique original
records**: each is repeated twice, plus ~3,100 noisy copies. A normal random train/test split
therefore puts copies of the same record on both sides:

| Evaluation                                   | Random Forest accuracy |
|----------------------------------------------|------------------------|
| Random split of the full augmented file      | ~97% (leakage)         |
| Split the original records first (this repo) | ~33% (= chance for 3 classes) |

This repo recovers the original records, splits **before** any augmentation, drops
`Number of Casualties` / `Number of Fatalities` (outcomes, not predictors), and evaluates
every model on the **same** test set (600 records).

## Results (leak-free, 20% test split, 3 balanced classes, chance = 0.333)

| Model | Accuracy | Macro F1 | ROC-AUC (OvR) |
|---|---|---|---|
| Logistic Regression | 0.350 | 0.346 | 0.497 |
| Random Forest | 0.337 | 0.334 | 0.511 |
| XGBoost | 0.317 | 0.316 | 0.499 |
| LightGBM | 0.332 | 0.332 | 0.519 |
| DNN | 0.332 | 0.330 | 0.484 |
| LSTM | 0.322 | 0.308 | 0.482 |
| GRU | 0.312 | 0.244 | 0.486 |

All models are at chance level (AUC ≈ 0.5): **the available features carry no usable signal for
severity** (class-wise feature means are nearly identical). The data appears synthetic/random,
so higher accuracies reported elsewhere come from leakage, not learning.

Notes: the severity labels are integer-encoded 0/1/2 (original names not preserved in this file).
LSTM/GRU treat the features as a sequence for comparison only; records have no temporal order.

## Run

```bash
pip install -r requirements.txt
python run_all.py                      # all models, leak-free
python run_all.py --keep-leaky         # keep casualty/fatality features
python run_all.py --augment-to 5000    # augment TRAIN set only
python run_all.py --no-dl              # skip deep learning
```
Outputs: `results/metrics.csv`, confusion matrices, `results/roc_curves.png`.

## Structure
```
data/        dataset
notebooks/   original notebook (kept for reference; contains the leakage issues above)
src/         preprocess.py, train_ml.py, train_dl.py, evaluate.py
results/     metrics and plots
run_all.py   entry point
```
