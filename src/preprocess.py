"""Leak-free data preparation.

The provided dataset (augmented_accident_dataset.csv) is already label-encoded and
already augmented: ~3,000 original records, each repeated, plus noisy copies with
non-integer values. Randomly splitting that file puts near-copies of the same record
in both train and test (data leakage -> ~97% accuracy that means nothing).

Strategy:
  1. Recover the original records (integer-valued rows), drop exact duplicates.
  2. Split the originals into train/test  (test set contains no augmented rows).
  3. Optionally re-augment the TRAIN set only (bootstrap + noise).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Accident Severity"
# Outcomes of an accident, not known beforehand -> target leakage if used as inputs.
LEAKY_COLS = ["Number of Casualties", "Number of Fatalities"]
CONTINUOUS = ["Number of Vehicles Involved", "Number of Casualties",
              "Number of Fatalities", "Speed Limit (km/h)", "Driver Age"]
NOISE_COLS = ["Speed Limit (km/h)", "Driver Age"]


def load_raw(path):
    # the supplied ".xls" is really a CSV; try CSV first, then Excel
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_excel(path)


def recover_originals(df):
    is_int = (df[CONTINUOUS].round(6) == df[CONTINUOUS].round(0)).all(axis=1)
    orig = df[is_int].drop_duplicates().reset_index(drop=True)
    print(f"[data] raw rows={len(df)}  exact duplicates={df.duplicated().sum()}  "
          f"original unique records={len(orig)}")
    return orig


def augment(X, y, target_size, seed=42):
    """Bootstrap + small noise on continuous columns. Used on TRAIN ONLY."""
    rng = np.random.default_rng(seed)
    need = target_size - len(X)
    if need <= 0:
        return X, y
    idx = rng.integers(0, len(X), need)
    Xa, ya = X.iloc[idx].copy().reset_index(drop=True), y[idx]
    for c in NOISE_COLS:
        lo, hi = X[c].min(), X[c].max()
        Xa[c] = np.clip(Xa[c] + rng.normal(0, 1.0, need), lo, hi).round()
    return pd.concat([X, Xa], ignore_index=True), np.concatenate([y, ya])


def prepare(path, drop_leaky=True, augment_to=None, test_size=0.2, seed=42):
    df = recover_originals(load_raw(path))
    X = df.drop(columns=[TARGET])
    if drop_leaky:
        X = X.drop(columns=LEAKY_COLS)
    y = df[TARGET].values.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr = X_tr.reset_index(drop=True)
    if augment_to:
        X_tr, y_tr = augment(X_tr, y_tr, augment_to, seed)
    scaler = StandardScaler().fit(X_tr)
    print(f"[split] train={len(X_tr)} test={len(X_te)} features={X.shape[1]} "
          f"drop_leaky={drop_leaky}")
    return dict(X_train=X_tr, X_test=X_te.reset_index(drop=True), y_train=y_tr, y_test=y_te,
                X_train_s=scaler.transform(X_tr), X_test_s=scaler.transform(X_te),
                class_names=["0", "1", "2"], feature_names=list(X.columns))
