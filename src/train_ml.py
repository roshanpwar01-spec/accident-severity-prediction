from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def run_ml(d, seed=42):
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), True),
        "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1), False),
        "XGBoost": (XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  eval_metric="mlogloss", random_state=seed), False),
        "LightGBM": (LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1), False),
    }
    out = {}
    for name, (m, scaled) in models.items():
        Xtr, Xte = (d["X_train_s"], d["X_test_s"]) if scaled else (d["X_train"], d["X_test"])
        m.fit(Xtr, d["y_train"])
        out[name] = m.predict_proba(Xte)
    return out
