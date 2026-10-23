"""Run every model on the SAME leak-free split and write results/."""
import argparse, os
import pandas as pd
from src.preprocess import prepare
from src.evaluate import score, report, roc_plot
from src.train_ml import run_ml

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/augmented_accident_dataset.csv")
p.add_argument("--keep-leaky", action="store_true", help="keep Casualties/Fatalities features")
p.add_argument("--augment-to", type=int, default=None, help="augment TRAIN set to this size")
p.add_argument("--no-dl", action="store_true")
a = p.parse_args()

os.makedirs("results", exist_ok=True)
d = prepare(a.data, drop_leaky=not a.keep_leaky, augment_to=a.augment_to)
probs = run_ml(d)
if not a.no_dl:
    from src.train_dl import run_dl
    probs.update(run_dl(d))

rows = []
for name, pr in probs.items():
    rows.append(score(name, d["y_test"], pr))
    report(name, d["y_test"], pr, d["class_names"], "results")
roc_plot(probs, d["y_test"], d["class_names"], "results")
tbl = pd.DataFrame(rows).round(4)
tbl.to_csv("results/metrics.csv", index=False)
print("\n", tbl.to_string(index=False), f"\nChance level accuracy ~ {1/len(d['class_names']):.3f}")
