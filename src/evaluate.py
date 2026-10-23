import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize


def score(name, y_true, proba):
    pred = proba.argmax(1)
    k = proba.shape[1]
    return dict(model=name,
                accuracy=accuracy_score(y_true, pred),
                macro_f1=f1_score(y_true, pred, average="macro"),
                roc_auc_ovr=roc_auc_score(y_true, proba, multi_class="ovr"))


def report(name, y_true, proba, class_names, out_dir):
    pred = proba.argmax(1)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, pred, target_names=class_names, zero_division=0))
    cm = confusion_matrix(y_true, pred)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{name}")
    plt.tight_layout(); plt.savefig(f"{out_dir}/cm_{name.replace(' ', '_')}.png", dpi=120); plt.close()


def roc_plot(results, y_true, class_names, out_dir):
    """results: {model: proba}. Macro-average one-vs-rest ROC per model."""
    Y = label_binarize(y_true, classes=range(len(class_names)))
    plt.figure(figsize=(6, 5))
    for name, proba in results.items():
        fpr, tpr, _ = roc_curve(Y.ravel(), proba.ravel())
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(Y, proba, average='micro'):.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC (micro-average, one-vs-rest)"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{out_dir}/roc_curves.png", dpi=120); plt.close()
