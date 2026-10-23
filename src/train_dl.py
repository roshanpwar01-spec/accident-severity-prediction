import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def _fit(model, Xtr, ytr, Xte, seed):
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    es = callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
    model.fit(Xtr, ytr, validation_split=0.2, epochs=100, batch_size=64, callbacks=[es], verbose=0)
    return model.predict(Xte, verbose=0)


def run_dl(d, seed=42):
    tf.keras.utils.set_random_seed(seed)
    Xtr, Xte, ytr = d["X_train_s"], d["X_test_s"], d["y_train"]
    n, k = Xtr.shape[1], len(d["class_names"])
    out = {}
    out["DNN"] = _fit(models.Sequential([
        layers.Input((n,)), layers.Dense(64, activation="relu"), layers.Dropout(0.3),
        layers.Dense(32, activation="relu"), layers.Dense(k, activation="softmax")]),
        Xtr, ytr, Xte, seed)
    # Records are independent (no temporal order), so the features are treated as a
    # length-n sequence of 1-d steps. This is a model comparison, not true time series.
    seq = lambda a: a.reshape(a.shape[0], n, 1)
    for name, L in [("LSTM", layers.LSTM), ("GRU", layers.GRU)]:
        out[name] = _fit(models.Sequential([
            layers.Input((n, 1)), L(32), layers.Dropout(0.3), layers.Dense(k, activation="softmax")]),
            seq(Xtr), ytr, seq(Xte), seed)
    return out
