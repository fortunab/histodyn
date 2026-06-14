import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def labels_from_cm(cm):
    cm = np.asarray(cm, dtype=int)
    tn, fp, fn, tp = cm.ravel()
    y_true = [0] * tn + [0] * fp + [1] * fn + [1] * tp
    y_pred = [0] * tn + [1] * fp + [0] * fn + [1] * tp
    return y_true, y_pred


def metrics_from_cm(cm):
    y_true, y_pred = labels_from_cm(cm)
    return {
        "Accuracy (%)": round(accuracy_score(y_true, y_pred) * 100, 1),
        "Precision (%)": round(precision_score(y_true, y_pred) * 100, 2),
        "Recall (%)": round(recall_score(y_true, y_pred) * 100, 2),
        "F1-score (%)": round(f1_score(y_true, y_pred) * 100, 2),
    }


def gads(p):
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))
