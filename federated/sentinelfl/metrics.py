"""Classification metrics computed from a 7x7 confusion matrix (no sklearn needed).

Definitions (stated explicitly because papers differ):
  * per-class precision / recall / F1, macro = unweighted mean over the 7 classes
  * weighted = support-weighted mean
  * per-class FPR = FP_c / (FP_c + TN_c)  (one-vs-rest)
  * attack-detection view (binary, Benign vs any attack):
      binary_fpr = benign flows flagged as any attack / all benign flows
      binary_fnr = attack flows predicted Benign / all attack flows
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .contract import CLASSES


def confusion(y_true, y_pred, n_classes: int = len(CLASSES)) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def _safe_div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def metrics_from_cm(cm, classes: Sequence[str] = CLASSES, benign_index: Optional[int] = 0) -> Dict:
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    total = cm.sum()
    fp = predicted - tp
    fn = support - tp
    tn = total - tp - fp - fn
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    fpr = _safe_div(fp, fp + tn)
    present = support > 0
    w = support / support.sum() if support.sum() else support
    out = {
        "n_samples": int(total),
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * w).sum()),
        "classes_absent_from_eval": [c for c, p in zip(classes, present) if not p],
        "per_class": [
            {"class": c, "precision": float(p), "recall": float(r), "f1": float(f_), "fpr": float(fr),
             "support": int(s)}
            for c, p, r, f_, fr, s in zip(classes, precision, recall, f1, fpr, support)
        ],
        "confusion_matrix": cm.tolist(),
        "classes": list(classes),
    }
    if benign_index is not None and cm.shape[0] > 1:
        b = benign_index
        benign_total = cm[b].sum()
        attack_total = total - benign_total
        out["binary_fpr"] = float((benign_total - cm[b, b]) / benign_total) if benign_total else 0.0
        out["binary_fnr"] = float((cm[:, b].sum() - cm[b, b]) / attack_total) if attack_total else 0.0
    return out


def metrics(y_true, y_pred, classes: Sequence[str] = CLASSES) -> Dict:
    return metrics_from_cm(confusion(y_true, y_pred, len(classes)), classes)
