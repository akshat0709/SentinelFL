"""ModelBundle = model weights + the adapter + the scaler they were trained with.

A bundle is a directory:

    bundle.json   metadata: name, kind, adapter, features, classes, preprocessing,
                  training config, validation/test metrics, provenance
    model.pt      FLMLP state_dict (keys network.0/3/6/8)
    scaler.json   (log1p_standard)  or  scaler.joblib (legacy sklearn StandardScaler)

Packaging the preprocessing WITH the weights is what prevents the silent failure of
serving an FL checkpoint with the legacy scaler (or vice versa). Any consumer — the
backend, a notebook, a Raspberry Pi client — does:

    bundle = ModelBundle.load(path)
    result = bundle.predict_dataframe(raw_csv_dataframe)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from .adapters import get_adapter, impute_for_inference
from .contract import CLASSES, FEATURES
from .scalers import Log1pStandardScaler
from .model import FLMLP


class ModelBundle:
    def __init__(self, model: FLMLP, scaler, meta: Dict):
        self.model = model.eval()
        self.scaler = scaler
        self.meta = dict(meta)
        self.adapter = get_adapter(self.meta.get("adapter", "cic_iot_2023"))
        if list(self.meta.get("features", FEATURES)) != list(FEATURES):
            raise ValueError("bundle feature order differs from the canonical contract")
        self.classes = list(self.meta.get("classes", CLASSES))

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        meta = dict(self.meta)
        if isinstance(self.scaler, Log1pStandardScaler):
            self.scaler.save(path / "scaler.json")
            meta["preprocessing"] = {"type": "log1p_standard", "file": "scaler.json"}
        else:
            import joblib
            joblib.dump(self.scaler, path / "scaler.joblib")
            meta["preprocessing"] = {"type": "sklearn_joblib", "file": "scaler.joblib"}
        meta.setdefault("created_at", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
        meta["features"] = list(FEATURES)
        meta["classes"] = list(self.classes)
        with open(path / "bundle.json", "w") as f:
            json.dump(meta, f, indent=2, default=_json_default)
        return path

    @classmethod
    def load(cls, path, map_location="cpu") -> "ModelBundle":
        path = Path(path)
        with open(path / "bundle.json") as f:
            meta = json.load(f)
        prep = meta.get("preprocessing", {})
        if prep.get("type") == "log1p_standard":
            scaler = Log1pStandardScaler.load(path / prep.get("file", "scaler.json"))
        elif prep.get("type") == "sklearn_joblib":
            import joblib
            scaler = joblib.load(path / prep.get("file", "scaler.joblib"))
        else:
            raise ValueError(f"unknown preprocessing type in bundle: {prep}")
        model = FLMLP(n_in=len(meta.get("features", FEATURES)), n_out=len(meta.get("classes", CLASSES)))
        state = torch.load(path / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state, strict=True)
        return cls(model, scaler, meta)

    @classmethod
    def from_legacy_artifacts(cls, artifacts_dir, name: str = "legacy-centralized-mlp") -> "ModelBundle":
        """Wrap ml/artifacts (mlp_model.pth + scaler.joblib) as a bundle without retraining."""
        import joblib
        d = Path(artifacts_dir)
        ckpt = torch.load(d / "mlp_model.pth", map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        feats = list(joblib.load(d / "feature_columns.joblib"))
        classes = list(joblib.load(d / "label_encoder.joblib").classes_)
        if feats != list(FEATURES) or classes != list(CLASSES):
            raise ValueError("legacy artifacts do not match the canonical 39-feature / 7-class contract")
        model = FLMLP()
        model.load_state_dict(state, strict=True)
        meta = {
            "name": name, "kind": "legacy_centralized", "adapter": "cic_iot_2023",
            "provenance": "ml/artifacts/mlp_model.pth + scaler.joblib (trained before deduplication; "
                          "no reproducible evaluation run in this repository)",
        }
        return cls(model, joblib.load(d / "scaler.joblib"), meta)

    # ------------------------------------------------------------------ inference
    def preprocess(self, df: pd.DataFrame, medians=None):
        aligned = self.adapter.align(df)
        X, n_filled = impute_for_inference(aligned.X, medians)
        return self.scaler.transform(X).astype(np.float32), aligned, n_filled

    @torch.no_grad()
    def predict_proba_scaled(self, X_scaled: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        self.model.eval()
        out = [torch.softmax(self.model(torch.from_numpy(np.ascontiguousarray(X_scaled[i:i + batch_size]))), 1).numpy()
               for i in range(0, len(X_scaled), batch_size)]
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), np.float32)

    def predict_dataframe(self, df: pd.DataFrame) -> Dict:
        X, aligned, n_filled = self.preprocess(df)
        proba = self.predict_proba_scaled(X)
        pred = proba.argmax(1)
        res = {
            "predicted_class": [self.classes[i] for i in pred],
            "class_index": pred.tolist(),
            "probabilities": proba,
            "actual_label": None if aligned.labels is None else aligned.labels.tolist(),
            "imputed_cells": n_filled,
            "adapter_report": aligned.report,
        }
        return res

    def describe(self) -> Dict:
        return {k: self.meta.get(k) for k in ("name", "kind", "adapter", "preprocessing", "training", "metrics",
                                              "provenance", "created_at")}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, os.PathLike):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
