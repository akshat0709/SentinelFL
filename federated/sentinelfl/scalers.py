"""Feature scaling used by every FLMLP trained in the federated pipeline.

log1p followed by standardisation whose mean/std are computed FEDERATEDLY: each client
sends only (n, sum, sum of squares) of its log1p-transformed training data, never rows.
Numerically identical to fl_core.local_stats / aggregate_stats / transform.

``Log1pStandardScaler.transform`` accepts DataFrames or arrays, so it is a drop-in
replacement for an sklearn scaler inside the backend's adapter pipeline.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .contract import FEATURES


def local_stats(X_raw) -> Tuple[int, np.ndarray, np.ndarray]:
    """Client side: aggregates of log1p(X) over the client's TRAINING rows only."""
    Z = np.log1p(np.maximum(np.asarray(X_raw, dtype=np.float64), 0.0))
    return int(len(Z)), Z.sum(axis=0), (Z ** 2).sum(axis=0)


def aggregate_stats(stats: Iterable[Tuple[int, np.ndarray, np.ndarray]]):
    """Server side: combine client aggregates into global mean / std."""
    stats = list(stats)
    N = sum(s[0] for s in stats)
    if N == 0:
        raise ValueError("no samples to aggregate")
    S = sum(s[1] for s in stats)
    SS = sum(s[2] for s in stats)
    mean = S / N
    std = np.sqrt(np.maximum(SS / N - mean ** 2, 0.0))
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), int(N)


class Log1pStandardScaler:
    kind = "log1p_standard"

    def __init__(self, mean, std, n_samples: Optional[int] = None, feature_names=None):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(std, dtype=np.float32)
        self.n_samples_seen_ = n_samples
        self.feature_names_in_ = np.array(feature_names if feature_names is not None else FEATURES, dtype=object)
        if self.mean_.shape != (len(self.feature_names_in_),) or self.scale_.shape != self.mean_.shape:
            raise ValueError("mean/std must have one value per feature")
        self.negative_values_clipped_ = 0

    # ---- construction -------------------------------------------------------------
    @classmethod
    def from_client_stats(cls, stats, feature_names=None) -> "Log1pStandardScaler":
        mean, std, n = aggregate_stats(stats)
        return cls(mean, std, n, feature_names)

    @classmethod
    def fit(cls, X_raw, feature_names=None) -> "Log1pStandardScaler":
        """Pooled fit — equal to the federated fit over the same rows (used for tests/baselines)."""
        return cls.from_client_stats([local_stats(X_raw)], feature_names)

    # ---- use ----------------------------------------------------------------------
    def transform(self, X) -> np.ndarray:
        if hasattr(X, "columns"):
            missing = [f for f in self.feature_names_in_ if f not in X.columns]
            if missing:
                raise ValueError(f"missing features for scaling: {missing[:5]}")
            X = X[list(self.feature_names_in_)].to_numpy(dtype=np.float32)
        X = np.asarray(X, dtype=np.float32)
        neg = X < 0
        if neg.any():
            # Training data has no negative values (checked in FLipynb cell 4); log1p is
            # undefined below -1, so negatives are clipped to 0 and counted.
            self.negative_values_clipped_ += int(neg.sum())
            X = np.where(neg, 0.0, X)
        return ((np.log1p(X) - self.mean_) / self.scale_).astype(np.float32)

    # ---- persistence ----------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "type": self.kind,
            "features": list(map(str, self.feature_names_in_)),
            "mean": [float(v) for v in self.mean_],
            "std": [float(v) for v in self.scale_],
            "n_samples": self.n_samples_seen_,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Log1pStandardScaler":
        feats = d.get("features")
        if not isinstance(feats, list):  # Colab scaler JSONs store the feature COUNT here
            feats = None
        return cls(d["mean"], d["std"], d.get("n_samples") or d.get("n_train"), feats)

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path) -> "Log1pStandardScaler":
        with open(path) as f:
            return cls.from_dict(json.load(f))
