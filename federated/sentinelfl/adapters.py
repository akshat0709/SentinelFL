"""Dataset adapters — the shared 'adapter architecture'.

An adapter turns a raw dataset (a pandas DataFrame read from a CSV) into the canonical
representation every model in SentinelFL consumes:

    AlignedData.X       DataFrame, exactly the 39 canonical columns in canonical order,
                        float64, +/-inf converted to NaN (nothing imputed, nothing dropped)
    AlignedData.labels  canonical class names (or None per row when unknown / absent)

What happens next depends on the use case and is deliberately NOT hidden in the adapter:

    training  -> clean_for_training(): drop NaN rows, drop exact duplicates and
                 label-conflicting rows (identical to the Colab cache builder, cell 3)
    inference -> impute_for_inference(): per-column median fill so every uploaded row
                 gets a prediction (identical to the backend's previous behaviour)

Then the model bundle's own scaler is applied (log1p+federated standardisation for FL
models, sklearn StandardScaler for the legacy centralized MLP).

Adapters never fabricate missing features: a dataset lacking any of the 39 canonical
columns raises DatasetCompatibilityError.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contract import CLASSES, CLASS_TO_ID, FEATURES, LABEL_COLUMN_CANDIDATES, normalize_label


class DatasetCompatibilityError(ValueError):
    """Raised when a dataset cannot be mapped to the canonical schema without fabricating data."""


@dataclass
class AlignedData:
    X: pd.DataFrame                      # (n, 39) canonical columns, float64, inf -> NaN
    labels: Optional[np.ndarray]         # (n,) object: canonical class name or None
    raw_labels: Optional[np.ndarray]     # (n,) object: label as found in the file
    report: Dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.X)

    def label_ids(self) -> np.ndarray:
        """Integer class ids; -1 where the label is unknown or missing."""
        if self.labels is None:
            raise ValueError("dataset has no label column")
        return np.array([CLASS_TO_ID.get(l, -1) if l is not None else -1 for l in self.labels], dtype=np.int64)


class BaseAdapter:
    dataset_type: str = "base"
    display_name: str = "Base adapter"
    supports_flmlp: bool = False

    def identify(self, df: pd.DataFrame) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def align(self, df: pd.DataFrame, label_column: Optional[str] = None) -> AlignedData:  # pragma: no cover
        raise NotImplementedError


def _find_label_column(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"label column '{explicit}' not found")
        return explicit
    stripped = {str(c).strip(): c for c in df.columns}
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in LABEL_COLUMN_CANDIDATES:
        if cand in stripped:
            return stripped[cand]
    for cand in LABEL_COLUMN_CANDIDATES:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


class CICIoT2023Adapter(BaseAdapter):
    """CIC IoT 2023 flow features -> canonical 39-feature / 7-class representation."""

    dataset_type = "cic_iot_2023"
    display_name = "CIC IoT 2023 (Canonical FLMLP Benchmark)"
    supports_flmlp = True
    MIN_OVERLAP_TO_IDENTIFY = 20

    def _column_map(self, df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
        lowered = {str(c).strip().lower(): c for c in df.columns}
        mapping, missing = {}, []
        for f in FEATURES:
            if f in df.columns:
                mapping[f] = f
            elif f.lower() in lowered:
                mapping[f] = lowered[f.lower()]
            else:
                missing.append(f)
        return mapping, missing

    def identify(self, df: pd.DataFrame) -> bool:
        cols = {str(c).strip().lower() for c in df.columns}
        return len(cols & {f.lower() for f in FEATURES}) >= self.MIN_OVERLAP_TO_IDENTIFY

    def validate(self, df: pd.DataFrame, label_column: Optional[str] = None) -> Dict:
        mapping, missing = self._column_map(df)
        label_col = _find_label_column(df, label_column)
        detected, supported, unsupported = [], [], []
        if label_col is not None:
            for raw in pd.Series(df[label_col]).dropna().astype(str).unique().tolist():
                detected.append(raw)
                canon = normalize_label(raw)
                if canon is None:
                    unsupported.append(raw)
                elif canon not in supported:
                    supported.append(canon)
        num = df.select_dtypes(include=[np.number])
        report = {
            "dataset_type": self.dataset_type,
            "dataset_name": self.display_name,
            "is_compatible": len(missing) == 0,
            "matched_features": [f for f in FEATURES if f in mapping],
            "missing_features": missing,
            "label_column": label_col,
            "detected_classes": detected,
            "supported_classes": supported,
            "unsupported_classes": unsupported,
            "row_count": int(len(df)),
            "column_count": int(df.shape[1]),
            "infinite_values_count": int(np.isinf(num.to_numpy(dtype=np.float64)).sum()) if num.shape[1] else 0,
            "nan_values_count": int(df.isna().to_numpy().sum()),
        }
        if missing:
            report["rejection_reason"] = (
                f"CIC IoT 2023 schema validation failed: missing {len(missing)} of 39 mandatory features "
                f"({missing[:5]}{'...' if len(missing) > 5 else ''}). Missing flow features are never "
                "zero-imputed or aliased."
            )
        return report

    def align(self, df: pd.DataFrame, label_column: Optional[str] = None) -> AlignedData:
        report = self.validate(df, label_column)
        if not report["is_compatible"]:
            raise DatasetCompatibilityError(report["rejection_reason"])
        mapping, _ = self._column_map(df)
        X = pd.DataFrame(
            {f: pd.to_numeric(df[mapping[f]], errors="coerce") for f in FEATURES},
            index=df.index,
        ).astype(np.float64)
        vals = X.to_numpy()
        bad = ~np.isfinite(vals) | (np.abs(np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)) > 1e300)
        report["non_finite_cells"] = int(bad.sum())
        if bad.any():
            vals = vals.copy()
            vals[bad] = np.nan
            X = pd.DataFrame(vals, columns=FEATURES, index=df.index)

        labels = raw = None
        if report["label_column"] is not None:
            raw = df[report["label_column"]].to_numpy(dtype=object)
            labels = np.array([normalize_label(v) for v in raw], dtype=object)
            report["unknown_label_rows"] = int(sum(1 for v in labels if v is None))
        return AlignedData(X=X, labels=labels, raw_labels=raw, report=report)


ADAPTERS: List[BaseAdapter] = [CICIoT2023Adapter()]


def get_adapter(name_or_df) -> BaseAdapter:
    """Return an adapter by dataset_type name, or detect it from a DataFrame's columns."""
    if isinstance(name_or_df, str):
        for a in ADAPTERS:
            if a.dataset_type == name_or_df:
                return a
        raise KeyError(f"unknown adapter '{name_or_df}'")
    for a in ADAPTERS:
        if a.identify(name_or_df):
            return a
    raise DatasetCompatibilityError("no registered adapter recognises this dataset schema")


# --------------------------------------------------------------------------------------
# Training-time cleaning (identical semantics to the Colab cache builder, FLipynb cell 3)
# --------------------------------------------------------------------------------------
def clean_for_training(
    X: np.ndarray,
    y: np.ndarray,
    file_id: Optional[np.ndarray] = None,
    drop_unknown_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    """Drop NaN rows, label-conflicting feature vectors and exact duplicates.

    Works on float32 values, i.e. exactly what the model sees. Keeps the FIRST occurrence
    of a duplicate (file order). Returns (X, y, file_id, stats).
    """
    X = np.asarray(X, dtype=np.float32).copy()
    y = np.asarray(y).astype(np.int64)
    fid = None if file_id is None else np.asarray(file_id)
    stats = {"rows_in": int(len(y))}

    keep = np.ones(len(y), dtype=bool)
    if drop_unknown_labels:
        keep &= y >= 0
        stats["unknown_label_rows_removed"] = int((y < 0).sum())
    X[~np.isfinite(X)] = np.nan
    nan_rows = np.isnan(X).any(axis=1)
    stats["nan_inf_rows_removed"] = int((nan_rows & keep).sum())
    keep &= ~nan_rows
    X, y = X[keep], y[keep]
    fid = None if fid is None else fid[keep]

    X[X == 0] = 0.0  # normalise -0.0 so it hashes like 0.0
    h = pd.util.hash_pandas_object(pd.DataFrame(X, columns=FEATURES), index=False).to_numpy()
    key = pd.DataFrame({"h": h, "y": y})
    conflict = (key.groupby("h")["y"].transform("nunique") > 1).to_numpy()
    dup = key.duplicated(["h", "y"], keep="first").to_numpy() & ~conflict
    stats["label_conflict_rows_removed"] = int(conflict.sum())
    stats["duplicate_rows_removed"] = int(dup.sum())
    if fid is not None:
        key["f"] = fid
        first_file = key.groupby(["h", "y"])["f"].transform("first").to_numpy()
        stats["cross_file_duplicates_removed"] = int((dup & (first_file != fid)).sum())
    keep2 = ~conflict & ~dup
    X, y = X[keep2], y[keep2].astype(np.int8)
    fid = None if fid is None else fid[keep2]
    stats["rows_out"] = int(len(y))
    stats["class_counts"] = {c: int(n) for c, n in zip(CLASSES, np.bincount(y, minlength=len(CLASSES)))}
    return X, y, fid, stats


# --------------------------------------------------------------------------------------
# Inference-time imputation (identical semantics to the backend's previous adapter)
# --------------------------------------------------------------------------------------
def impute_for_inference(X: pd.DataFrame, medians: Optional[Sequence[float]] = None) -> Tuple[pd.DataFrame, int]:
    """Fill NaN cells with per-column medians so every row can be scored.

    ``medians`` defaults to the medians of the batch itself (previous backend behaviour);
    pass training medians for batch-independent predictions. Returns (frame, n_filled).
    """
    vals = X.to_numpy(dtype=np.float64).copy()
    nan_mask = np.isnan(vals)
    if medians is None:
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(vals, axis=0) if len(vals) else np.zeros(vals.shape[1])
        med = np.nan_to_num(med, nan=0.0)
    else:
        med = np.asarray(medians, dtype=np.float64)
    if nan_mask.any():
        vals[nan_mask] = np.take(med, np.where(nan_mask)[1])
    return pd.DataFrame(vals, columns=list(X.columns), index=X.index), int(nan_mask.sum())
