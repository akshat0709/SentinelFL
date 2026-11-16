"""Build the cleaned training cache from raw CSVs THROUGH THE SHARED ADAPTER.

Equivalent to FLipynb cell 3 (load -> NaN/inf rows dropped -> label conflicts and exact
duplicates removed), but every file now passes through ``CICIoT2023Adapter.align`` — the
same adapter the backend uses for inference.

Output (same layout the notebook produced, so existing splits stay valid):
    X_clean.npy (float32, n x 39)  y_clean.npy (int8)  file_id.npy (int16)  clean_meta.json

Rebuilding over an existing cache changes row indices and invalidates every split and
result; ``build_cache`` refuses unless overwrite=True. Use ``verify_cache`` to check that
an existing cache equals what the adapter pipeline produces.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .adapters import CICIoT2023Adapter, clean_for_training
from .contract import CLASSES, FEATURES

SYNTHETIC_EXCLUDE = ("synthetic_gmm_stress_test.csv",)


def list_csvs(data_dir, exclude: Sequence[str] = SYNTHETIC_EXCLUDE, include: Optional[Sequence[str]] = None) -> List[str]:
    files = sorted(f for f in glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
                   if os.path.basename(f) not in exclude)
    if include is not None:
        keep = set(include)
        files = [f for f in files if os.path.relpath(f, data_dir) in keep or os.path.basename(f) in keep]
    return files


def load_through_adapter(files: Sequence[str], data_dir: str, strict_labels: bool = True, log_fn=print):
    adapter = CICIoT2023Adapter()
    Xs, ys, fids, names = [], [], [], []
    for fid, f in enumerate(files):
        df = pd.read_csv(f, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        aligned = adapter.align(df)
        if aligned.labels is None:
            raise ValueError(f"{f}: no label column found")
        ids = aligned.label_ids()
        if strict_labels and (ids < 0).any():
            bad = sorted({str(r) for r, i in zip(aligned.raw_labels, ids) if i < 0})[:5]
            raise ValueError(f"{os.path.relpath(f, data_dir)} has labels outside the 7 classes: {bad}")
        Xs.append(aligned.X.to_numpy(dtype=np.float32))
        ys.append(ids)
        fids.append(np.full(len(ids), fid, np.int16))
        names.append(os.path.relpath(f, data_dir))
        if log_fn:
            log_fn(f"  {names[-1]}: {len(ids):,} rows")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(fids), names


def build_cache(data_dir, out_dir, files: Optional[Sequence[str]] = None, overwrite: bool = False,
                log_fn=print) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "X_clean.npy")) and not overwrite:
        raise FileExistsError(f"{out_dir} already has a cache; existing splits/results depend on it. "
                              "Pass overwrite=True only if you will regenerate all splits and runs.")
    t0 = time.time()
    files = list(files) if files is not None else list_csvs(data_dir)
    X, y, fid, names = load_through_adapter(files, data_dir, log_fn=log_fn)
    Xc, yc, fidc, stats = clean_for_training(X, y, fid)
    np.save(os.path.join(out_dir, "X_clean.npy"), Xc)
    np.save(os.path.join(out_dir, "y_clean.npy"), yc)
    np.save(os.path.join(out_dir, "file_id.npy"), fidc.astype(np.int16))
    meta = {"features": FEATURES, "classes": CLASSES, "files": names, "stats": stats,
            "built_with": "sentinelfl.cache.build_cache (CICIoT2023Adapter + clean_for_training)",
            "secs": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "clean_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def verify_cache(data_dir, cache_dir, log_fn=print) -> Dict:
    """Rebuild in memory through the adapter (using the file list recorded in the cache's
    clean_meta.json) and compare with the cache on disk. Nothing is written."""
    with open(os.path.join(cache_dir, "clean_meta.json")) as f:
        meta = json.load(f)
    recorded = meta.get("files")
    files = [os.path.join(data_dir, n) for n in recorded] if recorded else list_csvs(data_dir)
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        return {"ok": False, "reason": f"source files missing: {missing[:3]}"}
    X, y, fid, _ = load_through_adapter(files, data_dir, log_fn=None)
    Xc, yc, fidc, stats = clean_for_training(X, y, fid)
    X0 = np.load(os.path.join(cache_dir, "X_clean.npy"), mmap_mode="r")
    y0 = np.load(os.path.join(cache_dir, "y_clean.npy"))
    same_shape = X0.shape == Xc.shape and y0.shape == yc.shape
    out = {"ok": False, "rows_cache": int(len(y0)), "rows_rebuilt": int(len(yc)), "files_used": len(files)}
    if same_shape:
        out["labels_equal"] = bool(np.array_equal(y0, yc))
        out["max_abs_diff"] = float(np.max(np.abs(np.asarray(X0) - Xc))) if len(yc) else 0.0
        out["ok"] = out["labels_equal"] and out["max_abs_diff"] == 0.0
    if log_fn:
        log_fn(f"cache verification: {out}")
    return out
