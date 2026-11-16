"""Client partitioning, split I/O and safe subsampling.

The partition functions reproduce FLipynb cell 4 exactly (same RNG call order), so they
regenerate the existing ``splits/*.npz`` when given the same cache and seed. Prefer
LOADING the existing split files: every reported result is tied to them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .contract import CLASSES

SCENARIOS = ["A_iid", "A_dir0.5", "A_dir0.1", "B_filehold_iid"]


# ----------------------------------------------------------------------------- I/O
def load_splits(path) -> Dict:
    """Return {'test': idx, 'clients': [(train_idx, val_idx), ...]} from a split .npz."""
    d = np.load(path)
    K = len([k for k in d.files if k.endswith("_train")])
    return {"test": d["test"].astype(np.int64),
            "clients": [(d[f"c{k}_train"].astype(np.int64), d[f"c{k}_val"].astype(np.int64)) for k in range(K)]}


def save_splits(path, test_idx, clients) -> None:
    arrays = {"test": np.asarray(test_idx, dtype=np.int32)}
    for k, (tr, va) in enumerate(clients):
        arrays[f"c{k}_train"] = np.asarray(tr, dtype=np.int32)
        arrays[f"c{k}_val"] = np.asarray(va, dtype=np.int32)
    np.savez_compressed(path, **arrays)


def check_integrity(split: Dict, n_rows: Optional[int] = None) -> Dict:
    """Disjointness (and completeness if n_rows given) of test + all client train/val sets."""
    parts = [split["test"]] + [a for tr, va in split["clients"] for a in (tr, va)]
    allidx = np.concatenate(parts)
    uniq = np.unique(allidx)
    out = {"disjoint": bool(len(uniq) == len(allidx)), "rows": int(len(allidx))}
    if n_rows is not None:
        out["complete"] = bool(len(uniq) == n_rows and uniq[0] == 0 and uniq[-1] == n_rows - 1)
    return out


# ------------------------------------------------------------------ partitioning (cell 4)
def local_split(idx, y, rng, local_val: float = 0.10):
    tr, va = [], []
    for c in np.unique(y[idx]):
        ci = rng.permutation(idx[y[idx] == c])
        nv = int(round(local_val * len(ci))) if len(ci) >= 10 else 0
        va.append(ci[:nv]); tr.append(ci[nv:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def iid_partition(pool, y, K, rng) -> List[np.ndarray]:
    shards = [[] for _ in range(K)]
    for c in range(len(CLASSES)):
        ci = rng.permutation(pool[y[pool] == c])
        for k, part in enumerate(np.array_split(ci, K)):
            shards[k].append(part)
    return [np.concatenate(s) for s in shards]


def dirichlet_partition(pool, y, K, alpha, rng, min_client=2000, benign_floor=0.5, max_tries=200):
    for t in range(1, max_tries + 1):
        shards = [[] for _ in range(K)]
        for c in range(len(CLASSES)):
            ci = rng.permutation(pool[y[pool] == c])
            p = rng.dirichlet(alpha * np.ones(K))
            if CLASSES[c] == "Benign":
                p = benign_floor / K + (1 - benign_floor) * p
            cuts = (np.cumsum(p) * len(ci)).astype(int)[:-1]
            for k, part in enumerate(np.split(ci, cuts)):
                shards[k].append(part)
        shards = [np.concatenate(s) for s in shards]
        if min(len(s) for s in shards) >= min_client:
            return shards, t
    raise RuntimeError("could not satisfy min_client; lower it or raise alpha")


def client_class_table(split: Dict, y) -> List[Dict]:
    rows = []
    for k, (tr, va) in enumerate(split["clients"]):
        cnt = np.bincount(y[tr], minlength=len(CLASSES))
        rows.append({"client": k, "n_train": int(len(tr)), "n_val": int(len(va)),
                     **{c: int(v) for c, v in zip(CLASSES, cnt)},
                     "classes_ge1pct": int((cnt / max(cnt.sum(), 1) >= 0.01).sum())})
    return rows


# ------------------------------------------------------------------ safe subsampling
def stratified_subsample(idx, y, n: int, seed: int = 0) -> np.ndarray:
    """Seeded, class-proportional subsample of ``idx`` (replaces taking the first n indices).

    Split index arrays are SORTED and the cache is ordered by source file (Benign first),
    so ``idx[:n]`` is almost entirely one class. This keeps every class's share and
    guarantees at least one row for each class that is present.
    """
    idx = np.asarray(idx)
    if n >= len(idx):
        return np.sort(idx)
    rng = np.random.default_rng(seed)
    labels = np.asarray(y)[idx]
    classes, counts = np.unique(labels, return_counts=True)
    quota = np.maximum(1, np.floor(counts / counts.sum() * n).astype(int))
    while quota.sum() > n:
        quota[np.argmax(quota)] -= 1
    rem = n - quota.sum()
    if rem > 0:
        frac = counts / counts.sum() * n - np.floor(counts / counts.sum() * n)
        for j in np.argsort(-frac)[:rem]:
            quota[j] += 1
    quota = np.minimum(quota, counts)
    picked = [rng.choice(idx[labels == c], size=q, replace=False) for c, q in zip(classes, quota)]
    return np.sort(np.concatenate(picked))
