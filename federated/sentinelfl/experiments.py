"""End-to-end experiment runner (resumable).

For each (scenario, seed) it trains — on the SAME cleaned cache, splits, federated scaler
and initial weights —

    fedavg       10-client FedAvg, R rounds x E local epochs
    centralized  pooled union of client train splits, R*E epochs (same number of passes)
    local        every client alone on its own data (non-IID harm / benefit of federation)
    xgboost      corrected 7-class XGBoost on the same splits (optional, needs xgboost)

selects every model by BEST VALIDATION macro-F1, and only then evaluates the selected
checkpoint ONCE on the scenario's held-out test set. Each finished run writes
``result.json`` (+ a ModelBundle); re-running skips finished runs, so a Colab disconnect
only loses the run in progress.

Typical use (Colab):
    from sentinelfl.experiments import run_grid, summarize
    run_grid(cache_dir, splits_dir, out_dir, scenarios=[...], seeds=[42, 43, 44])
    summarize(out_dir)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from .bundle import ModelBundle
from .contract import CLASSES, FEATURES
from .metrics import confusion, metrics_from_cm
from .model import FLMLP, payload_bytes, set_seed
from .partition import check_integrity, client_class_table, load_splits, stratified_subsample
from .scalers import Log1pStandardScaler, local_stats
from .training import (ClientData, bytes_raw_data, evaluate_cm, predict, run_centralized, run_fedavg,
                       run_local_only)

METHODS = ("fedavg", "centralized", "local", "xgboost")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _dump(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)


def _env() -> Dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cpu_threads": torch.get_num_threads()}


class _Data:
    """Scaled client data for one scenario; the test set is loaded lazily (only for final eval)."""

    def __init__(self, cache_dir, split_path, max_train_per_client=None, max_val_per_client=None,
                 subsample_seed=0, log_fn=print):
        self.X = np.load(os.path.join(cache_dir, "X_clean.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(cache_dir, "y_clean.npy")).astype(np.int64)
        self.split = load_splits(split_path)
        self.integrity = check_integrity(self.split, len(self.y))
        if not self.integrity["disjoint"]:
            raise ValueError(f"{split_path}: train/val/test overlap!")
        self.subsampled = bool(max_train_per_client or max_val_per_client)
        clients_idx = []
        for k, (tr, va) in enumerate(self.split["clients"]):
            if max_train_per_client:
                tr = stratified_subsample(tr, self.y, max_train_per_client, subsample_seed + k)
            if max_val_per_client:
                va = stratified_subsample(va, self.y, max_val_per_client, subsample_seed + 100 + k)
            clients_idx.append((tr, va))
        self.clients_idx = clients_idx
        # federated scaler: each client contributes only (n, sum, sumsq) of its TRAIN rows
        self.scaler = Log1pStandardScaler.from_client_stats([local_stats(self.X[tr]) for tr, _ in clients_idx])
        self.clients = [ClientData(self.scaler.transform(self.X[tr]), self.y[tr],
                                   self.scaler.transform(self.X[va]), self.y[va]) for tr, va in clients_idx]
        self.client_table = client_class_table({"clients": clients_idx}, self.y)
        if log_fn:
            log_fn(f"  clients={len(self.clients)} train={sum(c.n_train for c in self.clients):,} "
                   f"val={sum(len(c.y_val) for c in self.clients):,}"
                   f"{'  (SUBSAMPLED)' if self.subsampled else ''}")

    def pooled(self):
        Xtr = np.concatenate([c.X_train for c in self.clients]); ytr = np.concatenate([c.y_train for c in self.clients])
        Xva = np.concatenate([c.X_val for c in self.clients]); yva = np.concatenate([c.y_val for c in self.clients])
        return Xtr, ytr, Xva, yva

    def raw_pooled(self):
        tr = np.concatenate([t for t, _ in self.clients_idx]); va = np.concatenate([v for _, v in self.clients_idx])
        return np.asarray(self.X[tr]), self.y[tr], np.asarray(self.X[va]), self.y[va]

    def test(self, max_rows=None):
        te = self.split["test"]
        if max_rows:
            te = stratified_subsample(te, self.y, max_rows, 999)
        return np.asarray(self.X[te]), self.y[te]


def _test_eval_torch(state, scaler, X_test_raw, y_test) -> Dict:
    model = FLMLP(); model.load_state_dict(state); model.eval()
    Xs = scaler.transform(X_test_raw)
    t0 = time.perf_counter()
    pred = predict(model, Xs)
    secs = time.perf_counter() - t0
    m = metrics_from_cm(confusion(y_test, pred))
    m["inference_ms_per_flow"] = 1000 * secs / max(1, len(y_test))
    return m


def _client_val_metrics(state, clients) -> List[Dict]:
    model = FLMLP(); model.load_state_dict(state); model.eval()
    out = []
    for k, c in enumerate(clients):
        m = metrics_from_cm(evaluate_cm(model, c.X_val, c.y_val))
        out.append({"client": k, "n_val": int(len(c.y_val)), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                    "recall_per_class": {p["class"]: p["recall"] for p in m["per_class"]}})
    return out


def run_one(cache_dir, splits_dir, out_dir, scenario: str, seed: int, methods: Sequence[str] = ("fedavg", "centralized"),
            rounds: int = 20, local_epochs: int = 1, epochs: Optional[int] = None, batch_size: int = 512,
            lr: float = 1e-3, max_train_per_client: Optional[int] = None, max_val_per_client: Optional[int] = None,
            max_test_rows: Optional[int] = None, save_bundles: bool = True, log_fn=print) -> Dict[str, Dict]:
    run_dir = Path(out_dir) / "runs" / f"{scenario}_s{seed}"
    todo = [m for m in methods if not (run_dir / m / "result.json").exists()]
    done = {m: json.load(open(run_dir / m / "result.json")) for m in methods if m not in todo}
    if not todo:
        log_fn(f"[skip] {scenario} seed={seed}: all requested methods already finished")
        return done
    log_fn(f"[run] {scenario} seed={seed} methods={todo}")
    data = _Data(cache_dir, os.path.join(splits_dir, f"{scenario}.npz"), max_train_per_client,
                 max_val_per_client, subsample_seed=seed, log_fn=log_fn)
    epochs = epochs or rounds * local_epochs
    set_seed(seed)
    init_state = FLMLP().state_dict()            # same init for every method (FLipynb server)
    common = {"scenario": scenario, "seed": seed, "batch_size": batch_size, "lr": lr,
              "n_clients": len(data.clients), "subsampled": data.subsampled,
              "max_train_per_client": max_train_per_client, "split_integrity": data.integrity,
              "scaler": data.scaler.to_dict(), "client_class_table": data.client_table,
              "env": _env(), "started_at": _now()}
    results = dict(done)
    for method in todo:
        mdir = run_dir / method
        mdir.mkdir(parents=True, exist_ok=True)
        log_fn(f" -> {method}")
        t0 = time.time()
        if method == "fedavg":
            r = run_fedavg(data.clients, init_state, rounds, local_epochs, batch_size, lr, seed, log_fn)
            sel = {"rule": "best validation macro-F1 over rounds", "round": r["best_round"],
                   "val_macro_f1": r["best_val_macro_f1"]}
            res = {**common, "method": "fedavg", "rounds": rounds, "local_epochs": local_epochs,
                   "history": r["history"], "selection": sel,
                   "client_val_at_selection": _client_val_metrics(r["best_state"], data.clients),
                   "communication": {"payload_bytes_per_transfer": r["payload_bytes_per_transfer"],
                                     "bytes_total": r["bytes_total"],
                                     "bytes_to_reach_selection": r["history"][r["best_round"] - 1]["bytes_cumulative"],
                                     "raw_data_bytes_if_centralized": bytes_raw_data(sum(c.n_train for c in data.clients))}}
            state, kind = r["best_state"], "federated"
        elif method == "centralized":
            Xtr, ytr, Xva, yva = data.pooled()
            r = run_centralized(Xtr, ytr, Xva, yva, init_state, epochs, batch_size, lr, seed, log_fn)
            sel = {"rule": "best validation macro-F1 over epochs", "epoch": r["best_epoch"],
                   "val_macro_f1": r["best_val_macro_f1"]}
            res = {**common, "method": "centralized", "epochs": epochs, "history": r["history"], "selection": sel,
                   "communication": {"raw_data_bytes_if_centralized": bytes_raw_data(len(ytr))}}
            state, kind = r["best_state"], "centralized"
        elif method == "local":
            rs = run_local_only(data.clients, init_state, epochs, batch_size, lr, seed, log_fn)
            X_te, y_te = data.test(max_test_rows)
            per_client = []
            for rr in rs:
                tm = _test_eval_torch(rr["best_state"], data.scaler, X_te, y_te)
                per_client.append({"client": rr["client"], "best_epoch": rr["best_epoch"],
                                   "val_macro_f1": rr["best_val_macro_f1"], "test": tm})
            f1s = [p["test"]["macro_f1"] for p in per_client]
            accs = [p["test"]["accuracy"] for p in per_client]
            res = {**common, "method": "local", "epochs": epochs, "per_client": per_client,
                   "test_summary": {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_min": float(np.min(f1s)),
                                    "macro_f1_max": float(np.max(f1s)), "accuracy_mean": float(np.mean(accs))},
                   "note": "Each client's own model evaluated on the shared global test set."}
            # representative 'test' block for tables = the median client by macro-F1
            med = sorted(per_client, key=lambda p: p["test"]["macro_f1"])[len(per_client) // 2]
            res["test"] = med["test"]; res["test_is_median_client"] = med["client"]
            res["wall_secs"] = time.time() - t0; res["finished_at"] = _now()
            _dump(res, mdir / "result.json"); results[method] = res
            continue
        elif method == "xgboost":
            res = _run_xgboost(data, seed, max_test_rows, log_fn)
            if res is None:
                continue
            res = {**common, **res}
            res["wall_secs"] = time.time() - t0; res["finished_at"] = _now()
            _dump(res, mdir / "result.json"); results[method] = res
            continue
        else:
            raise ValueError(f"unknown method {method}")

        # ---- single, final test evaluation of the SELECTED checkpoint ----------------
        X_te, y_te = data.test(max_test_rows)
        res["test"] = _test_eval_torch(state, data.scaler, X_te, y_te)
        res["test_rows_subsampled"] = bool(max_test_rows)
        res["wall_secs"] = time.time() - t0
        res["finished_at"] = _now()
        if save_bundles:
            model = FLMLP(); model.load_state_dict(state)
            meta = {"name": f"{method}-{scenario}-s{seed}", "kind": kind, "adapter": "cic_iot_2023",
                    "training": {k: res.get(k) for k in ("scenario", "seed", "rounds", "local_epochs", "epochs",
                                                         "batch_size", "lr", "n_clients", "subsampled", "selection")},
                    "metrics": {"validation_macro_f1": res["selection"]["val_macro_f1"],
                                "test": {k: res["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1",
                                                                     "binary_fpr", "binary_fnr")}},
                    "provenance": "sentinelfl.experiments.run_one"}
            ModelBundle(model, data.scaler, meta).save(mdir / "bundle")
        _dump(res, mdir / "result.json")
        results[method] = res
        log_fn(f"    test acc={res['test']['accuracy']:.4f} macroF1={res['test']['macro_f1']:.4f} "
               f"(selected at {sel})")
    return results


def _run_xgboost(data: _Data, seed: int, max_test_rows, log_fn) -> Optional[Dict]:
    try:
        import xgboost as xgb
    except ImportError:
        log_fn("    xgboost not installed -> skipped")
        return None
    Xtr, ytr, Xva, yva = data.raw_pooled()          # trees need no scaling
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=len(CLASSES), n_estimators=400, max_depth=8,
                            learning_rate=0.1, tree_method="hist", eval_metric="mlogloss",
                            early_stopping_rounds=20, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    val = metrics_from_cm(confusion(yva, clf.predict(Xva)))
    X_te, y_te = data.test(max_test_rows)
    t0 = time.perf_counter(); pred = clf.predict(X_te); secs = time.perf_counter() - t0
    test = metrics_from_cm(confusion(y_te, pred)); test["inference_ms_per_flow"] = 1000 * secs / max(1, len(y_te))
    return {"method": "xgboost", "selection": {"rule": "early stopping on validation mlogloss",
                                              "best_iteration": int(getattr(clf, "best_iteration", -1)),
                                              "val_macro_f1": val["macro_f1"]},
            "test": test, "note": "Pooled (centralized) XGBoost on the same 7-class splits; reference only."}


def run_grid(cache_dir, splits_dir, out_dir, scenarios: Iterable[str], seeds: Iterable[int],
             methods: Sequence[str] = ("fedavg", "centralized"), log_fn=print, **kw) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for sc in scenarios:
        for s in seeds:
            run_one(cache_dir, splits_dir, out_dir, sc, s, methods, log_fn=log_fn, **kw)
    summarize(out_dir, log_fn=log_fn)


# ----------------------------------------------------------------------------- summary
def collect(out_dir) -> List[Dict]:
    rows = []
    for p in sorted(Path(out_dir, "runs").glob("*/*/result.json")):
        r = json.load(open(p))
        t = r.get("test", {})
        rows.append({
            "scenario": r["scenario"], "method": r["method"], "seed": r["seed"],
            "subsampled": r.get("subsampled", False),
            "val_macro_f1": r.get("selection", {}).get("val_macro_f1"),
            "selected_at": r.get("selection", {}).get("round") or r.get("selection", {}).get("epoch"),
            "test_accuracy": t.get("accuracy"), "test_macro_f1": t.get("macro_f1"),
            "test_weighted_f1": t.get("weighted_f1"), "test_binary_fpr": t.get("binary_fpr"),
            "test_binary_fnr": t.get("binary_fnr"),
            "greeth_f1": next((c["f1"] for c in t.get("per_class", []) if c["class"] == "Mirai-greeth_flood"), None),
            "greip_f1": next((c["f1"] for c in t.get("per_class", []) if c["class"] == "Mirai-greip_flood"), None),
            "local_macro_f1_min": r.get("test_summary", {}).get("macro_f1_min"),
            "comm_mb_total": (r.get("communication", {}).get("bytes_total") or 0) / 1e6 or None,
            "raw_data_mb": (r.get("communication", {}).get("raw_data_bytes_if_centralized") or 0) / 1e6 or None,
            "wall_min": (r.get("wall_secs") or 0) / 60,
            "inference_ms_per_flow": t.get("inference_ms_per_flow"),
            "path": str(p.parent),
        })
    return rows


def summarize(out_dir, log_fn=print) -> Dict:
    import pandas as pd
    rows = collect(out_dir)
    if not rows:
        log_fn("no finished runs yet")
        return {}
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "results_all_runs.csv", index=False)
    metrics_cols = ["test_accuracy", "test_macro_f1", "test_binary_fpr", "greeth_f1", "greip_f1",
                    "val_macro_f1", "comm_mb_total", "wall_min"]
    g = df.groupby(["scenario", "method"])
    summ = g[metrics_cols].agg(["mean", "std"]).round(4)
    summ["n_seeds"] = g.size()
    summ.to_csv(Path(out_dir) / "results_summary.csv")
    lines = ["| scenario | method | seeds | test acc | test macro-F1 | binary FPR | greeth F1 | greip F1 | comm MB |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (sc, m), sub in g:
        def ms(c):
            v = sub[c].dropna()
            if v.empty:
                return "—"
            return f"{v.mean():.4f}" + (f" ± {v.std():.4f}" if len(v) > 1 else "")
        lines.append(f"| {sc} | {m} | {len(sub)} | {ms('test_accuracy')} | {ms('test_macro_f1')} | "
                     f"{ms('test_binary_fpr')} | {ms('greeth_f1')} | {ms('greip_f1')} | "
                     f"{ms('comm_mb_total') if m == 'fedavg' else '—'} |")
    if df["subsampled"].any():
        lines.append("\n**Some runs used subsampled client data (smoke/quick runs) — do not report them as results.**")
    md = "\n".join(lines)
    (Path(out_dir) / "results_summary.md").write_text(md + "\n")
    _dump(to_benchmarks(out_dir), Path(out_dir) / "benchmarks.json")
    log_fn(md)
    return {"rows": rows, "markdown": md}


_CATEGORY = {"centralized": "centralized", "local": "local_client", "xgboost": "centralized_xgboost"}


def to_benchmarks(out_dir) -> List[Dict]:
    """Convert finished runs into the JSON shape served by GET /api/v1/evaluation/benchmarks.

    One entry per (scenario, method): metrics are the mean over finished seeds; the
    confusion matrix and per-class rows are SUMMED over seeds (counts), so they stay
    internally consistent with the reported totals.
    """
    groups: Dict = {}
    for p in sorted(Path(out_dir, "runs").glob("*/*/result.json")):
        r = json.load(open(p))
        if "test" not in r:
            continue
        groups.setdefault((r["scenario"], r["method"]), []).append(r)
    out = []
    for (sc, m), rs in sorted(groups.items()):
        cm = np.sum([np.array(r["test"]["confusion_matrix"]) for r in rs], axis=0)
        agg = metrics_from_cm(cm)
        mean = lambda k: float(np.mean([r["test"].get(k, np.nan) for r in rs]))
        comm = [r.get("communication", {}).get("bytes_total") for r in rs]
        raw = [r.get("communication", {}).get("raw_data_bytes_if_centralized") for r in rs]
        cat = "federated_iid" if (m == "fedavg" and "iid" in sc) else ("federated_non_iid" if m == "fedavg" else _CATEGORY.get(m, m))
        out.append({
            "id": f"{m}-{sc}",
            "name": f"{ {'fedavg': 'FedAvg', 'centralized': 'Centralized FLMLP', 'local': 'Local-only client (median)', 'xgboost': 'XGBoost'}.get(m, m)} — {sc} ({len(rs)} seed{'s' if len(rs) > 1 else ''})",
            "category": cat,
            "modelArchitecture": "XGBoost (hist)" if m == "xgboost" else "FLMLP (39 -> 128 -> 64 -> 32 -> 7)",
            "datasetUsed": f"CIC IoT 2023 cleaned cache, split {sc}",
            "datasetName": f"CIC IoT 2023 cleaned cache, split {sc}",
            "accuracy": mean("accuracy"), "precision": mean("macro_precision"), "recall": mean("macro_recall"),
            "f1Score": mean("macro_f1"), "falsePositiveRate": mean("binary_fpr"), "falseNegativeRate": mean("binary_fnr"),
            "trainingTimeSec": float(np.mean([r.get("wall_secs", 0) for r in rs])),
            "inferenceLatencyMs": mean("inference_ms_per_flow"),
            "dataTransferredMb": (float(np.mean(comm)) / 1e6) if m == "fedavg" and all(comm) else
                                 ((float(np.mean(raw)) / 1e6) if m == "centralized" and all(raw) else 0.0),
            "clientCount": int(rs[0].get("n_clients", 1)) if m in ("fedavg", "local") else 1,
            "isBaseline": m == "centralized",
            "evaluatedAt": max(r.get("finished_at", "") for r in rs),
            "seeds": [r["seed"] for r in rs],
            "subsampled": any(r.get("subsampled") for r in rs),
            "dataSource": "experiment_runs",
            "confusionMatrix": {"classes": CLASSES, "matrix": cm.tolist(), "totalSamples": int(cm.sum())},
            "perClassPerformance": [{"className": p["class"], "precision": p["precision"], "recall": p["recall"],
                                     "f1Score": p["f1"], "support": p["support"], "falsePositiveRate": p["fpr"]}
                                    for p in agg["per_class"]],
        })
    return out
