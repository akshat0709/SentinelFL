"""Tests for the shared sentinelfl pipeline.

Synthetic data here is ONLY for checking code correctness (shapes, equivalences,
determinism, resumability). It says nothing about real detection performance.

Run from the repo root:  python -m pytest federated/tests -q
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "federated"))

import fl_core  # noqa: E402
import sentinelfl  # noqa: E402
from sentinelfl import (CLASSES, FEATURES, CICIoT2023Adapter, DatasetCompatibilityError,  # noqa: E402
                        Log1pStandardScaler, ModelBundle, clean_for_training, impute_for_inference)
from sentinelfl.cache import build_cache, verify_cache  # noqa: E402
from sentinelfl.experiments import collect, run_one, summarize, to_benchmarks  # noqa: E402
from sentinelfl.metrics import metrics  # noqa: E402
from sentinelfl.partition import (dirichlet_partition, iid_partition, load_splits, local_split,  # noqa: E402
                                  save_splits, check_integrity, stratified_subsample)
from sentinelfl.training import ClientData, run_fedavg  # noqa: E402

SAMPLE_CSV = REPO / "ml" / "data" / "samples" / "cic_iot_2023_sample.csv"
ARTIFACTS = REPO / "ml" / "artifacts"


def synthetic(n_per_class=400, seed=0):
    """Separable, non-negative, CIC-shaped synthetic flows (code tests only)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 50, size=(len(CLASSES), len(FEATURES)))
    X = np.concatenate([np.abs(rng.normal(c, 3.0, size=(n_per_class, len(FEATURES)))) for c in centers])
    y = np.repeat(np.arange(len(CLASSES)), n_per_class)
    return X.astype(np.float32), y.astype(np.int64)


# ------------------------------------------------------------------ contract / adapter
def test_contract_matches_fl_core_and_backend():
    assert FEATURES == fl_core.FEATURES and CLASSES == fl_core.CLASSES
    sys.path.insert(0, str(REPO / "backend"))
    from app.adapters.cic_iot_2023 import CANONICAL_39_FEATURES, CANONICAL_7_CLASSES
    assert list(CANONICAL_39_FEATURES) == FEATURES and list(CANONICAL_7_CLASSES) == CLASSES


def test_adapter_aligns_sample_csv_case_insensitively_and_maps_labels():
    df = pd.read_csv(SAMPLE_CSV)
    shuffled = df[list(reversed(df.columns))].rename(columns={"Tot size": "tot SIZE", "label": "Label"})
    shuffled.loc[0, "Label"] = "DDoS-SYN_Flood"          # raw CIC name
    shuffled.loc[1, "Label"] = "SomethingElse"           # unknown
    a = CICIoT2023Adapter().align(shuffled)
    assert list(a.X.columns) == FEATURES and a.X.shape == (len(df), 39)
    assert a.labels[0] == "DDoS-SYN" and a.labels[1] is None
    assert a.report["unknown_label_rows"] == 1
    assert a.label_ids()[1] == -1


def test_adapter_rejects_missing_features_instead_of_imputing():
    df = pd.read_csv(SAMPLE_CSV).drop(columns=["IAT", "Rate"])
    with pytest.raises(DatasetCompatibilityError):
        CICIoT2023Adapter().align(df)


def test_inf_becomes_nan_then_inference_imputation_fills():
    df = pd.read_csv(SAMPLE_CSV)
    df.loc[0, "Rate"] = np.inf
    a = CICIoT2023Adapter().align(df)
    assert np.isnan(a.X.loc[0, "Rate"]) and a.report["non_finite_cells"] == 1
    filled, n = impute_for_inference(a.X)
    assert n == 1 and np.isfinite(filled.to_numpy()).all()


def test_clean_for_training_drops_nan_duplicates_and_conflicts():
    X, y = synthetic(5)
    X = np.concatenate([X, X[:1], X[1:2], X[2:3]])        # dup, dup, conflict copy
    y = np.concatenate([y, y[:1], y[1:2], [(y[2] + 1) % 7]])
    X[3, 0] = np.nan
    Xc, yc, _, st = clean_for_training(X, y)
    assert st["nan_inf_rows_removed"] == 1
    assert st["duplicate_rows_removed"] == 2
    assert st["label_conflict_rows_removed"] == 2           # both rows of the conflicting vector
    assert len(yc) == len(y) - 5


# ------------------------------------------------------------------ scaler == fl_core
def test_log1p_scaler_equals_fl_core_and_federated_equals_pooled():
    X, _ = synthetic(50)
    parts = np.array_split(X, 4)
    fed = Log1pStandardScaler.from_client_stats([sentinelfl.local_stats(p) for p in parts])
    m, s, n = fl_core.aggregate_stats([fl_core.local_stats(p) for p in parts])
    np.testing.assert_allclose(fed.mean_, m, rtol=1e-6); np.testing.assert_allclose(fed.scale_, s, rtol=1e-6)
    np.testing.assert_allclose(fed.transform(X), fl_core.transform(X, m, s), rtol=1e-5, atol=1e-5)
    pooled = Log1pStandardScaler.fit(X)
    np.testing.assert_allclose(pooled.mean_, fed.mean_, rtol=1e-5)
    # DataFrame input in any column order gives the same result
    df = pd.DataFrame(X, columns=FEATURES)[list(reversed(FEATURES))]
    np.testing.assert_allclose(fed.transform(df), fed.transform(X))


def test_existing_colab_scaler_json_loads():
    sc = Log1pStandardScaler.load(REPO / "federated" / "preprocessing" / "A_iid_scaler.json")
    assert sc.mean_.shape == (39,) and sc.n_samples_seen_ == 1343859


# ------------------------------------------------------------------ bundles
def test_bundle_roundtrip_predictions_identical(tmp_path):
    X, y = synthetic(30)
    sc = Log1pStandardScaler.fit(X)
    torch.manual_seed(0)
    b = ModelBundle(fl_core.FLMLP(), sc, {"name": "t", "kind": "federated", "adapter": "cic_iot_2023"})
    b.save(tmp_path / "b")
    b2 = ModelBundle.load(tmp_path / "b")
    df = pd.DataFrame(X, columns=FEATURES)
    np.testing.assert_allclose(b.predict_dataframe(df)["probabilities"], b2.predict_dataframe(df)["probabilities"])
    assert json.load(open(tmp_path / "b" / "bundle.json"))["preprocessing"]["type"] == "log1p_standard"


@pytest.mark.skipif(not (ARTIFACTS / "mlp_model.pth").exists(), reason="legacy artifacts not present")
def test_legacy_bundle_reproduces_backend_predictions():
    sys.path.insert(0, str(REPO / "backend"))
    from app.services.ml_service import ml_service
    assert ml_service.is_loaded, ml_service.load_error
    df = pd.read_csv(SAMPLE_CSV)
    backend_results, _ = ml_service.predict_dataframe(df)
    legacy = ModelBundle.from_legacy_artifacts(ARTIFACTS)
    ours = legacy.predict_dataframe(df)
    assert ours["predicted_class"] == [r["predicted_class"] for r in backend_results]
    for r, p in zip(backend_results, ours["probabilities"]):
        np.testing.assert_allclose([r["probabilities"][c] for c in CLASSES], p, atol=1e-4)


# ------------------------------------------------------------------ partition / sampling
def test_stratified_subsample_keeps_all_classes_unlike_first_n():
    y = np.repeat(np.arange(7), [5000, 300, 2000, 1500, 600, 600, 600])   # file-ordered, Benign first
    idx = np.arange(len(y))
    first = idx[:500]
    assert len(np.unique(y[first])) == 1                                     # the old bug
    sub = stratified_subsample(idx, y, 500, seed=1)
    assert len(sub) == 500 and len(np.unique(y[sub])) == 7
    np.testing.assert_array_equal(sub, stratified_subsample(idx, y, 500, seed=1))   # deterministic


def test_partitions_are_disjoint_and_complete():
    _, y = synthetic(300)
    rng = np.random.default_rng(42)
    pool = np.arange(len(y))[100:]
    test = np.arange(100)
    for shards in (iid_partition(pool, y, 5, rng), dirichlet_partition(pool, y, 5, 0.5, rng, min_client=50)[0]):
        clients = [local_split(s, y, rng) for s in shards]
        chk = check_integrity({"test": test, "clients": clients}, len(y))
        assert chk["disjoint"] and chk["complete"]


# ------------------------------------------------------------------ FedAvg
def test_fedavg_one_round_equals_manual_weighted_average():
    X, y = synthetic(60)
    sc = Log1pStandardScaler.fit(X); Xs = sc.transform(X)
    rng = np.random.default_rng(0); perm = rng.permutation(len(y)); parts = np.array_split(perm, 3)
    clients = [ClientData(Xs[p[:-20]], y[p[:-20]], Xs[p[-20:]], y[p[-20:]]) for p in parts]
    fl_core.set_seed(7); init = fl_core.FLMLP().state_dict()
    res = run_fedavg(clients, init, rounds=1, local_epochs=1, batch_size=64, seed=7, log_fn=None)
    locals_ = []
    for k, c in enumerate(clients):
        m = fl_core.FLMLP(); m.load_state_dict(init)
        fl_core.train_local(m, c.X_train, c.y_train, epochs=1, batch_size=64, lr=1e-3, seed=7 + 1000 + k)
        locals_.append(m.state_dict())
    manual = fl_core.weighted_average(locals_, [c.n_train for c in clients])
    assert max((manual[k] - res["final_state"][k]).abs().max().item() for k in manual) < 1e-6
    assert res["history"][0]["bytes_round"] == 2 * 3 * 15687 * 4


def test_metrics_match_sklearn():
    from sklearn.metrics import accuracy_score, f1_score
    rng = np.random.default_rng(3)
    yt = rng.integers(0, 7, 500); yp = np.where(rng.random(500) < 0.7, yt, rng.integers(0, 7, 500))
    m = metrics(yt, yp)
    assert abs(m["accuracy"] - accuracy_score(yt, yp)) < 1e-12
    assert abs(m["macro_f1"] - f1_score(yt, yp, average="macro")) < 1e-12


# ------------------------------------------------------------------ cache + runner end-to-end
@pytest.fixture
def synthetic_project(tmp_path):
    """CSV files -> adapter cache -> splits; mirrors the real Drive layout at toy size."""
    X, y = synthetic(250, seed=1)
    data_dir = tmp_path / "CICBOTNET"; data_dir.mkdir()
    for c in range(7):                      # two capture files per class, raw CIC-style labels
        rows = np.where(y == c)[0]
        for j, part in enumerate(np.array_split(rows, 2)):
            df = pd.DataFrame(X[part], columns=FEATURES)
            df["Label"] = CLASSES[c] if c else "BenignTraffic"
            df.to_csv(data_dir / f"{CLASSES[c]}_{j}.csv", index=False)
    pd.DataFrame(X[:3], columns=FEATURES).assign(Label="Benign").to_csv(data_dir / "Benign_dups.csv", index=False)
    cache = tmp_path / "cache"
    meta = build_cache(str(data_dir), str(cache), log_fn=None)
    assert meta["stats"]["duplicate_rows_removed"] == 3
    with pytest.raises(FileExistsError):
        build_cache(str(data_dir), str(cache), log_fn=None)
    assert verify_cache(str(data_dir), str(cache), log_fn=None)["ok"]
    yc = np.load(cache / "y_clean.npy").astype(np.int64)
    rng = np.random.default_rng(42)
    allidx = np.arange(len(yc)); test = rng.choice(allidx, 300, replace=False)
    pool = np.setdiff1d(allidx, test)
    splits = tmp_path / "splits"; splits.mkdir()
    save_splits(splits / "A_iid.npz", test, [local_split(s, yc, rng) for s in iid_partition(pool, yc, 4, rng)])
    shards, _ = dirichlet_partition(pool, yc, 4, 0.1, rng, min_client=50)
    save_splits(splits / "A_dir0.1.npz", test, [local_split(s, yc, rng) for s in shards])
    return cache, splits, tmp_path / "out"


def test_runner_end_to_end_resumable_and_benchmarks(synthetic_project):
    cache, splits, out = synthetic_project
    kw = dict(rounds=3, local_epochs=1, batch_size=64, log_fn=lambda *_: None)
    res = run_one(cache, splits, out, "A_iid", 42, ("fedavg", "centralized", "local"), **kw)
    for m in ("fedavg", "centralized", "local"):
        assert "test" in res[m] and 0.0 <= res[m]["test"]["macro_f1"] <= 1.0
    assert res["fedavg"]["selection"]["round"] in (1, 2, 3)
    assert res["centralized"]["epochs"] == 3                    # same number of passes as FL
    assert res["fedavg"]["communication"]["bytes_total"] == 3 * 2 * 4 * 15687 * 4
    # selected bundle reproduces the stored test metrics through the adapter path
    b = ModelBundle.load(Path(out) / "runs" / "A_iid_s42" / "fedavg" / "bundle")
    split = load_splits(splits / "A_iid.npz")
    Xc = np.load(cache / "X_clean.npy"); yc = np.load(cache / "y_clean.npy").astype(np.int64)
    df = pd.DataFrame(Xc[split["test"]], columns=FEATURES)
    pred = np.array(b.predict_dataframe(df)["class_index"])
    assert abs((pred == yc[split["test"]]).mean() - res["fedavg"]["test"]["accuracy"]) < 1e-9
    # resume: nothing re-runs
    mtime = os.path.getmtime(out / "runs" / "A_iid_s42" / "fedavg" / "result.json")
    run_one(cache, splits, out, "A_iid", 42, ("fedavg", "centralized"), **kw)
    assert os.path.getmtime(out / "runs" / "A_iid_s42" / "fedavg" / "result.json") == mtime
    run_one(cache, splits, out, "A_dir0.1", 42, ("fedavg",), **kw)
    summarize(out, log_fn=lambda *_: None)
    bm = json.load(open(out / "benchmarks.json"))
    assert {b["id"] for b in bm} == {"fedavg-A_iid", "centralized-A_iid", "local-A_iid", "fedavg-A_dir0.1"}
    for e in bm:
        assert e["dataSource"] == "experiment_runs"
        assert sum(map(sum, e["confusionMatrix"]["matrix"])) == e["confusionMatrix"]["totalSamples"]
    assert (out / "results_summary.md").exists() and len(collect(out)) == 4
