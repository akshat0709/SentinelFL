"""Backend <-> shared sentinelfl pipeline integration tests."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import get_current_user, AuthenticatedUser
from app.services.ml_service import ml_service

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "ml" / "data" / "samples" / "cic_iot_2023_sample.csv"


def test_backend_adapter_uses_shared_adapter_and_matches_contract():
    import sentinelfl
    from app.adapters.cic_iot_2023 import CICIoT2023Adapter, CANONICAL_39_FEATURES
    assert CANONICAL_39_FEATURES == sentinelfl.FEATURES
    df = pd.read_csv(SAMPLE)
    X, labels, report = CICIoT2023Adapter().align_and_preprocess(df, scaler=None)
    shared = sentinelfl.CICIoT2023Adapter().align(df)
    np.testing.assert_allclose(X, shared.X.to_numpy(np.float32))
    assert labels == list(shared.labels)


@pytest.fixture
def federated_bundle(tmp_path):
    """A FedAvg-style bundle (log1p federated scaler) built from the sample CSV — code test only."""
    import fl_core
    from sentinelfl import Log1pStandardScaler, ModelBundle, CICIoT2023Adapter
    df = pd.read_csv(SAMPLE)
    X = CICIoT2023Adapter().align(df).X.to_numpy()
    torch.manual_seed(0)
    b = ModelBundle(fl_core.FLMLP(), Log1pStandardScaler.fit(X),
                    {"name": "test-fedavg-bundle", "kind": "federated", "adapter": "cic_iot_2023"})
    return b, b.save(tmp_path / "bundle")


def test_ml_service_serves_federated_bundle_through_adapter(federated_bundle):
    bundle, path = federated_bundle
    try:
        assert ml_service.load_bundle(str(path))
        health = ml_service.get_health()
        assert health["model_version"] == "test-fedavg-bundle"
        assert health["preprocessing"] == "log1p_standard"
        assert health["scaler_type"] == "Log1pStandardScaler"
        df = pd.read_csv(SAMPLE)
        results, _ = ml_service.predict_dataframe(df)
        expected = bundle.predict_dataframe(df)["predicted_class"]
        assert [r["predicted_class"] for r in results] == expected
    finally:
        ml_service.load_artifacts()                     # restore the legacy model for other tests
    assert ml_service.model_version == "v1.0.0-centralized-mlp"


def test_bad_bundle_path_reports_error_without_crashing(tmp_path):
    assert ml_service.load_bundle(str(tmp_path / "missing")) is False
    assert "bundle load failed" in (ml_service.load_error or "")
    ml_service.load_artifacts()
    assert ml_service.is_loaded


def test_model_health_endpoint_exposes_preprocessing():
    client = TestClient(app)
    r = client.get("/api/v1/models/health")
    assert r.status_code == 200
    assert r.json()["preprocessing"].startswith("sklearn StandardScaler")


def test_fl_service_subsampling_is_stratified():
    from sentinelfl.partition import stratified_subsample
    y = np.repeat(np.arange(7), [50000, 3000, 20000, 15000, 6000, 6000, 6000])
    idx = np.arange(len(y))
    sub = stratified_subsample(idx, y, 5000, seed=42)
    counts = np.bincount(y[sub], minlength=7)
    assert (counts > 0).all()
    assert abs(counts[0] / 5000 - 50000 / len(y)) < 0.01


def test_fl_service_job_runs_on_synthetic_cache_and_exports_servable_bundle(tmp_path, monkeypatch):
    """Exercises the real background FedAvg worker (code test; synthetic data)."""
    import time as _t
    from app.core.config import settings
    from app.schemas.federated import FLTrainRequest
    from app.services.fl_service import fl_service
    from sentinelfl.partition import iid_partition, local_split, save_splits
    rng = np.random.default_rng(0)
    centers = rng.uniform(0, 30, size=(7, 39))
    # file-ordered like the real cache: all Benign rows first
    X = np.concatenate([np.abs(rng.normal(c, 2.0, size=(400, 39))) for c in centers]).astype(np.float32)
    y = np.repeat(np.arange(7), 400).astype(np.int8)
    cache = tmp_path / "cache"; cache.mkdir(); splits = tmp_path / "splits"; splits.mkdir()
    np.save(cache / "X_clean.npy", X); np.save(cache / "y_clean.npy", y)
    allidx = np.arange(len(y)); test = rng.choice(allidx, 500, replace=False); pool = np.setdiff1d(allidx, test)
    save_splits(splits / "A_iid.npz", test, [local_split(s, y.astype(np.int64), rng) for s in iid_partition(pool, y, 10, rng)])
    monkeypatch.setattr(fl_service, "cache_dir", cache)
    monkeypatch.setattr(fl_service, "splits_dir", splits)
    monkeypatch.setattr(settings, "FL_RUNS_DIR", tmp_path / "fl_runs")
    job = fl_service.start_training_job(FLTrainRequest(scenario="A_iid", rounds=2, num_clients=3,
                                                       max_samples_per_client=200, batch_size=64))
    for _ in range(600):
        st = fl_service.get_job_status(job.job_id)
        if st.status in ("completed", "failed"):
            break
        _t.sleep(0.1)
    assert st.status == "completed", st.error
    bundle_dir = tmp_path / "fl_runs" / job.job_id / "bundle"
    assert (bundle_dir / "bundle.json").exists()
    try:
        assert ml_service.load_bundle(str(bundle_dir))
        assert ml_service.preprocessing == "log1p_standard"
    finally:
        ml_service.load_artifacts()
