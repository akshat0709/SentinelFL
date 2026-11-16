"""Federated training on the selected uploaded dataset (instead of the benchmark cache)."""
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.fl_service import fl_service
from sentinelfl.contract import FEATURES

AUTH = {"Authorization": "Bearer local-researcher-token"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(settings, "PROCESSED_CACHE_DIR", tmp_path / "processed")
    monkeypatch.setattr(settings, "FL_RUNS_DIR", tmp_path / "fl_runs")
    (tmp_path / "processed").mkdir()
    monkeypatch.setattr(settings, "SUPABASE_URL", "")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "")
    monkeypatch.setattr(settings, "SUPABASE_ANON_KEY", "")
    fl_service.active_job = None
    return TestClient(app)


def _csv(tmp_path, n, labels):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n, len(FEATURES))) * 100, columns=FEATURES)
    df["label"] = [labels[i % len(labels)] for i in range(n)]
    p = tmp_path / f"d{n}_{len(labels)}.csv"
    df.to_csv(p, index=False)
    return p


def _upload(client, path):
    with open(path, "rb") as f:
        r = client.post("/api/v1/datasets/local-upload", headers=AUTH, files={"file": (path.name, f, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _wait(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/v1/federated/jobs/{job_id}").json()
        if j["status"] in ("completed", "failed"):
            return j
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_fl_trains_on_selected_dataset(client, tmp_path):
    ds = _upload(client, _csv(tmp_path, 600, ["BenignTraffic", "DDoS-SYN_Flood", "Mirai-udpplain"]))
    r = client.post("/api/v1/federated/train", headers=AUTH,
                    json={"dataset_id": ds, "rounds": 2, "num_clients": 5, "batch_size": 64})
    assert r.status_code == 200, r.text
    src = r.json()["data_source"]
    assert src["kind"] == "uploaded_dataset" and src["dataset_id"] == ds
    assert src["rows_used"] == 600 and set(src["classes_present"]) == {"Benign", "DDoS-SYN", "Mirai-udpplain"}
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "completed", job.get("error")
    assert len(job["history"]) == 2 and job["data_source"]["dataset_id"] == ds
    det = client.get(f"/api/v1/federated/runs/{job['job_id']}").json()
    assert det["data_source"]["dataset_name"] and sum(c["sampleCount"] for c in det["clients"]) < 600


def test_fl_rejects_single_class_or_tiny_datasets_with_reason(client, tmp_path):
    one = _upload(client, _csv(tmp_path, 300, ["DDoS-ACK_Fragmentation"]))
    r = client.post("/api/v1/federated/train", headers=AUTH, json={"dataset_id": one})
    assert r.status_code == 400 and "only 1 usable class" in r.json()["detail"]

    tiny = _upload(client, _csv(tmp_path, 40, ["BenignTraffic", "DDoS-SYN_Flood"]))
    r = client.post("/api/v1/federated/train", headers=AUTH, json={"dataset_id": tiny})
    assert r.status_code == 400 and "needs at least" in r.json()["detail"]


def test_fl_dataset_requires_auth(client):
    r = client.post("/api/v1/federated/train", json={"dataset_id": "ds-local-x"})
    assert r.status_code == 401
