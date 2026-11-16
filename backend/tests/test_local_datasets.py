"""Offline (no-Supabase) dataset mode: upload to the backend's disk and use everywhere."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "ml" / "data" / "samples" / "cic_iot_2023_sample.csv"
AUTH = {"Authorization": "Bearer local-researcher-token"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(settings, "PROCESSED_CACHE_DIR", tmp_path / "processed")
    (tmp_path / "processed").mkdir()
    monkeypatch.setattr(settings, "SUPABASE_URL", "")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "")
    monkeypatch.setattr(settings, "SUPABASE_ANON_KEY", "")
    return TestClient(app)


def _upload(client, path=SAMPLE):
    with open(path, "rb") as f:
        return client.post("/api/v1/datasets/local-upload", headers=AUTH,
                           files={"file": (path.name, f, "text/csv")})


def test_local_upload_and_full_flow_without_supabase(client):
    r = _upload(client)
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["id"].startswith("ds-local-") and rec["row_count"] == 21 and rec["column_count"] == 40
    ds = rec["id"]

    assert [d["id"] for d in client.get("/api/v1/datasets/local", headers=AUTH).json()] == [ds]

    v = client.post(f"/api/v1/datasets/{ds}/validate", headers=AUTH)
    assert v.status_code == 200, v.text

    p = client.get(f"/api/v1/datasets/{ds}/preview", headers=AUTH)
    assert p.status_code == 200, p.text

    c = client.post(f"/api/v1/models/classify-dataset/{ds}", headers=AUTH)
    assert c.status_code == 200, c.text
    assert c.json()["total_flows"] == 21

    job = client.post("/api/v1/preprocessing/jobs", headers=AUTH, json={
        "dataset_id": ds, "target_column": "label", "drop_duplicates": True, "scaling_method": "standard",
        "categorical_encoding": "onehot", "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15})
    assert job.status_code == 200, job.text
    assert job.json()["status"] == "completed"

    d = client.delete(f"/api/v1/datasets/local/{ds}", headers=AUTH)
    assert d.status_code == 200
    assert client.get("/api/v1/datasets/local", headers=AUTH).json() == []
    assert client.post(f"/api/v1/datasets/{ds}/validate", headers=AUTH).status_code == 404


def test_local_upload_rejects_bad_files(client, tmp_path):
    bad = tmp_path / "x.exe"; bad.write_bytes(b"MZ")
    assert _upload(client, bad).status_code == 400
    empty = tmp_path / "e.csv"; empty.write_bytes(b"")
    assert _upload(client, empty).status_code == 400
    with open(SAMPLE, "rb") as f:
        assert client.post("/api/v1/datasets/local-upload",
                           files={"file": ("s.csv", f, "text/csv")}).status_code == 401


def test_local_dataset_owned_by_uploader_only(client, monkeypatch):
    ds = _upload(client).json()["id"]
    from app.database import local_store
    rec = local_store.get(ds); rec["uploaded_by"] = "someone-else"
    (local_store.root() / ds / "meta.json").write_text(__import__("json").dumps(rec))
    assert client.post(f"/api/v1/datasets/{ds}/validate", headers=AUTH).status_code == 403
    assert client.get("/api/v1/datasets/local", headers=AUTH).json() == []


def test_local_store_rejects_path_traversal():
    from app.database import local_store
    assert local_store.get("ds-local-../../etc") is None
    with pytest.raises((ValueError, FileNotFoundError)):
        local_store.read_bytes("local://ds-local-abc/../../../etc/passwd")
