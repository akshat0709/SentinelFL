"""Local (offline) dataset store — used when Supabase is not configured.

Uploaded files are kept on the backend's disk under ``storage/local_datasets/<id>/`` with a
``meta.json`` that mirrors the columns of the Supabase ``public.datasets`` row, so every
existing route (validate, preview, preprocessing, classify, evaluate) works unchanged:

    verify_dataset_access(...)  -> returns the local record for ``ds-local-*`` ids
    download_dataset_bytes(...) -> reads ``local://<id>/<file>`` paths from disk

This is a single-machine development / demo mode: there is no multi-tenant isolation beyond
"the uploader owns the dataset". Use Supabase for shared or production deployments.
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings

LOCAL_PREFIX = "ds-local-"
STORAGE_SCHEME = "local://"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def root() -> Path:
    d = settings.STORAGE_DIR / "local_datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_local_id(dataset_id: str) -> bool:
    return isinstance(dataset_id, str) and dataset_id.startswith(LOCAL_PREFIX)


def is_local_path(storage_path: Optional[str]) -> bool:
    return isinstance(storage_path, str) and storage_path.startswith(STORAGE_SCHEME)


def _dir_for(dataset_id: str) -> Path:
    if not is_local_id(dataset_id) or "/" in dataset_id or ".." in dataset_id:
        raise ValueError("invalid local dataset id")
    return root() / dataset_id


def _count_rows_cols(content: bytes) -> Dict[str, int]:
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
        return {"row_count": int(len(df)), "column_count": int(df.shape[1])}
    except Exception:
        lines = content.decode("utf-8", errors="replace").splitlines()
        cols = len(lines[0].split(",")) if lines else 0
        return {"row_count": max(0, len(lines) - 1), "column_count": cols}


def save(content: bytes, filename: str, user_id: str) -> Dict[str, Any]:
    dataset_id = f"{LOCAL_PREFIX}{uuid.uuid4().hex[:12]}"
    safe = _SAFE_NAME.sub("_", Path(filename or "dataset.csv").name) or "dataset.csv"
    d = _dir_for(dataset_id)
    d.mkdir(parents=True, exist_ok=False)
    (d / safe).write_bytes(content)
    record = {
        "id": dataset_id,
        "name": Path(safe).stem,
        "filename": safe,
        "storage_path": f"{STORAGE_SCHEME}{dataset_id}/{safe}",
        "workspace_id": "local-workspace",
        "uploaded_by": user_id,
        "file_size_bytes": len(content),
        "status": "ready",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "storage_backend": "local",
        **_count_rows_cols(content),
    }
    (d / "meta.json").write_text(json.dumps(record, indent=2))
    return record


def get(dataset_id: str) -> Optional[Dict[str, Any]]:
    if not is_local_id(dataset_id):
        return None
    try:
        meta = _dir_for(dataset_id) / "meta.json"
    except ValueError:
        return None
    if not meta.exists():
        return None
    return json.loads(meta.read_text())


def list_for(user_id: str) -> List[Dict[str, Any]]:
    out = []
    for meta in sorted(root().glob(f"{LOCAL_PREFIX}*/meta.json"), reverse=True):
        try:
            rec = json.loads(meta.read_text())
        except ValueError:
            continue
        if rec.get("uploaded_by") == user_id:
            out.append(rec)
    return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)


def read_bytes(storage_path: str) -> bytes:
    rel = storage_path[len(STORAGE_SCHEME):]
    dataset_id, _, fname = rel.partition("/")
    d = _dir_for(dataset_id)
    p = (d / fname).resolve()
    if d.resolve() not in p.parents or not p.exists():
        raise FileNotFoundError(storage_path)
    return p.read_bytes()


def delete(dataset_id: str) -> bool:
    try:
        d = _dir_for(dataset_id)
    except ValueError:
        return False
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True
