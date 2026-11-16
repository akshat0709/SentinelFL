# CLAUDE.md — SentinelFL (MiniProjectBotnet)

Guidance for Claude (and humans) working in this repository. Read this before changing code,
numbers in the UI, or experiment results.

- **Repo:** https://github.com/veervishal-dubey/MiniProjectBotnet
- **Live frontend:** https://sentinelfl.vercel.app/ (Vercel, Vite SPA)
- **Research notebooks (source of the real FL results):** Google Drive,
  `MyDrive/Mini Project CSY/FL_Project/` — main notebook `FLipynb` (Colab), cells 1–8.

---

## 1. What this project is

SentinelFL is a 7-class IoT botnet / DDoS flow classifier trained with **Federated Learning
(sample-weighted FedAvg)** so that clients never share raw traffic. It has two halves:

1. **Research pipeline** (Colab + `federated/`): data cleaning, client partitioning, federated
   scaler, Flower-based FedAvg simulation, centralized baseline. This is where the real results
   come from.
2. **Web platform** (`frontend/` + `backend/`): React dashboard + FastAPI service for dataset
   upload/validation, preprocessing, single-flow and batch detection, evaluation, and a small
   in-process FedAvg demo. Auth/storage via Supabase, with an offline "Local Researcher" demo mode.

### What "adapter" means here (important)
"Adapter-based architecture" = the **software Adapter pattern for datasets**, NOT neural adapter
layers (no LoRA, bottleneck adapters, or personalized heads exist anywhere). Each dataset adapter
identifies a CSV schema, validates it, and maps it to the canonical 39-feature / 7-class contract.

**One shared adapter for every model.** The CIC IoT 2023 adapter lives in
`federated/sentinelfl/adapters.py` and is used by: the training cache builder, centralized
training, FedAvg, local-only training, XGBoost, `ModelBundle` inference, and the backend
(`backend/app/adapters/cic_iot_2023.py` keeps its API but delegates alignment + label mapping to
it). After alignment, cleaning is explicit and use-case specific:
`clean_for_training` (drop NaN rows, exact duplicates, label conflicts — identical to FLipynb
cell 3) vs `impute_for_inference` (per-column median fill, previous backend behaviour).

Backend registry (`backend/app/adapters/`):

| Adapter | File | FLMLP inference |
|---|---|---|
| CIC IoT 2023 | `cic_iot_2023.py` | Supported (needs all 39 features; no zero-imputation) |
| Edge-IIoTset | `edge_iiotset.py` | Identified, then **rejected** with a diagnostic |
| N-BaIoT / N-IoT | `n_iot.py` | Identified (115 decay-window stats), then **rejected** |

`registry.py` tries adapters in order and falls back to CIC IoT 2023 validation. Do not make
incompatible datasets "work" by imputing or aliasing features — rejection is intended behaviour.

---

## 2. Canonical contracts (do not change silently)

**Classes (index order):** `Benign, DDoS-ACK, DDoS-SYN, DDoS-TCP, Mirai-greeth_flood,
Mirai-greip_flood, Mirai-udpplain`

**39 features (exact order)** — source of truth is `federated/fl_core.py::FEATURES`, identical to
`CANONICAL_39_FEATURES` in `backend/app/adapters/cic_iot_2023.py` and `ml/artifacts/feature_columns.joblib`:

```
psh_flag_number, IAT, ARP, cwr_flag_number, Min, syn_flag_number, ece_flag_number, Max,
Tot size, DNS, fin_flag_number, ack_flag_number, Std, rst_flag_number, Telnet, Header_Length,
rst_count, Variance, HTTP, ICMP, Protocol Type, Tot sum, AVG, fin_count, IRC, IGMP, IPv,
Number, TCP, SSH, UDP, Time_To_Live, SMTP, LLC, Rate, DHCP, ack_count, syn_count, HTTPS
```
> README §4.1 lists a *different* 39-feature set (Duration, Srate, Drate, Urg Count, Magnitude…).
> That list is wrong; trust the code.

**Model `FLMLP`:** 39 → 128 → 64 → 32 → 7, ReLU, Dropout 0.2 after the first two hidden layers,
no BatchNorm (FedAvg-safe). Linear layers at `network.0/3/6/8`. **15,687 parameters**
(≈62.7 KB as float32). Defined in `federated/fl_core.py`, duplicated in
`backend/app/services/ml_service.py` — keep them identical.

`ml/artifacts/mlp_model.pth` is a dict: `model_state_dict`, `input_features` (39),
`num_classes` (7), `hidden_layers` ([128, 64, 32]). It must be loaded into `FLMLP` (with the
Dropout layers) via `checkpoint["model_state_dict"]`. A plain Linear/ReLU stack without Dropout
puts layers at indices 0/2/4/6 and fails with missing/unexpected-key errors (seen in
`Final MLP.ipynb`, cell 2).

---

## 3. Model bundles — weights always travel with their adapter + scaler

Two scalers exist and must never be mixed:

| Model | Preprocessing after the adapter |
|---|---|
| `ml/artifacts/mlp_model.pth` (legacy centralized MLP, trained before dedup) | `scaler.joblib` (sklearn `StandardScaler`, no log1p) |
| Every model from the FL pipeline (FedAvg, centralized baseline, local-only) | `log1p` then federated mean/std (`sentinelfl.Log1pStandardScaler`, = `fl_core.transform`) |

`sentinelfl.ModelBundle` packages `model.pt` + `scaler.json`/`scaler.joblib` + `bundle.json`
(adapter name, features, classes, training config, val/test metrics, provenance). Serve any
bundle with `ACTIVE_MODEL_BUNDLE=<bundle dir>`; `ml_service.load_bundle()` swaps in the bundle's
weights **and** scaler, and every prediction route goes through the same adapter. Without the
variable the legacy artifacts load (`ModelBundle.from_legacy_artifacts` wraps them identically —
verified prediction-for-prediction against the backend). Raw `.pt` files in `federated/runs/` are
NOT bundles; load them only together with `federated/preprocessing/<scenario>_scaler.json`.

---

## 4. Federated learning — how it actually works

**`federated/fl_core.py`** (shared by Colab, backend, scripts):
- `local_stats` (client) → `(n, Σz, Σz²)` of `log1p(X)`; `aggregate_stats` (server) → mean/std.
  Only aggregates leave a client. Matches pooled scaling to ~1e-7.
- `train_local`: Adam (lr 1e-3), batch 512, CrossEntropy, optimizer **recreated every call**
  (state resets each round), seeded dropout + shuffling. `class_weights` supported but unused.
- `weighted_average`: FedAvg weighted by client sample count. **All parameters are shared**;
  nothing is kept local.

**Research setup (Colab `FLipynb`, cells 3–8):**
- **The cache used by every FL run** was built from **26 CSV files** (no `Benign2.csv`):
  2,756,340 raw rows → 104 NaN/inf rows dropped → exact duplicates removed → **1,756,693 rows**
  (`federated/cache/X_clean.npy`). A later notebook run saw 27 files (2,886,172 → 1,886,247
  incl. `Benign2.csv`) but did **not** overwrite the cache (`OVERWRITE_CACHE=False`), so
  `Benign2.csv` is not in any split or result. Rebuilding the cache would change every split.
- **K = 10 clients** (not 3, not 5), each holding out 10 % local validation per class.
- Split A: stratified random 15 % test. Split B (`B_filehold_iid`): last capture file per class
  + 15 % random Benign as test (unseen captures).
- Scenarios: `A_iid`, `A_dir0.5`, `A_dir0.1` (Dirichlet per class, Benign floor 0.5,
  min 2,000 rows/client), `B_filehold_iid`. Seed 42.
- Flower (`flwr` Message API) simulation, `FedAvg(fraction_train=1.0)`, 20 rounds, 1 local epoch.
  FedAvg output verified against manual `weighted_average` (max diff 1.19e-7).
- Communication: 15,687 × 4 B = 62,748 B per client per direction per round
  → 10 clients × 2 × 20 rounds ≈ **25.1 MB per run**.

**Shared pipeline `federated/sentinelfl/`** (used for all new results):
`contract` (re-exports fl_core FEATURES/CLASSES + label aliases) · `adapters` · `scalers` ·
`model` (re-exports FLMLP/train_local/weighted_average from fl_core) · `partition`
(cell-4 partitioners, split I/O, integrity check, `stratified_subsample`) · `training`
(`run_fedavg`, `run_centralized`, `run_local_only` — same init, same train_local, best-val
selection, byte accounting) · `metrics` · `bundle` · `cache` (build/verify cache through the
adapter) · `experiments` (resumable grid runner, one-shot test eval, summaries, `benchmarks.json`).
`run_fedavg` equals the Flower FedAvg (same seeds `seed + 1000*round + client`; unit-tested
against manual `weighted_average`). Notebook cell 6 compares it with the original Flower run.

**Backend `fl_service.py`** is a pure-PyTorch demo loop for the UI. It needs
`federated/cache/X_clean.npy`, `y_clean.npy` (gitignored) and `federated/splits/*.npz`
(not in repo), so `/api/v1/federated/train` fails on a fresh clone. It now draws **seeded,
class-stratified** subsamples (default 5,000/client) instead of the first N sorted indices, and
exports each job as a servable bundle (`storage/fl_runs/<job>/bundle`). Demo jobs are subsampled:
never report them as results.

---

## 5. Verified results (only these are backed by artifacts)

All on **validation** data. The test set has **never** been evaluated yet.

The notebook was run twice with identical code and seed (26 Oct and a 27 Oct rerun); both
outputs are saved. Small differences come from CPU non-determinism, not code changes.

| Run (`federated/runs/`) | What | Round/epoch 20 or final | Best val macro-F1 |
|---|---|---|---|
| `A_iid_s42_R20_E1` (26 Oct) | FedAvg, K=10, IID, 20 rounds, E=1 | acc 0.9895, macro-F1 0.9840 | — |
| `A_iid_s42_R20_E1_20261027_074922` (rerun) | same | acc 0.9898, macro-F1 0.9845 | — |
| `CENT_A_iid_s42_E40` (26 Oct) | Centralized, same init/scaler/data, 40 epochs | final 0.8994 | 0.9940 (ep 17) |
| `CENT_A_iid_s42_E40_20261027_075903` (rerun) | same | final 0.9361 | 0.9946 (ep 31) |

Centralized training oscillates heavily (val macro-F1 ranged 0.82–0.995 across epochs), so
**always compare best-validation checkpoints**, never final epochs. Per-epoch checkpoints exist
only for the rerun (`.../epochs/model_eNNN.pt`).

FL plateaued at macro-F1 ≈ 0.88–0.89 until round 12 because of Mirai-greip ↔ Mirai-greeth
confusion, then rose. That class pair is the main source of error everywhere.

**Not run yet on real data:** the full study in `notebooks/SentinelFL_Complete_Study.ipynb`
(all scenarios × seeds 42/43/44 × fedavg/centralized/local/xgboost, one-shot test evaluation).
The code is complete and tested; results appear in `experiments/results/` once it is run.

---

## 6. Research-integrity rules (follow strictly)

1. **Never invent or "estimate" metrics.** Every number shown in the UI, README, or docs must be
   traceable to a file in `federated/runs/`, `experiments/`, or `backend/storage/evaluations/`.
2. **Test set is untouched until final evaluation.** Model selection uses validation only; then
   evaluate the chosen checkpoint **once** on test A and test B.
3. **Compare like with like:** same model, init, scaler, data, and checkpoint-selection rule for
   centralized vs federated. Don't compare FL's final round with the centralized final epoch.
4. Report mean ± std over ≥3 seeds, with per-class precision/recall/F1 and confusion matrices.
5. Keep existing runs; write new runs to new timestamped folders (the notebook already does this).

### Previously unsupported numbers (status)
| Item | Status |
|---|---|
| Hardcoded `/evaluation/benchmarks` (175,669-flow matrices, non-IID/local numbers) | **Removed.** Endpoint serves `experiments/results/benchmarks.json` or `[]`. |
| `docs/experiments.md` EXP-01…05 table | **Replaced** with the protocol; numbers come from runner output only. |
| `experiments/verified_fl_5clients_5rounds/` 100 % accuracy | Caveat README added; first-N sampling bug fixed in `fl_service`. |
| README 99.51 %, 100 %, 5 clients, wrong feature list, Vitest | **Corrected.** |
| Landing page: 3 silos, 33 features, BoT-IoT, RF/LR | **Corrected** (10 clients, 39 features, CIC IoT 2023, centralized MLP + XGBoost). |
| Design-doc 99.39 % / 98.12 % for `mlp_model.pth` | Still unreproducible (no run in repo; model predates dedup). Don't cite. |
| XGBoost 99.92 % / 79.87 % (`Untitled4.ipynb`) | **Invalid** (5 classes, mislabelled Mirai, 0 DDoS-TCP test rows). Replaced by the `xgboost` method in the runner. |
| Frontend demo data (`src/api/mockData.ts`: 3 silos, accuracies) | Not shown by API-backed pages; `getActiveExperiment` falls back to it offline. Treat as UI placeholder only. |

---

## 7. Repository map

```
backend/            FastAPI service (Python 3.11+)
  app/main.py         app, CORS, router registration, startup audit event
  app/adapters/       dataset adapters + registry (the "adapter architecture")
  app/api/routes/     health, root, datasets, preprocessing, models, federated, evaluation, logs, system
  app/services/       ml_service (inference/eval), fl_service (demo FedAvg), preprocessor,
                      validator, job_runner, audit_service
  app/core/           config.py (env + paths), security.py (Supabase JWT / local fallback)
  app/database/       supabase.py (admin client, storage, workspace access checks),
                      local_store.py (offline-mode dataset storage)
  tests/              pytest suite
  storage/            runtime: evaluations/, processed/ (gitignored), fl_runs/ (gitignored)
frontend/           React 18 + TS + Vite + Tailwind + Recharts, deployed on Vercel
  src/pages/          Landing, Login/Signup/Forgot/Reset, Dashboard, Datasets, Preprocessing,
                      Models, Federated, Detection, Evaluation, Logs, Settings
  src/lib/            backendApi.ts, supabase.ts, datasetService, modelService, csvParser, storage
  src/tests/          node --test suites (run via tsx)
federated/          fl_core.py (model/transform/FedAvg source of truth)
  sentinelfl/         shared pipeline: adapter, scaler, partition, training, metrics, bundle,
                      cache, experiments (see §4)
  tests/              pytest for sentinelfl (synthetic data = code checks only)
  runs/ preprocessing/ reports/   Colab outputs (cache/ and splits/ are NOT committed)
notebooks/          SentinelFL_Complete_Study.ipynb — Colab notebook for the full study
ml/                 artifacts/ (mlp_model.pth, scaler/label_encoder/feature_columns .joblib),
                    data/samples/ (cic_iot_2023, edge_iiotset, n_iot sample CSVs)
experiments/        results/ (runner output served by the API), verified_fl_5clients_5rounds/ (caveat)
scripts/            run_experiments.py (main CLI), run_5client_fl_experiment.py (historical),
                    verify_evaluation_metrics.py,
                    generate_dataset_samples.py, setup/*.sql (Supabase schema + RLS, run in order 01→04)
docs/               system-architecture.md, ml_fl_integration_contract.md, experiments.md
                    (features.md, problem-statement.md are empty)
iot_simulator/      placeholders only
```

---

## 8. Commands

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional; without Supabase it runs in local/demo mode
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pytest -v                       # backend tests
# Swagger: http://127.0.0.1:8000/docs   Health: /health, /health/ready

# Frontend
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
npm test                        # npx tsx --test src/tests/*.test.ts; live-API suites need backend on :8000
npm run build                   # tsc && vite build — must pass before deploying

# Shared pipeline tests + experiments
python -m pytest federated/tests -q
python scripts/run_experiments.py --help         # full study; see notebooks/ for Colab
python scripts/run_experiments.py --out <dir> --summarize-only --publish-to experiments/results

# Serve a federated model (bundle = weights + adapter + scaler)
ACTIVE_MODEL_BUNDLE=ml/bundles/fedavg-A_iid-s42 python -m uvicorn app.main:app --port 8000

# Both via Docker
docker compose up --build       # backend :8000, frontend :5173
```

**Offline mode (no Supabase).** When `VITE_SUPABASE_URL` is not set, the Datasets page uploads to
the local backend (`POST /api/v1/datasets/local-upload`), which stores files in
`backend/storage/local_datasets/<ds-local-id>/` (gitignored). `verify_dataset_access` and
`download_dataset_bytes` resolve `ds-local-*` ids / `local://` paths from that store, so validate,
preview, preprocessing, classify-dataset and evaluation work unchanged. The Local Researcher login
token (`local-researcher-session-token`, kept in `frontend/src/lib/localSession.ts`) is attached
to API calls when there is no Supabase session. Single-machine demo mode only: no multi-tenant
isolation beyond "uploader owns the dataset".

Smoke test: log in with "Enter as Local Researcher (Offline / Demo)", go to Detection, upload
`ml/data/samples/cic_iot_2023_sample.csv` (expected: classified), then `edge_iiotset_sample.csv`
(expected: rejected with a diagnostic).

---

## 9. Deployment & configuration

- **Frontend** → Vercel (`frontend/vercel.json`: Vite, `dist/`, SPA rewrite to `index.html`).
  Env vars (set in Vercel, prefix `VITE_`): `VITE_API_BASE_URL`, `VITE_SUPABASE_URL`,
  `VITE_SUPABASE_PUBLISHABLE_KEY`. If `VITE_API_BASE_URL` is unset the app calls
  `http://localhost:8000`. The live site says it runs in "isolated frontend mode", so no public
  backend is assumed — check the Vercel env before relying on API features in production.
- **Backend** → not deployed publicly as far as the repo shows; `backend/Dockerfile` +
  `docker-compose.yml` exist. Env: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (backend only,
  never in frontend), `SUPABASE_ANON_KEY`, `STORAGE_BUCKET_NAME=sentinelfl-datasets`,
  `FRONTEND_ORIGINS` (add `https://sentinelfl.vercel.app` when a backend is hosted),
  `ENVIRONMENT`, `HOST`, `PORT`, optional `ML_ARTIFACTS_DIR`, `FEDERATED_*_DIR`,
  `EXPERIMENT_RESULTS_DIR` (default `experiments/results`), `ACTIVE_MODEL_BUNDLE` (bundle dir).
  The Docker image now also copies `experiments/`.
- CORS also allows localhost / 127.0.0.1 / private LAN IPs by regex.
- Never commit `.env` files or the service-role key.

---

## 10. Conventions for changes

- Keep `fl_core.py` as the single source of truth for model, transform, and FedAvg; if you change
  it, mirror `ml_service.FLMLP` and re-verify that checkpoint keys/shapes still load (`strict=True`).
- New datasets: add the mapping in `federated/sentinelfl/adapters.py` (shared by training and
  serving), a thin backend wrapper in `backend/app/adapters/` + `registry.py`, tests in both
  `federated/tests/` and `backend/tests/test_dataset_adapters.py`, and a sample CSV in
  `ml/data/samples/`. Never zero-impute or alias missing features.
- Train/serve new models through `sentinelfl` and ship them as `ModelBundle`s.
- Don't overwrite files in `federated/runs/` or `ml/artifacts/`; add new versions alongside.
- UI metrics should come from API responses backed by saved artifacts, not constants in TSX/py.
- Run `pytest` (backend), `python -m pytest federated/tests`, and `npm run build` before committing.
  Baseline: backend 74 pass + 2 env-dependent failures (need Supabase / `federated/splits/`);
  federated 14 pass; frontend 47 pass with the backend running.

## 11. Open work (priority order)

1. Run `notebooks/SentinelFL_Complete_Study.ipynb` on Colab (phase 1 seed 42 ≈ 3 h CPU, resumable;
   then seeds 43/44), publish `experiments/results/` and one FedAvg bundle to `ml/bundles/`.
2. Commit `federated/splits/*.npz` (~10 MB) so the backend FL demo works from a clean clone.
3. Decide whether to keep `Benign2.csv` out (current cache) or rebuild cache + splits with it
   (then all runs must be redone); retire or retrain `mlp_model.pth` on the deduplicated data.
4. Host the backend (e.g. Render/Railway with the Dockerfile) and set `VITE_API_BASE_URL` on
   Vercel; add `https://sentinelfl.vercel.app` to `FRONTEND_ORIGINS`.
5. Optional research extensions: class-weighted loss for the greeth/greip pair, partial client
   participation, FedProx for α = 0.1.
