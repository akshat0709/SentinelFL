# CLAUDE.md — SentinelFL (MiniProjectBotnet)

Guidance for Claude (and humans) working in this repository. Read this before changing code,
numbers in the UI, or experiment results. Section 12 is a log of everything that broke during
development and how it was fixed — check it before debugging something that "used to work".

- **Upstream repo:** https://github.com/veervishal-dubey/MiniProjectBotnet
- **Working repo (deploy from here):** https://github.com/akshat0709/SentinelFL
  (local clone: `~/Desktop/SentinelFL`; remote `origin`. As of 16 Nov 2026 it is still **empty** —
  every push attempt failed; see §12.)
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
   upload/validation, preprocessing, single-flow and batch detection with a dataset-level
   **attack verdict**, evaluation, and in-process FedAvg training (on the benchmark cache or on an
   uploaded dataset). Auth/storage via Supabase, with an offline "Local Researcher" mode.

### What "adapter" means here (important)
"Adapter-based architecture" = the **software Adapter pattern for datasets**, NOT neural adapter
layers (no LoRA, bottleneck adapters, or personalized heads exist anywhere). Each dataset adapter
identifies a CSV schema, validates it, and maps it to the canonical 39-feature / 7-class contract.

**One shared adapter for every model.** The CIC IoT 2023 adapter lives in
`federated/sentinelfl/adapters.py` and is used by: the training cache builder, centralized
training, FedAvg (Colab and backend), local-only training, XGBoost, `ModelBundle` inference, and
the backend (`backend/app/adapters/cic_iot_2023.py` keeps its API but delegates alignment + label
mapping to it). After alignment, cleaning is explicit and use-case specific:
`clean_for_training` (drop NaN rows, exact duplicates, label conflicts — identical to FLipynb
cell 3) vs `impute_for_inference` (per-column median fill).

Backend registry (`backend/app/adapters/`):

| Adapter | File | FLMLP inference |
|---|---|---|
| CIC IoT 2023 | `cic_iot_2023.py` | Supported (needs all 39 features) |
| Edge-IIoTset | `edge_iiotset.py` | Identified, then **rejected** with a diagnostic |
| N-BaIoT / N-IoT | `n_iot.py` | Identified (115 decay-window stats), then **rejected** |

Rejection is intended behaviour: those datasets lack the 39 features (N-BaIoT shares none) and
use attack classes the model was never trained on (`mirai_scan`, `bashlite_combo`, `Ransomware`,
`DDoS_UDP`, …). Do not make them "work" by imputing or aliasing features.

**Only exception — explicit "Run anyway"** (`force_incompatible=true` on `/models/predict-csv`
and `/models/classify-dataset`, button on Detection and Evaluation after a rejection):
missing features are filled with training means (`ml_service.neutral_fill_values()`, which scale
to exactly 0); the response carries `forced_compatibility`, `imputed_features`,
`compatibility_warning`; a forced evaluation scores only rows whose label is *exactly* one of the
7 classes (strict `normalize_label`) and is marked `is_valid: false`. Nothing forced is cached,
saved to evaluation history, exported, persisted in sessionStorage, or used for training; each
forced run is audited as a WARN event. Expect meaningless output (e.g. every Edge-IIoTset row →
DDoS-SYN).

---

## 2. Canonical contracts (do not change silently)

**Classes (index order):** `Benign, DDoS-ACK, DDoS-SYN, DDoS-TCP, Mirai-greeth_flood,
Mirai-greip_flood, Mirai-udpplain`

**39 features (exact order)** — source of truth is `federated/fl_core.py::FEATURES`, identical to
`CANONICAL_39_FEATURES` in `backend/app/adapters/cic_iot_2023.py`,
`ml/artifacts/feature_columns.joblib` and `federated/cache/clean_meta.json["features"]`:

```
psh_flag_number, IAT, ARP, cwr_flag_number, Min, syn_flag_number, ece_flag_number, Max,
Tot size, DNS, fin_flag_number, ack_flag_number, Std, rst_flag_number, Telnet, Header_Length,
rst_count, Variance, HTTP, ICMP, Protocol Type, Tot sum, AVG, fin_count, IRC, IGMP, IPv,
Number, TCP, SSH, UDP, Time_To_Live, SMTP, LLC, Rate, DHCP, ack_count, syn_count, HTTPS
```
> README §4.1 once listed a different feature set (Duration, Srate, Drate…). Trust the code.

**Model `FLMLP`:** 39 → 128 → 64 → 32 → 7, ReLU, Dropout 0.2 after the first two hidden layers,
no BatchNorm (FedAvg-safe). Linear layers at `network.0/3/6/8`. **15,687 parameters**
(≈62.7 KB as float32). Defined in `federated/fl_core.py`, duplicated in
`backend/app/services/ml_service.py` — keep them identical.

`ml/artifacts/mlp_model.pth` is a dict: `model_state_dict`, `input_features` (39),
`num_classes` (7), `hidden_layers` ([128, 64, 32]). Load it into `FLMLP` (with the Dropout
layers) via `checkpoint["model_state_dict"]`; a plain Linear/ReLU stack puts layers at 0/2/4/6
and fails with missing/unexpected keys (seen in `Final MLP.ipynb`, cell 2).

---

## 3. Model bundles — weights always travel with their adapter + scaler

Two scalers exist and must never be mixed:

| Model | Preprocessing after the adapter |
|---|---|
| `ml/artifacts/mlp_model.pth` (legacy centralized MLP, trained before dedup) | `scaler.joblib` (sklearn `StandardScaler`, no log1p) |
| Every model from the FL pipeline (FedAvg, centralized baseline, local-only, backend FL jobs) | `log1p` then federated mean/std (`sentinelfl.Log1pStandardScaler`, = `fl_core.transform`) |

`sentinelfl.ModelBundle` packages `model.pt` + `scaler.json`/`scaler.joblib` + `bundle.json`
(adapter, features, classes, training config, data source, metrics, provenance). Serve any bundle
with `ACTIVE_MODEL_BUNDLE=<bundle dir>`; `ml_service.load_bundle()` swaps weights **and** scaler.
Without it the legacy artifacts load (`ModelBundle.from_legacy_artifacts`, verified
prediction-for-prediction). Raw `.pt` files in `federated/runs/` are NOT bundles; load them only
with `federated/preprocessing/<scenario>_scaler.json`. Every backend FL job exports a bundle to
`backend/storage/fl_runs/<job>/bundle`.

---

## 4. Federated learning — how it actually works

**`federated/fl_core.py`** (shared by Colab, backend, scripts):
- `local_stats` (client) → `(n, Σz, Σz²)` of `log1p(X)`; `aggregate_stats` (server) → mean/std.
  Only aggregates leave a client.
- `train_local`: Adam (lr 1e-3), batch 512, CrossEntropy, optimizer **recreated every call**,
  seeded dropout + shuffling. `class_weights` supported but unused.
- `weighted_average`: FedAvg weighted by client sample count. **All parameters are shared.**

**Research setup (Colab `FLipynb`, cells 3–8):**
- Cache built from **26 CSV files** (no `Benign2.csv`): 2,756,340 raw rows → **1,756,693 rows**
  after NaN/duplicate/label-conflict removal (`federated/cache/X_clean.npy`, 274 MB; class
  counts in `clean_meta.json`). A later run saw 27 files but did not overwrite the cache.
- **K = 10 clients**, 10 % local validation per class. Split A: stratified 15 % test. Split B
  (`B_filehold_iid`): last capture file per class + 15 % Benign as test.
- Scenarios `A_iid`, `A_dir0.5`, `A_dir0.1` (Dirichlet per class, Benign floor 0.5, ≥ 2,000
  rows/client), `B_filehold_iid`. Seed 42. Flower FedAvg, 20 rounds, 1 local epoch.
- Communication: 62,748 B per client per direction per round → ≈ **25.1 MB per run**.

**Shared pipeline `federated/sentinelfl/`:** `contract` · `adapters` · `scalers` · `model` ·
`partition` (partitioners, split I/O, integrity check, `stratified_subsample`) · `training`
(`run_fedavg`, `run_centralized`, `run_local_only`) · `metrics` · `bundle` · `cache` ·
`experiments` (resumable grid runner, one-shot test eval, `benchmarks.json`).

**Backend `fl_service.py`** (`POST /api/v1/federated/train`) has two data sources:
1. **Benchmark cache** (default): needs `federated/cache/X_clean.npy`, `y_clean.npy`,
   `clean_meta.json` (gitignored, 274 MB, download from Drive `FL_Project/cache`) and
   `federated/splits/*.npz` (committed, ~10 MB). Seeded, class-stratified subsamples
   (UI sends 3,000/client). Not available on a hosted backend unless the cache is uploaded there.
2. **Uploaded dataset** (`dataset_id`, auth required; FL page "Train on: <active dataset>"):
   shared adapter → `clean_for_training` → stratified 20 % test → IID (or Dirichlet for `A_dir*`)
   partitions over K clients → 10 % local val. Rejected with a reason if < 2 classes or
   < 20 rows/client remain (e.g. a single-capture file like `DDoS-ACK 2.csv`).
The job's `data_source` is saved (`data_source.json`) and the FL page shows **that job's run**
(badge "Your run" vs "Saved benchmark run"), using the configured rounds (≤ 50).
Backend FL jobs are demos: never report them as results.

---

## 5. Detection, verdict and confidence

- Per-flow predictions come from `ml_service.predict_csv_bytes`. Endpoints (IPs/ports) are not
  in CIC-IoT-2023 CSVs → shown as "not in dataset"; the drawer shows raw model-input features
  (`key_features`) plus the attack's name, family and description.
- **Attack verdict** (`frontend/src/components/detection/AttackVerdict.tsx`, `computeVerdict`,
  tests in `src/tests/attackVerdict.test.ts`): < 5 % flows malicious → no attack. Votes = sum of
  per-flow confidence. Family (DDoS vs Mirai botnet) is decided first (clear if ≥ 1.5× the next
  family, else "mixed"); one attack is named only if it has ≥ 1.5× the votes of the runner-up,
  otherwise all attacks within that margin (max 3) are co-main.
  Verified: `DDoS-ACK 2 (1).csv` → DDoS / DDoS ACK fragmentation flood (99.7 %);
  `synthetic_gmm_stress_test_first_400.csv` → DDoS (72.4 %) / SYN flood + TCP flood (35.2 % vs 34.5 %).
- **Confidence is a raw softmax and is overconfident.** On `synthetic_gmm_stress_test_first_400`
  (legacy MLP): accuracy 94.5 %, but 12 of the 22 errors had ≥ 99 % confidence; weakest classes
  Mirai-greeth 67 %, DDoS-ACK 73 %, Mirai-greip 78 %. The UI never shows a rounded "100.0%"
  (`fmtConfidence` → ">99.99%"). Calibration (temperature scaling) is open work.

---

## 6. Verified results (only these are backed by artifacts)

All on **validation** data. The test set has **never** been evaluated.

| Run (`federated/runs/`) | What | Round/epoch 20 or final | Best val macro-F1 |
|---|---|---|---|
| `A_iid_s42_R20_E1` (26 Oct) | FedAvg, K=10, IID, 20 rounds, E=1 | acc 0.9895, macro-F1 0.9840 | — |
| `A_iid_s42_R20_E1_20261027_074922` (rerun) | same | acc 0.9898, macro-F1 0.9845 | — |
| `CENT_A_iid_s42_E40` (26 Oct) | Centralized, same init/scaler/data, 40 epochs | final 0.8994 | 0.9940 (ep 17) |
| `CENT_A_iid_s42_E40_20261027_075903` (rerun) | same | final 0.9361 | 0.9946 (ep 31) |

Centralized training oscillates (val macro-F1 0.82–0.995), so **always compare best-validation
checkpoints**. FL plateaued at macro-F1 ≈ 0.88–0.89 until round 12 because of Mirai-greip ↔
Mirai-greeth confusion — the main error source everywhere.

**Not run yet on real data:** `notebooks/SentinelFL_Complete_Study.ipynb` (all scenarios × seeds
42/43/44 × fedavg/centralized/local/xgboost, one-shot test evaluation).

---

## 7. Research-integrity rules (follow strictly)

1. **Never invent or "estimate" metrics.** Every number in the UI, README or docs must trace to
   a file in `federated/runs/`, `experiments/`, `backend/storage/evaluations/`, or an API response
   computed from real data. No fallback constants (see §12 for the ones that were removed).
2. **Test set is untouched until final evaluation.** Select on validation; evaluate once.
3. **Compare like with like** (model, init, scaler, data, checkpoint rule).
4. Report mean ± std over ≥ 3 seeds, per-class P/R/F1 and confusion matrices.
5. Keep existing runs; write new ones to new timestamped folders.
6. Forced ("Run anyway") outputs and backend demo FL jobs are never results.

### Previously unsupported numbers (status)
| Item | Status |
|---|---|
| Hardcoded `/evaluation/benchmarks` (175,669-flow matrices etc.) | **Removed.** Serves `experiments/results/benchmarks.json` or `[]`. |
| `docs/experiments.md` EXP-01…05 table | **Replaced** with the protocol. |
| `experiments/verified_fl_5clients_5rounds/` 100 % accuracy | Caveat README; first-N sampling bug fixed. |
| README 99.51 %, 100 %, 5 clients, wrong feature list | **Corrected.** |
| Landing page: 3 silos, 33 features, BoT-IoT, RF/LR | **Corrected.** |
| Design-doc 99.39 % / 98.12 % for `mlp_model.pth` | Unreproducible. Don't cite. |
| XGBoost 99.92 % / 79.87 % (`Untitled4.ipynb`) | **Invalid** (5 classes, mislabelled Mirai, 0 DDoS-TCP test rows). |
| Placeholder flow details (IPs, ports, packet sizes, durations) | **Removed** (test `test_batch_predictions_never_invent_endpoints_or_metrics`). |
| FL page fallbacks: 98.9 % accuracy, 134,385 samples/client, client loss = global × 1.05, F1 = acc × 0.992, sampleWeight 0.1, baseline F1 0.9828 | **Removed** (`mapFLRunDetail`; missing values stay undefined). |
| "100.0%" confidence | **Fixed** display (`fmtConfidence`). |

---

## 8. Repository map

```
backend/            FastAPI service (Python 3.11+)
  app/main.py         app, CORS, router registration
  app/adapters/       dataset adapters + registry
  app/api/routes/     health, datasets (incl. local-upload), preprocessing, models
                      (predict-csv, classify-dataset, load_dataset_bytes_for_user), federated
                      (train with optional dataset_id), evaluation, logs, system
  app/services/       ml_service (inference/eval/forced runs), fl_service (FedAvg jobs,
                      prepare_dataset_source), preprocessor, validator, job_runner, audit_service
  app/core/           config.py, security.py (Supabase JWT / local fallback)
  app/database/       supabase.py, local_store.py (offline datasets)
  tests/              pytest (incl. test_shared_pipeline, test_local_datasets, test_fl_uploaded_dataset)
  storage/            runtime: evaluations/, processed/, fl_runs/, local_datasets/
frontend/           React 18 + TS + Vite + Tailwind + Recharts (Vercel)
  src/pages/          Landing, auth pages, Dashboard, Datasets, Preprocessing, Models,
                      Federated, Detection, Evaluation, Logs, Settings
  src/components/detection/AttackVerdict.tsx
  src/api/client.ts   API client (+ mapFLRunDetail); src/lib/ supabase, localSession, …
  src/tests/          node --test suites (tsx)
federated/          fl_core.py · sentinelfl/ · tests/ · runs/ preprocessing/ reports/
                    splits/ (committed) · cache/ (gitignored, 274 MB, from Drive)
notebooks/          SentinelFL_Complete_Study.ipynb
ml/                 artifacts/ (legacy MLP + joblibs), data/samples/
experiments/        results/ (served by the API), verified_fl_5clients_5rounds/ (caveat)
scripts/            run_experiments.py, setup/00–04_*.sql (Supabase schema + RLS, run in order)
render.yaml, backend/Dockerfile, .dockerignore, frontend/vercel.json   hosting
```

---

## 9. Commands

```bash
# Backend (from backend/)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
#   "Address already in use" → lsof -ti:8000 | xargs kill
pytest -q

# Frontend (from frontend/)
npm install
npm run dev -- --host 127.0.0.1 --port 5173
npm run build                   # tsc && vite build — must pass before deploying
npx tsx --test src/tests/attackVerdict.test.ts
npm test                        # live-API suites need the backend on :8000

# Shared pipeline
python -m pytest federated/tests -q
python scripts/run_experiments.py --help

# Serve a federated model
ACTIVE_MODEL_BUNDLE=ml/bundles/fedavg-A_iid-s42 python -m uvicorn app.main:app --port 8000
```

**Test baseline (16 Nov 2026):** backend 81 pass + 2 environment failures
(`test_complete_preprocessing_flow` needs Supabase; `test_fl_train_job_creation_and_polling`
needs `federated/cache/`); federated 14 pass; frontend attackVerdict 5 pass, other suites need a
running backend; `npm run build` passes.

**Offline mode (no Supabase).** Without `VITE_SUPABASE_URL`, or when logged in as "Local
Researcher", datasets upload to the backend (`POST /api/v1/datasets/local-upload` →
`backend/storage/local_datasets/<ds-local-id>/`). The local token
(`local-researcher-session-token`) is attached to API calls. Disabled when
`ENVIRONMENT=production` (`ALLOW_LOCAL_AUTH`).

**Applying changes to the Mac clone via the device bridge:** write a `git format-patch` file into
`~/Desktop/SentinelFL`, then
`GIT_COMMITTER_NAME="Akshat Gupta" GIT_COMMITTER_EMAIL="akshat0709@users.noreply.github.com" git am x.patch && rm x.patch`
(git identity is not configured on the Mac; the bridge needs delete permission for git's lock files).

---

## 10. Deployment & configuration

**Status (16 Nov 2026): NOT deployed.** Code builds and tests pass locally, but:
1. `akshat0709/SentinelFL` on GitHub is empty → push first: `git push -u origin main`
   (local `main` tracks `upstream/main`; 16+ commits ahead).
2. Backend has never run on Render; the Docker image build was never verified (blocked in the
   dev sandbox). The first Render build is the real test.
3. Rotate the Supabase secret key (14 characters were printed during setup, see §12).

- **Frontend → Vercel** (`frontend/vercel.json`, SPA rewrite). Env: `VITE_API_BASE_URL`
  (Render URL), `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`. Unset API URL → the app
  calls `http://localhost:8000`.
- **Backend → Render** (`render.yaml`, free plan, Docker, CPU-only torch, `$PORT`,
  `WEB_CONCURRENCY=1`). Env: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (secret key, backend
  only), `SUPABASE_ANON_KEY` (publishable key), `STORAGE_BUCKET_NAME=sentinelfl-datasets`,
  `FRONTEND_ORIGINS=https://sentinelfl.vercel.app`, `ENVIRONMENT=production`; optional
  `ACTIVE_MODEL_BUNDLE`, `EXPERIMENT_RESULTS_DIR`, `FEDERATED_*_DIR`.
- **Supabase:** run `scripts/setup/00`→`04` in the SQL editor (idempotent). Bucket
  `sentinelfl-datasets`; free plan global upload limit **50 MB**. Auth → URL Configuration must
  list the Vercel URL.
- **Hosted limitations:** benchmark FL training fails (no 274 MB cache on Render; dataset-based
  FL works); Render free sleeps after ~15 min (≈1 min cold start) and has 512 MB RAM (tight for
  torch; very large CSVs may OOM); offline Local Researcher login is disabled in production.
- Never commit `.env` files or the secret key.

---

## 11. Conventions for changes

- `fl_core.py` is the single source of truth for model, transform, FedAvg; mirror changes in
  `ml_service.FLMLP` and re-verify checkpoints load with `strict=True`.
- New datasets: mapping in `federated/sentinelfl/adapters.py`, thin backend wrapper +
  `registry.py`, tests in both test suites, a sample CSV in `ml/data/samples/`.
- Train/serve new models through `sentinelfl` and ship them as `ModelBundle`s.
- Don't overwrite `federated/runs/` or `ml/artifacts/`; add versions alongside.
- UI numbers come from API responses, never constants or "?? 0.98"-style fallbacks.
- Results must belong to the dataset/run they are shown for (clear state on dataset change).
- Before committing: backend `pytest`, `python -m pytest federated/tests`, `npm run build`.
- Don't use broad `rm` globs in `backend/storage/` — some evaluation JSONs there are tracked.

---

## 12. What failed during development, and the fix

Grouped by area; "Open" items are still unresolved.

### Access & tooling
| Problem | Cause | Fix |
|---|---|---|
| Couldn't read the Drive notebooks / Mac folder | Drive permission not granted; desktop bridge offline | User granted Drive access; reconnected the bridge |
| Mac shell couldn't import `sentinelfl` | No PyTorch in the bridge VM (`fl_core` imports torch) | Read feature names from `federated/cache/clean_meta.json` instead |
| Markdown accidentally inserted into a `.py` script | Edit error | Reverted to a comment |
| Test artifacts and `package-lock.json` slipped into a commit | Over-broad `git add` | Commit amended |
| A cleanup `rm` glob deleted 7 **tracked** evaluation JSONs | Pattern too broad in `backend/storage/evaluations/` | Restored from `HEAD~1` before committing; rule added in §11 |

### Git / GitHub
| Problem | Cause | Fix |
|---|---|---|
| `git am` "patch not found"; user ran commands containing `/path/to` placeholders | Patch only existed in the sandbox | Patches written straight into `~/Desktop/SentinelFL` through the bridge |
| `git push` failed with HTTP 408 | Network/timeout on a large first push | Suggested `git config http.postBuffer 524288000`. **Open:** `origin` is still empty |
| `git am` via the bridge left `.lock` / `tmp_obj` files | Bridge forbids deletes by default | Requested delete permission, removed locks, `git am --abort`, re-applied |
| `git am` refused to commit | No git identity on the Mac | Run with `GIT_COMMITTER_NAME/EMAIL` env vars |

### Backend
| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: sentinelfl` in the backend adapter | `federated/` not on `sys.path` before import | `app/adapters/__init__` imports `app.core.config` first (adds the path) |
| Label-map patch didn't apply | Anchor text mismatch | Re-anchored; labels merged |
| `Address already in use` on :8000 | Old uvicorn still running | `lsof -ti:8000 \| xargs kill` |
| sklearn `InconsistentVersionWarning` on startup | Joblibs pickled with sklearn 1.6.1 | Harmless; retrain/re-export to silence |
| `NameError: Depends` after adding auth to `/federated/train` | Missing import | Imports fixed |
| FL "Dataset cache missing" | 274 MB cache is gitignored, only on Drive | User downloaded `X_clean.npy`, `y_clean.npy`, `clean_meta.json`; copied to `federated/cache/` and verified (1,756,693 × 39, splits consistent) |
| FL ignored the selected dataset; cards showed the saved Colab run; rounds silently capped at 5 | Backend only knew the cache; page reloaded the A_iid run | `dataset_id` training source; page shows the job's own run; real round count |
| FL on `DDoS-ACK 2` / 21-row sample rejected | Single class / too few rows (by design) | Clear error message; made `cic_iot_2023_mixed_4900.csv` (700 × 7 classes from the A_iid test split) for demos |

### Supabase
| Problem | Cause | Fix |
|---|---|---|
| Script 01 failed on a fresh project | Tables didn't exist | Added `00_base_schema.sql` |
| Script 02: `column reference "name" is ambiguous` | Unqualified column in a storage policy | `objects.name` |
| Script 03: cannot drop `get_user_workspace_ids` | Policies depend on it | `DROP FUNCTION … CASCADE` + recreated the dependent policies |
| Script 02: `must be owner of table objects` | Supabase owns `storage.objects` (RLS already on) | `ALTER TABLE storage.objects …` commented out |
| "Upload denied by Supabase Storage RLS policy" | Local Researcher session has no Supabase JWT | `useSupabaseStorage()` = Supabase configured **and** no local session → local upload |
| Uploads > 10 MB / > 50 MB failed | Bucket file-size limit / free-plan global cap of 50 MB | Checked bucket and global limits; 50 MB is a hard free-plan limit |
| 14 characters of the secret key printed while checking `.env` | Debug output | Disclosed; **Open:** rotate the key |

### Frontend / UX
| Problem | Cause | Fix |
|---|---|---|
| Datasets page unusable without Supabase | Upload path required Storage | Offline local-upload mode |
| Detection "Failed to fetch" / API offline | Backend was not running | Started it (run commands in §9) |
| Evaluation and Detection showed the previous dataset's results | State/localStorage not tied to dataset id | Cleared on dataset change; only matching `dataset_id` shown |
| Detection drawer showed invented IPs, ports, packet sizes | Placeholder generator | Removed; "not in dataset" + raw features |
| Extra `</div>` broke DetectionPage JSX | Edit error | Removed |
| N-BaIoT / Edge-IIoTset gave no result on Detection or Evaluation | Rejected by design | Explicit, clearly flagged "Run anyway" (§1) |
| Duplicate object keys in `client.ts` (TS1117) | Two replacement passes overlapped | Duplicates removed |
| "Run anyway" fired on a normal click | Click event passed as the `force` argument | Handlers wrapped: `() => handlePredictActiveDataset()` |
| `Badge variant="danger"` type error | Variant doesn't exist | `variant="red"` |
| Many invented fallback numbers on the FL page | `?? 0.98`-style defaults | `mapFLRunDetail`, values from the backend only |
| Run button label wrapped onto 3 lines | Flex shrink | `whitespace-nowrap`, `flex-shrink-0` |
| "100.0% confidence" | One-decimal rounding of 0.999997 | `fmtConfidence` (">99.99%") |
| Only per-row labels, no answer to "which attack is this?" | No dataset-level summary | Attack Verdict (family first, confidence-weighted votes, co-main on near-ties) |

### Hosting
| Problem | Cause | Fix |
|---|---|---|
| Docker image build couldn't be tested | Docker Hub / PyTorch CPU index blocked in the dev sandbox | Start command verified in a simulated layout. **Open:** verify on Render |

### Still open (known bugs)
- Dataset cards show the wrong record count (e.g. 310 for a 22,550-row file; 150 / 221 for a
  400-row file). Inference uses all rows; only the stored `row_count` is wrong.
- ~70 MB of old processed CSVs are tracked under `backend/storage/processed/` (should be ignored).
- "Max 100 MB" upload text should say 50 MB.
- Confidence is uncalibrated (§5).

---

## 13. Open work (priority order)

1. Push to `akshat0709/SentinelFL`, deploy backend (Render) + frontend env (Vercel), rotate the
   Supabase secret key, smoke-test the hosted site.
2. Fix the record count; untrack `backend/storage/processed/`; hide the benchmark FL option when
   the server has no cache; fix the 100 MB text.
3. Run `notebooks/SentinelFL_Complete_Study.ipynb` on Colab (seed 42 ≈ 3 h CPU, then 43/44);
   publish `experiments/results/` and one FedAvg bundle to `ml/bundles/`.
4. Temperature-scale the served model on validation data (calibrated confidence).
5. Decide on `Benign2.csv` (current cache excludes it); retire or retrain `mlp_model.pth` on the
   deduplicated data.
6. Research extensions: class-weighted loss for greeth/greip, partial participation, FedProx for
   α = 0.1.
