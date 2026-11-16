# SentinelFL — Federated IoT Threat Intelligence

A research-grade cybersecurity platform engineered to detect malicious IoT network flow traffic using machine learning and privacy-preserving federated learning (FedAvg).

---

## 1. Project Overview

**SentinelFL** (MiniProjectBotnet) is an end-to-end IoT threat intelligence platform designed to address botnet attacks and DDoS campaigns against edge devices without centralizing raw network telemetry. The system ingests network flow captures, validates and preprocesses data without data leakage, executes centralized ML and distributed Federated Learning (FedAvg), classifies batch network flows, and computes both local ground-truth evaluation and comparative benchmark analytics from saved experiment runs.

### Core Capabilities
1. **Dataset Ingestion & Storage**: Private Supabase storage with workspace isolation (RLS), with graceful local fallback for offline demonstration.
2. **Modular Dataset Adapter Architecture**:
   - **CIC IoT 2023 Adapter**: Automatic schema validation, infinite/NaN sanitization, and canonical 39-feature extraction supporting live FLMLP inference.
   - **Edge-IIoTset Adapter**: Validates Zeek/Wireshark network telemetry schema, prevents silent zero-imputation or arbitrary feature aliasing, and provides diagnostic rejections for unsupported FLMLP inference.
   - **N-IoT / N-BaIoT Adapter**: Validates 115 stream decay window statistics (`MI_dir_`, `H_L`, `HH_L`), enforcing cybersecurity integrity by refusing unsupported inference.
3. **Leak-Free Preprocessing**: Train-only parameter fitting (scalers/encoders), reproducible seed configuration, and artifact persistence.
4. **PyTorch ML Model Integration**: Canonical 39 → 128 → 64 → 32 → 7 multi-layer perceptron (FLMLP) detecting 7 distinct traffic classes. Models are served as **ModelBundles** (weights + dataset adapter + their own scaler), so centralized and federated models share one adapter.
5. **Federated Learning (FedAvg)**: 10 simulated clients (IID, Dirichlet non-IID $\alpha \in \{0.5, 0.1\}$, and file-holdout), sample-weighted FedAvg, federated log1p standardisation, communication tracking (62,748 bytes per model transfer; ≈1.25 MB per round for 10 clients), best-validation checkpoint selection, exported as servable bundles.
6. **Full Dataset Batch Inference**: Memory-safe chunked batch inference processing complete datasets, with pagination, search, sorting, class filtering, and CSV export.
7. **Single-Flow Simulator**: Interactive socket & 39-feature vector simulator with attack/benign presets for real-time risk triage.
8. **Dual-Scope Evaluation**: Distinct separation of current active dataset evaluation (7×7 confusion matrix, macro metrics, FPR, FNR) and benchmark experiments loaded from `experiments/results/benchmarks.json` (empty until the experiment runner has been executed).
9. **Granular Service Health Checks**: Readiness probing across backend, database, storage, model artifacts, inference service, and FL worker.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SentinelFL React Frontend                         │
│   (Vite + TypeScript + TailwindCSS / Lucide Icons / Active Dataset Context) │
└──────────────┬──────────────────────────────┬───────────────────────────────┘
               │ Supabase JWT (or Demo Mode)  │ Supabase Auth & Storage API
               ▼                              ▼
┌──────────────────────────────┐   ┌──────────────────────────────────────────┐
│       FastAPI Backend        │   │          Supabase Cloud Platform         │
│  - /health (Detailed Probes) │   │  - GoTrue Auth (JWT verification)        │
│  - /api/v1/datasets          │   │  - PostgreSQL with Row-Level Security    │
│  - /api/v1/datasets/adapters │◄──┤  - Private Storage:                      │
│  - /api/v1/preprocessing     │   │    'sentinelfl-datasets'                 │
│  - /api/v1/models            │   │  - Local Fallback:                       │
│  - /api/v1/federated         │   │    Offline researcher demonstration mode │
│  - /api/v1/evaluation        │   └──────────────────────────────────────────┘
└──────────────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ML & Federated Learning Engines                        │
│  - Dataset Adapters: backend/app/adapters/ (CIC-IoT-2023, Edge-IIoTset, N-IoT)│
│  - PyTorch Checkpoint: ml/artifacts/mlp_model.pth (FLMLP 39→128→64→32→7)    │
│  - Preprocessing Artifacts: scaler.joblib, label_encoder.joblib             │
│  - Federated Core: federated/fl_core.py (PyTorch Federated Averaging)       │
│  - Shared pipeline: federated/sentinelfl/ (adapter, scaler, FedAvg, eval)   │
│  - Partitions: federated/splits/ (10 clients, IID / Dirichlet / holdout)    │
│  - Servable models: ModelBundle dirs (ACTIVE_MODEL_BUNDLE)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack & Architectural Clarifications

- **Frontend**: React 18, TypeScript, Vite, Lucide React, TailwindCSS, Recharts; tests run with the Node test runner (tsx).
- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2.
- **Data & Preprocessing**: Pandas, NumPy, Scikit-learn, Joblib.
- **Machine Learning**: PyTorch (CPU / CUDA compatible).
- **Federated Learning Implementation**:
  - The federated learning core uses a pure PyTorch implementation of Federated Averaging (FedAvg), mathematically identical to Flower's FedAvg aggregation ($w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k$).
  - *Engineering Note on Flower (`flwr`)*: While initial project sketches referenced Flower, Flower is not required at runtime in `backend/.venv` due to dependency constraints with modern FastAPI versions; the custom PyTorch FedAvg engine (`federated/fl_core.py` and `app/services/fl_service.py`) provides lightweight, deterministic, and fully reproducible multi-client federated training without gRPC overhead.
- **Security & Database**: Supabase Auth (JWT), Supabase Storage, PostgreSQL with RLS, with an offline demonstration mode for self-contained execution.

---

## 4. Machine Learning & Federated Learning Specifications

### 4.1. Feature Contract (39 Features)
The canonical PyTorch MLP expects these 39 CIC IoT 2023 flow features in exactly this order
(source of truth: `federated/fl_core.py::FEATURES`):
```
psh_flag_number, IAT, ARP, cwr_flag_number, Min, syn_flag_number, ece_flag_number, Max,
Tot size, DNS, fin_flag_number, ack_flag_number, Std, rst_flag_number, Telnet, Header_Length,
rst_count, Variance, HTTP, ICMP, Protocol Type, Tot sum, AVG, fin_count, IRC, IGMP, IPv,
Number, TCP, SSH, UDP, Time_To_Live, SMTP, LLC, Rate, DHCP, ack_count, syn_count, HTTPS
```

### 4.2. Target Classes (7 Classes)
Categorical encoding maps labels to 7 canonical threat categories:
1. `Benign`
2. `DDoS-ACK`
3. `DDoS-SYN`
4. `DDoS-TCP`
5. `Mirai-greeth_flood`
6. `Mirai-greip_flood`
7. `Mirai-udpplain`

### 4.3. Dataset Adapters & Compatibility Policy
- **CIC IoT 2023**: Fully compatible with the 39-feature FLMLP model. Supported for batch inference and evaluation.
- **Edge-IIoTset**: Automatically identified via Zeek / Wireshark fields (`frame.time`, `tcp.flags`, etc.). Incompatible with the 39-feature FLMLP model; the adapter rejects inference with an informative diagnostic message rather than fabricating predictions.
- **N-IoT (N-BaIoT)**: Automatically identified via 115 stream decay window statistics (`MI_dir_`, `H_L`, `HH_L`). Incompatible with the 39-feature FLMLP model; the adapter safely rejects inference to prevent scientific invalidity.

### 4.4. Model Checkpoints & Artifact Directory
Located in `ml/artifacts/`:
- `mlp_model.pth`: Pretrained PyTorch FLMLP state dictionary (39 → 128 → 64 → 32 → 7).
- `scaler.joblib`: Fitted Scikit-learn StandardScaler (39 features).
- `label_encoder.joblib`: Fitted LabelEncoder for the 7 canonical classes.
- `feature_columns.joblib`: Pickled list of the 39 feature names in canonical order.
- Federated models are not stored here; they are exported as ModelBundle directories by `scripts/run_experiments.py` (see §7).

---

## 5. Getting Started & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 5.1. Backend Setup
```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (optional for local demo)
cp .env.example .env

# Start FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Server runs on http://127.0.0.1:8000
```

### 5.2. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev -- --host 127.0.0.1 --port 5173
# Accessible on http://127.0.0.1:5173
```

### 5.3. Quick Demonstration Workflow
1. Open `http://127.0.0.1:5173/` in your browser.
2. Click **"Enter as Local Researcher (Offline / Demo)"** on the login page.
3. On the **Detection** page:
   - In **Single Flow Simulator**, select the `DDoS-SYN Flood` or `Benign Traffic` preset and click **Classify Flow**.
   - In **Batch CSV Detection**, upload `ml/data/samples/cic_iot_2023_sample.csv` and click **Analyze Traffic File** to view real inference on 21 flows across all 7 classes.
   - Test safety boundaries by uploading `ml/data/samples/edge_iiotset_sample.csv` or `n_iot_sample.csv`; the system safely rejects the incompatible schemas with clear diagnostic feedback.

---

## 6. Testing Suite

SentinelFL includes comprehensive automated tests across both backend and frontend.

### 6.1. Running Backend Tests (pytest)
```bash
cd backend
.\.venv\Scripts\python.exe -m pytest -v
```
**Coverage** (run `pytest` for current counts; two tests need Supabase credentials and `federated/splits/`):
- Modular Dataset Adapter architecture (CIC IoT 2023, Edge-IIoTset, N-IoT identification, extraction, and validation).
- Refusal of silent zero-imputation and incompatible dataset rejection.
- CSV validation, malformed input handling, missing values, duplicates, infinite values.
- Leak-free preprocessing, train/val/test splits, scaler preservation.
- PyTorch model checkpoint loading, single-flow inference, chunked batch classification.
- Evaluation metrics (7×7 confusion matrix, macro precision/recall/F1, FPR, FNR, ground-truth isolation).
- Federated data partitioning (IID, Dirichlet non-IID), FedAvg aggregation.
- Supabase JWT authentication, local researcher fallback, and workspace isolation.
- Health readiness endpoints (`/health` readiness breakdown).

### 6.2. Running Frontend Tests (Node test runner via tsx)
```bash
cd frontend
npm test
```
The live-API suites expect the backend running on `http://localhost:8000`.

### 6.4. Shared pipeline tests
```bash
python -m pytest federated/tests -q
```

### 6.3. Production Build Validation
```bash
cd frontend
npm run build
```
Type checks and bundles the client application with 0 TypeScript errors.

---

## 7. Experiments (centralized vs federated)

All experiments run through the shared pipeline in `federated/sentinelfl/` on the cleaned cache and
fixed split files from the Colab study (10 clients; `A_iid`, `A_dir0.5`, `A_dir0.1`, `B_filehold_iid`).
Every method uses the same adapter, federated scaler, initial weights and checkpoint rule (best
validation macro-F1); the selected checkpoint is evaluated once on the held-out test split.
Protocol: `docs/experiments.md`. Colab: `notebooks/SentinelFL_Complete_Study.ipynb`.

```bash
python scripts/run_experiments.py --cache <FL_Project/cache> --splits <FL_Project/splits> \
    --out <results_dir> --scenarios A_iid A_dir0.5 A_dir0.1 B_filehold_iid \
    --seeds 42 43 44 --methods fedavg centralized local xgboost \
    --publish-to experiments/results
```

Results are written to `experiments/results/results_summary.md` and served by
`GET /api/v1/evaluation/benchmarks`. Results verified so far (validation only, Colab Flower runs
in `federated/runs/`): FedAvg A_iid 20 rounds — validation macro-F1 0.9840 / 0.9845 (two runs).

To serve a federated model: `ACTIVE_MODEL_BUNDLE=<path to bundle dir> uvicorn app.main:app`.

## 8. Current Scope & Known Boundaries

1. **Dataset Schema Compatibility**: Live inference with `mlp_model.pth` is explicitly constrained to datasets adhering to the CIC IoT 2023 39-feature schema. Incompatible datasets (Edge-IIoTset, N-IoT) are validated at ingestion but protected against invalid inference by the Dataset Adapter registry.
2. **Batch Flow vs. Live Network Sniffing**: SentinelFL performs memory-safe chunked batch classification on ingested network flow CSVs. Live continuous packet capture (direct Zeek/libpcap daemon tapping physical interfaces) requires local interface privileges and is not simulated as fake live streaming.
3. **Classical Baseline**: XGBoost (corrected 7-class labels, same splits) is available as a reference method in the experiment runner; the production inference engine is the 7-class PyTorch FLMLP.
