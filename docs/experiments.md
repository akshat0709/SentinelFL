# SentinelFL — Experiment Protocol

This file describes **how** results are produced. The numbers themselves are written by the
runner to `experiments/results/` (`results_summary.md`, `results_summary.csv`,
`benchmarks.json`) and must only be quoted from there.

> The previous version of this document listed results (EXP-01…05) that were not produced by
> any saved run. It has been replaced.

## Data
- CIC IoT 2023 captures in `Mini Project CSY/CICBOTNET/` (Google Drive), 7 classes:
  Benign, DDoS-ACK, DDoS-SYN, DDoS-TCP, Mirai-greeth_flood, Mirai-greip_flood, Mirai-udpplain.
- Cleaned cache `FL_Project/cache/`: 26 CSV files (no `Benign2.csv`), 2,756,340 raw rows →
  **1,756,693 rows** after removing NaN/inf rows and exact duplicates. Built by FLipynb cell 3;
  `sentinelfl.cache.verify_cache` checks it equals the shared-adapter pipeline.
- Splits `FL_Project/splits/` (seed 42, K = 10 clients, 10 % local validation per client):

| Scenario | Test set | Client partition |
|---|---|---|
| `A_iid` | random stratified 15 % (263,504 rows) | IID per class |
| `A_dir0.5` | same as A | Dirichlet α = 0.5 per class, Benign floor 0.5 |
| `A_dir0.1` | same as A | Dirichlet α = 0.1 per class, Benign floor 0.5 |
| `B_filehold_iid` | last capture file of each class + 15 % Benign (405,690 rows) | IID |

## Pipeline (identical for every method)
1. **Dataset adapter** — `sentinelfl.CICIoT2023Adapter` maps raw columns/labels to the
   39-feature / 7-class contract (also used by the backend for inference).
2. **Scaling** — log1p, then standardisation with mean/std aggregated federatedly from each
   client's training rows (`n, Σz, Σz²` only).
3. **Model** — FLMLP 39→128→64→32→7 (15,687 parameters), Adam lr 1e-3, batch 512,
   optimiser re-created each pass, same initial weights for every method of a seed.

## Methods
| Method | Description | Passes over data |
|---|---|---|
| `fedavg` | 10 clients, all participate each round, sample-weighted FedAvg | R × E (20 × 1) |
| `centralized` | pooled union of client training splits | 20 epochs |
| `local` | each client alone on its own data, evaluated on the shared test set | 20 epochs |
| `xgboost` | pooled XGBoost (hist, early stopping on validation) — reference | — |

## Model selection and testing
- Checkpoint = **best validation macro-F1** (rounds for FedAvg, epochs otherwise).
- The selected checkpoint is evaluated **once** on the scenario's test split.
- Seeds 42, 43, 44 change initial weights and training order; splits stay fixed (seed 42).

## Reported metrics
Accuracy; per-class and macro precision/recall/F1; weighted F1; confusion matrix;
binary FPR (benign flagged as attack) and FNR (attack predicted benign); per-client validation
macro-F1 at the selected round; communication = 2 × K × 62,748 bytes per round
(≈ 25.1 MB for 20 rounds, K = 10) vs. raw-data transfer for centralization; wall-clock time.

## How to run
See `notebooks/SentinelFL_Complete_Study.ipynb` (Colab) or `scripts/run_experiments.py --help`.
