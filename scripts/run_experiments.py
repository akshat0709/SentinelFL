#!/usr/bin/env python3
"""SentinelFL experiment runner (centralized vs FedAvg vs local-only vs XGBoost).

Uses the SAME cleaned cache and split files as the existing Colab runs, the shared dataset
adapter/scaler (federated/sentinelfl), best-validation checkpoint selection, and a single
final evaluation on each scenario's held-out test split. Resumable: finished runs are skipped.

Examples
--------
# Full study on Colab (paths on Google Drive):
python scripts/run_experiments.py \
    --cache "/content/drive/MyDrive/Mini Project CSY/FL_Project/cache" \
    --splits "/content/drive/MyDrive/Mini Project CSY/FL_Project/splits" \
    --out "/content/drive/MyDrive/Mini Project CSY/FL_Project/results_v2" \
    --scenarios A_iid A_dir0.5 A_dir0.1 B_filehold_iid --seeds 42 43 44 \
    --methods fedavg centralized local xgboost

# Quick plumbing check (subsampled — NOT reportable):
python scripts/run_experiments.py --cache ... --splits ... --out /tmp/smoke \
    --scenarios A_iid --seeds 42 --rounds 2 --max-train-per-client 2000 --max-val-per-client 500 \
    --max-test-rows 5000

# Only rebuild the summary / benchmarks.json from finished runs:
python scripts/run_experiments.py --out ... --summarize-only

# Check that the existing cache equals what the shared adapter produces from the raw CSVs:
python scripts/run_experiments.py --verify-cache --data "/content/drive/MyDrive/Mini Project CSY/CICBOTNET" --cache ...
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "federated"))

from sentinelfl.experiments import METHODS, run_grid, summarize  # noqa: E402
from sentinelfl.partition import SCENARIOS  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", help="dir with X_clean.npy / y_clean.npy")
    p.add_argument("--splits", help="dir with <scenario>.npz split files")
    p.add_argument("--out", required=True, help="results directory (created if missing)")
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--methods", nargs="+", default=["fedavg", "centralized"], choices=METHODS)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--epochs", type=int, default=None, help="centralized/local epochs (default rounds*local_epochs)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-train-per-client", type=int, default=None, help="stratified subsample (smoke runs only)")
    p.add_argument("--max-val-per-client", type=int, default=None)
    p.add_argument("--max-test-rows", type=int, default=None)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--verify-cache", action="store_true")
    p.add_argument("--data", help="raw CSV dir (for --verify-cache)")
    p.add_argument("--publish-to", default=None,
                   help="copy benchmarks.json + summaries here (e.g. experiments/results in the repo)")
    a = p.parse_args(argv)

    if a.threads:
        import torch
        torch.set_num_threads(a.threads)

    if a.verify_cache:
        from sentinelfl.cache import verify_cache
        if not (a.data and a.cache):
            p.error("--verify-cache needs --data and --cache")
        res = verify_cache(a.data, a.cache)
        return 0 if res.get("ok") else 1

    if not a.summarize_only:
        if not (a.cache and a.splits):
            p.error("--cache and --splits are required to run experiments")
        run_grid(a.cache, a.splits, a.out, a.scenarios, a.seeds, a.methods,
                 rounds=a.rounds, local_epochs=a.local_epochs, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                 max_train_per_client=a.max_train_per_client, max_val_per_client=a.max_val_per_client,
                 max_test_rows=a.max_test_rows)
    else:
        summarize(a.out)

    if a.publish_to:
        os.makedirs(a.publish_to, exist_ok=True)
        for f in ("benchmarks.json", "results_summary.md", "results_summary.csv", "results_all_runs.csv"):
            if os.path.exists(os.path.join(a.out, f)):
                shutil.copy(os.path.join(a.out, f), os.path.join(a.publish_to, f))
        print(f"published summaries to {a.publish_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
