# Experiment results (served by the backend)

Populated by `scripts/run_experiments.py --publish-to experiments/results` (or the last cell of
`notebooks/SentinelFL_Complete_Study.ipynb`):

- `benchmarks.json` — served by `GET /api/v1/evaluation/benchmarks`
- `results_summary.md` / `results_summary.csv` — mean ± std per scenario and method
- `results_all_runs.csv` — one row per (scenario, method, seed)

Until these files exist the Evaluation page shows an empty state. Never edit numbers by hand.
