# Caveat — not a reportable result

Produced by `scripts/run_5client_fl_experiment.py`, which took the FIRST 5,000 indices of each
client's sorted split array and of the test set. The cache is ordered by source file (Benign
first), so these subsets are expected to be (almost) single-class, which explains the 100 %
accuracy. The sampling bug is fixed in `backend/app/services/fl_service.py` and the shared
pipeline (`sentinelfl.partition.stratified_subsample`). Use `scripts/run_experiments.py` for results.
