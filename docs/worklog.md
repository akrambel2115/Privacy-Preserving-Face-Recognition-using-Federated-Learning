# Worklog

- 2026-05-07: Purged the nested benchmark checkpoint blobs from the local branch history and widened `.gitignore` to block future `results/**/checkpoint.pt` / `.pth` artifacts.
- 2026-04-28: Removed tracked checkpoint binaries from the push path and added ignore rules for future `.pt` / `.pth` artifacts under `results/checkpoints/`.
- 2026-05-04: Rewrote the branch history to strip `results/checkpoints/best_run.pt` and `results/checkpoints/smoke_checkpoint.pt`, then pushed `hyperparameter` successfully.
- 2026-05-05: Improved evaluation-run usability: fixed DP noise generation compatibility, added round/client progress logging, and made mean feature initialization run on the eval loader (no augmentations) with optional batch-level init progress logs.
- 2026-05-06: Fixed offline simulator crash by adding the missing `_sorted_client_names` helper and implemented optional checkpoint saving via `--checkpoint-path` (compatible with `scripts/verify_checkpoint.py`).
- 2026-05-06: Enabled Local-DP to run together with SecAgg+ by removing the compatibility blocks and adding anchor-DP support to the secure payload path.
- 2026-05-06: Added DP flags to `scripts/run_server.py` and plumbed `dp-clip-norm` / `dp-noise-multiplier` / `dp-anchor-noise-multiplier` through the SecAgg+ ServerApp run config.
- 2026-05-06: Added default DP keys to the Flower app config in `pyproject.toml` so DP (and DP+SecAgg+) can be toggled by config.
- 2026-05-06: Implemented SecAgg+ checkpoint saving via `checkpoint-path` run config and added `scripts/benchmark_tradeoff_suite.py` to run a 4-mode benchmark (none/ldp/secagg/ldp+secagg) across a sigma sweep, dumping checkpoints and summary reports.
- 2026-05-06: Added a repo-root wrapper `benchmark_tradeoff_suite.py` which forwards to `scripts/benchmark_tradeoff_suite.py` (so the suite can be launched from the project root without typing the `scripts/` prefix).