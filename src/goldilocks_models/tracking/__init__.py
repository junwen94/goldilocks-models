"""Tracking: experiment run transport (MLflow Phase 1; swap-friendly).

This subpackage provides a thin adapter over the experiment-tracking
backend. The training entry point and CLI route every parameter,
metric, and intermediate artefact through this adapter rather than
calling MLflow / wandb directly, so the tracker is a single point of
change.

Phase 1 backend: local MLflow store under ``mlruns/`` (single user,
offline-friendly). Future phases may swap to wandb or a hosted MLflow;
training code does not need to change.

Boundary with neighbouring subpackages
--------------------------------------
* ``evaluation`` computes metrics; ``tracking`` ships them.
* ``registry`` writes the final versioned artefact directory plus
  manifest for handoff to ``goldilocks-core``. ``tracking`` writes
  the development run log; ``registry`` writes the release.
* Hydra config is logged via ``tracking.logger`` at run start (not by
  Hydra itself) so any backend gets the same view.

Per-run log
-----------
Every training run records:
  * the full resolved Hydra config (all parameters, post-overrides),
  * validation metrics and slice tables (handed in by ``evaluation``),
  * a dataset hash pinning the Parquet snapshot,
  * ``goldilocks_models.__version__`` and the git SHA at run time,
  * any custom artefacts the model wants to surface (training-curve
    plots, predicted-vs-true scatter, loss histograms).

Submodules
----------
logger : The main interface — ``start_run`` / ``log_params`` /
         ``log_metrics`` / ``log_artifact`` / ``end_run``, plus a
         context-manager wrapper. Handles run naming, tag conventions,
         dataset-hash pinning, and exception-safe finalisation. Wraps
         MLflow's Python client; the user-facing API does not expose
         MLflow types so we can swap backends later.
"""
