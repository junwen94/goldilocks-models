"""Evaluation: metrics, slice-aware reports, and downstream-truth checks
for trained models.

This subpackage implements the evaluation logic that each task's
``__init__.py`` declares. Tasks state WHICH metrics to compute and on
WHICH slices to report; evaluation provides the actual functions.

A *metric* here is a function from (predictions, labels) to a scalar
or a structured score. A *slice* is a filter over the test set.
A *report* is the cross-product (metric x slice) rendered as a table
for MLflow logging, model-card inclusion, and side-by-side ablation.

Boundary with neighbouring subpackages
--------------------------------------
* ``tasks/<task>`` declares the metric and slice list for the task.
* ``training/callbacks`` computes a small subset of metrics during
  training (cheap, online). Evaluation here is post-hoc, more
  thorough, and writes out reports.
* ``tracking`` is the transport (MLflow, model-card rendering), not
  the metric computation itself.

Submodules
----------
metrics  : Metric implementations — MAE / RMSE / classification
           accuracy / top-k / earth-mover's distance / convergence-
           fraction / MAPE-in-tail-buckets, plus
           faithfulness / completeness / calibration for the
           explanation task. Pure functions of (y_true, y_pred),
           framework-neutral (numpy in, numpy out).
slices   : Slice definitions and a slice-application engine. A slice
           is a named filter expression over the test DataFrame; the
           engine returns ``{slice_name: indices}`` for a registered
           slice list. Reusable across tasks.
reports  : Render metric-x-slice tables to Markdown for the model
           card and JSON for MLflow / manifest. Includes comparison
           reports (run A vs run B) for ablation work.

Convention
----------
Metrics and slices are pure / declarative; reports orchestrate them.
Anything stateful (dataset hashes, run identifiers) is passed in
from above; evaluation utilities never reach into the filesystem or
MLflow themselves — that is ``tracking``'s job.
"""
