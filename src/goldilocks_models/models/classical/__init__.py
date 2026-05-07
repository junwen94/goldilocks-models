"""Classical models: scikit-learn-compatible estimators.

This subpackage hosts thin wrappers over scikit-learn, XGBoost, and
LightGBM with task-friendly defaults. It is the Phase 1 starting point
for every task and uses only the core dependency set — no torch, no
PyTorch Geometric, no transformers.

-Why classical first
--------------------
-* Strong baselines on tabular structure features.
+Role in Phase 1
+---------------
+Phase 1 trains classical and gnn-class models in parallel on the
+``kpoints`` task. The classical line carries:
+
+* Strong baselines on tabular structure features.
 * Fast to train (minutes per run on a laptop), so iteration is cheap
   while the data contract with goldilocks-data is still settling.
 * Easy to interpret (feature importances, SHAP) — useful for
   sanity-checking that the model is using physics-relevant signals
   (pseudo family, anisotropy ratio, n_atoms) rather than spurious
   correlates.
+
+The gnn line provides the high-capacity counterpart (see
+``goldilocks_models.models.gnn``); the two are compared in the
+kpoints ablation reports rather than cascaded.


Planned modules (created on demand, not pre-stubbed)
----------------------------------------------------
xgboost  : XGBRegressor / XGBClassifier wrappers, including a custom
           ordinal objective for the kpoints kindex target.
lightgbm : LGBMRegressor / LGBMClassifier wrappers — typically faster
           than XGBoost on small categorical-heavy data.
sklearn  : ElasticNet / RandomForest / GradientBoosting baselines for
           ablation comparisons.

Convention
----------
Every class in this subpackage exposes:
  * ``__init__(**hyperparams)`` accepting a flat dict (Hydra-friendly).
  * ``fit(X, y, sample_weight=None, eval_set=None)``.
  * ``predict(X)`` and (where applicable) ``predict_proba(X)``.
  * ``save(path)`` and ``load(path)`` using joblib.

The save / load pair writes a single ``.joblib`` file containing the
fitted estimator. ``goldilocks_models.registry.exporter`` picks this
file up as the artefact ``model.joblib`` when packaging a release.
"""
