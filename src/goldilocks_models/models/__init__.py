"""Models: pluggable algorithm implementations, decoupled from any task.

A *model* in this package is responsible for:
  * fitting its parameters from (features, labels),
  * producing predictions from features,
  * saving and loading its own weights.

A model is NOT responsible for:
  * data loading or feature engineering   -> goldilocks_models.data
  * task-specific loss / metric / slices  -> goldilocks_models.tasks.<task>
  * experiment logging                    -> goldilocks_models.tracking
  * artefact versioning and manifests     -> goldilocks_models.registry

This separation lets the same model class serve multiple tasks. An
XGBoost regressor is just an XGBoost regressor; whether the target is
kindex (kpoints task) or wallclock seconds (resources task) is a
concern of the task, not of the model.

There is deliberately no abstract ``Model`` base class yet. The
interface will be extracted once at least two implementations exist for
the same task and a stable common surface emerges. Inventing the base
class up front tends to lock in assumptions that do not survive
contact with the second implementation.

Subpackages
-----------
classical : scikit-learn-compatible estimators — thin wrappers over
            scikit-learn, XGBoost and LightGBM with task-friendly
            defaults. Phase 1 starting point. Uses only the core
            dependency set; no torch.
nn        : feed-forward neural networks via PyTorch + Lightning
            (MLPs over flattened structure features, set-transformer
            over per-atom features, ...). Optional dep group ``nn``.
gnn       : crystal-graph neural networks via PyTorch Geometric
            (CGCNN-style, SchNet-style, ALIGNN-style, MACE-style).
            Optional dep group ``gnn``.
llm       : language-model-based explainers and dialog wrappers via
            HuggingFace transformers + PEFT for the explanation task.
            Optional dep group ``llm``.

The subpackage boundary aligns with the optional-dependency boundary so
that a deployment that only needs the classical baselines does not pay
the import cost of torch, PyTorch Geometric or transformers.

Convention (not enforced by a base class — yet)
-----------------------------------------------
Each model class should expose, at minimum:
  * a constructor that accepts a hyperparameter dict (Hydra-friendly),
  * ``fit(X, y, sample_weight=None, **kwargs)``,
  * ``predict(X)`` returning the task's natural output type,
  * ``save(path)`` and ``load(path)`` using the framework's native
    serialisation (joblib for classical, torch state_dict for nn/gnn,
    safetensors for llm).
Stick to these names so ``goldilocks_models.registry.exporter`` can
treat all models uniformly when writing artefacts.
"""
