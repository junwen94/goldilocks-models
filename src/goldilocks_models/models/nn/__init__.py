"""Neural-network models: PyTorch-based feed-forward and attention
architectures over non-graph inputs.

This subpackage hosts neural networks that operate on tabular and
permutation-invariant inputs: MLPs over flattened structure features,
set-transformer-style attention over per-atom features, and similar
non-graph architectures. Crystal-graph models (CGCNN, SchNet, ALIGNN,
MACE-style) live separately in ``goldilocks_models.models.gnn``.

Stack: PyTorch + Lightning (``lightning.LightningModule``). Optional
dependency group ``nn`` — install with ``uv sync --extra nn``.

When to reach for nn
--------------------
* Classical baselines (XGBoost / LightGBM) have plateaued on a task.
* You want a learned latent representation that downstream tasks can
  share (a structure embedding reused across kpoints / pseudo / ...).
* The dataset has grown large enough that batched / GPU training pays
  for the engineering overhead.

For Phase 1 ``kpoints``, classical baselines are likely sufficient and
should be tried first. The nn subpackage exists so the infrastructure
is in place when a tabular baseline saturates or representation sharing
becomes worthwhile.

Planned modules (created on demand)
-----------------------------------
mlp               : Multi-layer perceptron over flattened structure
                    features. Smallest possible nn baseline.
set_transformer   : Permutation-invariant attention over per-atom
                    features (DeepSets / Set Transformer family).
                    Removes the flat-feature-vector limitation when
                    atom count varies across structures.
lightning_module  : Shared LightningModule base wiring optimiser,
                    scheduler, logging and checkpoint hooks. Concrete
                    networks subclass this to avoid re-implementing the
                    training loop.

Convention
----------
Mirrors ``goldilocks_models.models.classical``:
  * ``__init__(**hyperparams)`` accepting a flat dict (Hydra-friendly).
  * ``fit(X, y, sample_weight=None, eval_set=None)`` — wraps a Lightning
    ``Trainer`` invocation under the hood, so classical and nn models
    are interchangeable from training / registry code.
  * ``predict(X)`` and (where applicable) ``predict_proba(X)``.
  * ``save(path)`` / ``load(path)`` writing a directory containing
    ``state_dict.pt`` plus a ``hyperparams.json`` sidecar so the
    network can be reconstructed before loading weights.

``registry.exporter`` packages the whole directory as the released
artefact when ``save(path)`` writes a directory rather than a file.
"""
