"""Training utilities: trainers, callbacks, and loss functions shared
across model classes.

This subpackage holds the cross-cutting machinery that supports
``fit()`` on individual model classes. A ``models.<family>.<class>``
exposes ``fit(X, y, ...)``; under the hood that method delegates to
the framework-native trainer (XGBoost's training loop, Lightning's
``Trainer``, HuggingFace's ``Trainer``) and to the project-specific
callbacks / losses defined here.

A *training* utility is reusable across model classes. Anything
specific to a single model class belongs alongside that class in
``models/<family>/``. Anything specific to a single task's evaluation
or reporting belongs in ``goldilocks_models.evaluation``.

Submodules
----------
trainers   : Thin trainer wrappers — sklearn-style, Lightning-style,
             HuggingFace-style — that translate Hydra configs into the
             framework's native trainer invocation. Concrete model
             classes call into these so the training entry point is
             uniform across model families.
callbacks  : Reusable callbacks: early stopping with custom criteria,
             convergence-fraction logging during training, slice-aware
             validation, schedule-version sanity checks. Lightning
             callbacks for nn / gnn; XGBoost / LightGBM equivalents
             for classical where applicable.
losses     : Custom loss functions and objectives — ordinal
             cross-entropy and earth-mover's distance for the kpoints
             kindex target, asymmetric MAE for under-prediction of
             k-density, interval loss for k_line_density (held in
             reserve), interval-based losses for resource targets.
             Provided in framework-specific variants (torch
             ``nn.Module``, XGBoost custom objective callable).

Convention
----------
Anything in ``training/`` is invoked by a model's ``fit()`` method or
by the CLI's training entry point — not by users directly. The
``configs/`` Hydra layer points to specific trainers / callbacks /
losses by name; ``models/`` reads those references and instantiates
them at fit time.
"""
