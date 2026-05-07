"""Tasks: definitions of the prediction problems Goldilocks-models solves.

A *task* defines:
  * the prediction target (what column to predict / classify),
  * the loss function used during training,
  * the evaluation metrics and the slices over which they are reported,
  * any task-specific post-processing (e.g. mapping continuous predictions
    back onto a discrete grid).

A *task* deliberately does NOT define which model is used — that lives in
``goldilocks_models.models``. The same model class (e.g. XGBoost) can serve
multiple tasks, and the same task can be solved by multiple model classes.
This separation makes ablations across the model axis cheap.

Subpackages
-----------
kmesh    : recommend k-point density / mesh from structure (Phase 1 active).
ecutwfc  : recommend plane-wave cutoff for wavefunctions; ecutrho is treated
           as a derived quantity (dual * ecutwfc, dual fixed by pseudo type)
           and is not a separate task. Phase 2 placeholder — not yet swept
           in goldilocks-data.
smearing : recommend smearing kind (Gaussian / MP / cold / Fermi-Dirac) and
           width for metallic systems. Phase 2 placeholder.
pseudo   : recommend pseudopotential family + version + precision. Because
           every PseudoDojo / PAW-JTH family commits to a single XC
           functional, choosing a pseudo implicitly fixes XC in practice.
           Phase 1.5 onwards.
xc       : recommend exchange-correlation functional independently of pseudo.
           Only meaningful when the user constrains pseudo upstream
           (e.g. "must be PAW") and asks for an XC under that constraint.
           Phase 3+ placeholder.
resource : predict wallclock / RAM / ntasks from structure + parameters.
llm      : LLM-based explanation / dialog layer (later phase placeholder).

Coupling and inference order
----------------------------
These targets are physically coupled. Recommended values for one depend
on choices already made for others:

    user input (structure, optional constraints)
                 │
                 ▼
              pseudo  ──── implicitly fixes xc for PseudoDojo / PAW-JTH
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
     ecutwfc   smearing   kmesh
       │         │          │
       └─────────┴──────────┘
                 │
                 ▼
             resource (structure + all of the above)

Per-task models in this package are trained as **conditional predictors**:
each task's feature schema declares which upstream targets it consumes as
input features (e.g. the ecutwfc model takes pseudo_family as a feature).

Joint / sequential inference at recommendation time — walking this DAG
and stitching per-task predictions into a coherent SCF input — is the
responsibility of ``goldilocks-core``, not this package. We ship per-task
models plus a manifest declaring dependencies; core orchestrates the DAG.

In Phase 1 only ``kmesh`` is trained on real swept data. The other tasks
remain placeholders until their parameter axis is swept in goldilocks-data.
"""
