"""explanation: NL-explanation / dialog layer that turns recommendations
into natural-language guidance for users.

Target          : free-form text — explanations of why the recommended
                  ecutwfc / kpoints / pseudo were chosen, what tolerances
                  were assumed, and which surprises in the user's
                  structure (heavy elements, magnetism, low symmetry)
                  drove the decision.
Label source    : TBD — initial training data is likely a mix of
                  human-written explanations seeded from a small expert
                  pool plus LLM-bootstrapped pairs validated by humans.
Loss            : depends on approach — SFT cross-entropy for instruction
                  tuning, DPO / RLAIF for alignment, or pure prompt
                  engineering with no fine-tuning at all.
Primary metric  : human evaluation (faithfulness, completeness, calibration);
                  LLM-as-judge as a cheaper proxy.
Reporting slices: by "recommendation surprise" level — explanations of
                  default-looking recommendations are easy; explanations
                  for outliers (lanthanides, very large cells, magnetic
                  metals) are where this task earns its keep.
Scope           : code-agnostic, calc_type-agnostic.
Status          : Phase 3+ placeholder.
Inputs          : structure features + the full set of upstream task
                  recommendations (this is a downstream consumer).
Dependencies    : pseudo, xc, ecutwfc, smearing, kpoints, resources.

Notes
-----
Naming: the task is named ``explanation`` (what is predicted) rather
than ``llm`` (the algorithm typically used). Algorithm-family naming
lives under ``goldilocks_models.models.llm``. The two axes are
deliberately orthogonal: this task could in principle be served by a
non-LLM template engine, and ``models/llm/`` could in principle serve
other tasks (rare in practice but the structure does not preclude it).

Whether the model is fine-tuned in-repo or served as a prompt-engineering
layer on top of a hosted model is an open question. The package leaves
the door open by keeping training, prompt assets, and evaluation
harnesses here even if the deployed artefact is just a prompt template
plus a retrieval index.
"""
