# goldilocks-models

ML and LLM training, tracking, and evaluation for the UKRI Goldilocks DFT-recommendation system.

This package consumes ML-ready Parquet from `goldilocks-data` and ships
versioned per-task model artefacts plus manifests to `goldilocks-core`,
which orchestrates recommendation at inference time.

## Documentation

- **[PLAN.md](PLAN.md)** — full design source of truth (contract, architecture, decisions, roadmap).
- **[docs/](docs/)** — derived chapters, added on demand.

## Phase 1 scope

- DFT code: Quantum ESPRESSO (`pw.x`)
- Calculation type: SCF only
- Structures: Materials Cloud MC3D PBEsol v2
- Pseudopotentials: PseudoDojo NC + PAW-JTH (15 active families)
- Active task: `kpoints` (k-mesh recommendation as kindex regression)
- Other tasks (`ecutwfc` / `smearing` / `pseudo` / `xc` / `resources` / `explanation`) are placeholders awaiting upstream data sweeps.

## Quick start

```bash
uv sync                         # core deps only
uv sync --extra nn              # plus PyTorch + Lightning
uv sync --extra gnn             # plus PyTorch Geometric
uv sync --extra llm             # plus HuggingFace transformers + PEFT
uv sync --all-extras            # everything
```

## Repository layout

```
src/goldilocks_models/
├── data/        # Parquet IO, feature engineering, splits, dataset wrappers
├── tasks/       # 7 prediction problems — what to predict
├── models/      # 4 algorithm families — how to predict
├── training/    # train loops, callbacks, losses
├── evaluation/  # metrics and slice reports
├── tracking/    # MLflow adapter
├── registry/    # versioned artefacts + manifests for handoff to goldilocks-core
└── cli/         # gm train / eval / predict / register
```

## Sibling repositories

| Repo | Role | Relationship |
|---|---|---|
| `goldilocks-data` | DFT sweeps + Parquet datasets | input |
| `goldilocks-models` (here) | ML / LLM training | — |
| `goldilocks-core` | Recommendation + parsing + LLM explanation | output (models + manifests) |
| `goldilocks-webapp` | Frontend | indirect (via core) |

UKRI Goldilocks grant EP/Z530657/1.
