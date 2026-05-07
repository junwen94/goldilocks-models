# goldilocks-models — PLAN

> Source of truth for the goldilocks-models repository (lives at
> `docs/PLAN.md`; README.md is the short pointer that links here).
> Per-task implementation details live in the corresponding
> `goldilocks_models.tasks.<task>.__init__` docstrings; this document
> covers cross-cutting design.

## 0. Status of decisions

| Decision | Status | Where |
|---|---|---|
| 4-repo ecosystem layout | locked | §1 |
| Phase 1 scope (QE / SCF / MC3D PBEsol v2) | locked | §2 |
| 4-dimensional inference input | locked | §3 |
| Task list (7) and naming convention | locked | §4 |
| `kpoints` target = kindex (ordinal) | locked | §4.3, `tasks/kpoints/__init__.py` |
| `tasks × models` matrix structure | locked | §5 |
| Algorithm-family layout (classical / nn / gnn / llm) | locked | §5 |
| GNN in Phase 1 (alongside classical baselines) | locked | §5.2, §12 |
| LLM-encoder embeddings as features (frozen) in Phase 1 | locked | §6, §12 |
| Data-layer modules (loaders / features / splits / datasets) | sketch locked | §6 |
| Configs framework: Hydra | sketched | §7 |
| Tracking framework: local MLflow Phase 1 | tentative | §8 |
| Registry: versioned artefact + manifest | sketched | §9 |
| CLI surface: `gm train / eval / predict / register` | sketched | §10 |

## 1. Context: UKRI Goldilocks ecosystem

UKRI Goldilocks (grant EP/Z530657/1) produces "just right" DFT inputs
(ecutwfc / k-grid / smearing / pseudo / SLURM scripts) for users.
Four sibling repositories, in dataflow order:

| Repo | Role | This repo's relationship |
|---|---|---|
| `goldilocks-data` | Generates training data via AiiDA + Quantum ESPRESSO on SCARF; emits ML-ready Parquet | input |
| `goldilocks-models` (here) | Trains, tracks, evaluates ML / LLM models; exports versioned artefacts + manifests | — |
| `goldilocks-core` | Recommendation + parsing + LLM explanation; consumes our manifests | output |
| `goldilocks-webapp` | Frontend | indirect (via core) |

Two contracts:
- ← `goldilocks-data`: Parquet rows tagged `under` / `just_right` / `over` per SCF.
- → `goldilocks-core`: per-task model artefacts plus a JSON manifest declaring scope, dependencies, and schedule version. Core orchestrates the inference DAG; we do not.

## 2. Phase 1 scope

| Dimension | Phase 1 lock |
|---|---|
| DFT code | Quantum ESPRESSO (`pw.x` 7.3) |
| Calculation type | SCF only (no relax / bands / DOS / phonon / MD) |
| Structures | Materials Cloud MC3D PBEsol v2 (5k sample → 15k) |
| Pseudopotentials | PseudoDojo NC v0.4 (11) + PAW-JTH v1.0 (4) = 15 active |
| Sweep axis | k-mesh only; ecutwfc / smearing fixed at PseudoDojo defaults |
| HPC | SCARF (STFC) |
| XC functional | PBEsol (committed via pseudo family choice) |

Out of scope for Phase 1: VASP / CP2K / ABINIT, relax / bands / DOS / phonon / MD, USPP / GBRV pseudos, ARCHER2 / Iridis HPC, multi-user PostgreSQL.

## 3. Inference input (4 dimensions)

Every recommendation is conditional on:

| Dimension | Phase 1 value | Future |
|---|---|---|
| `structure` | crystal (always required) | always |
| `code` | QE | VASP, CP2K, ABINIT |
| `calc_type` | SCF | relax, bands, DOS, phonon, MD |
| `user_constraints` | none / optional | "must be PAW", "≤ 4 nodes", "energy tol < 1 meV/atom", ... |

Each task model declares `supported_scope: {code, calc_type}` in its manifest. Core refuses to use a model outside its trained scope.

## 4. Tasks

Tasks are named by what they predict (the target), not by the algorithm.
Algorithm-family names live in `models/`; the two axes are orthogonal.

### 4.1 Task list

| Subpackage | Target | Type | Status |
|---|---|---|---|
| `kpoints` | kindex (ordinal index in goldilocks-core's mesh-transition schedule) | regression / ordinal | Phase 1 active |
| `ecutwfc` | plane-wave cutoff (Ry) | regression | Phase 2 placeholder |
| `smearing` | (kind, width) for metals | classification + regression | Phase 2 placeholder |
| `pseudo` | pseudopotential family | classification | Phase 1.5 placeholder |
| `xc` | exchange-correlation functional | classification | Phase 3+ placeholder |
| `resources` | (wallclock, peak_RAM, ntasks) | multi-output regression | Phase 1.5 / 2 placeholder |
| `explanation` | natural-language explanation | NL generation | Phase 3+ placeholder |

Per-task details (target, label, loss, metric, slices, scope, dependencies) are documented in each subpackage's `__init__.py`.

### 4.2 Coupling and inference DAG

Targets are physically coupled. The Phase 1 (QE / SCF) DAG:

```
user input: (structure, code, calc_type, constraints)
             │
             ▼
          pseudo  ──── implicitly fixes xc for PseudoDojo / PAW-JTH
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
 ecutwfc   smearing   kpoints
   │         │          │
   └─────────┴──────────┘
             │
             ▼
         resources (structure + all of the above)
             │
             ▼
         explanation
```

Per-task models are conditional predictors: each declares which upstream targets it consumes as input features. **Inference orchestration** — walking this DAG, respecting `supported_scope`, and stitching predictions into a coherent input deck — is the responsibility of `goldilocks-core`, not this package.

### 4.3 The `kpoints` task in detail (Phase 1 active)

Summary; full version in `tasks/kpoints/__init__.py`.

- **Target**: `kindex` (1-indexed integer) — row index in `goldilocks_core.kmesh.build_kmesh_entries(structure)`.
- **Why kindex (and not k_line_density)**: goldilocks-data's schedule is mesh-transition-regular, so `kindex ↔ mesh` is 1-to-1, no plateau ambiguity. Per-structure semantic differences are recoverable from structure features (reciprocal lattice lengths). Continuous alternatives (k_line_density, k_distance) suffer from interval ambiguity because mesh is intrinsically discrete.
- **Label**: smallest kindex tagged `just_right` for each (structure, pseudo).
- **Loss**: start with MAE on kindex (simple regression, round to integer at inference); upgrade to ordinal cross-entropy or earth-mover's distance if off-by-many errors hurt downstream metrics.
- **Primary metric**: exact-match accuracy on kindex; secondary "within ±1"; downstream truth is **convergence-fraction** (does the predicted mesh actually converge).
- **Reporting slices**: `crystal_system`, `is_metal_guess`, `contains_heavy`, `cell_volume` bins, `anisotropy_ratio = max(b_i)/min(b_i)`, `pseudo_family`.
- **Manifest must record**: `schedule_generator = "goldilocks_core.kmesh.build_kmesh_entries"`, `schedule_generator_version` (git SHA), `schedule_max_index`. A change to the generator shifts kindex semantics and requires retraining.
- **Phase 1 model line-up**: classical (XGBoost with ordinal objective on kindex) baseline plus a gnn (CGCNN-style) model trained in parallel; nn as a fallback if either underperforms. Both lines may consume LLM-derived structure embeddings (frozen encoder, see §6) as additional features when the embedding cache is populated.

## 5. Models

### 5.1 tasks × models matrix

|  | classical | nn | gnn | llm |
|---|---|---|---|---|
| `kpoints` | ✓ baseline | ✓ | ✓ strong candidate | — |
| `ecutwfc` | ✓ | ✓ | ✓ | — |
| `smearing` | ✓ | ✓ | ✓ | — |
| `pseudo` | ✓ baseline | ✓ | ✓ | — |
| `xc` | ✓ | ✓ | ✓ | — |
| `resources` | ✓ baseline | ✓ | possibly | — |
| `explanation` | — | — | — | ✓ |

### 5.2 Algorithm families

| Subpackage | Frameworks | Optional-dep group | Phase 1 |
|---|---|---|---|
| `classical` | scikit-learn, XGBoost, LightGBM | (core) | ✓ start here |
| `nn` | PyTorch + Lightning | `nn` | as needed |
| `gnn` | PyTorch Geometric | `gnn` | ✓ alongside classical |
| `llm` | HuggingFace transformers + PEFT | `llm` | Phase 3+ |

Subpackage boundaries align with optional-dependency groups so a deployment using only classical baselines does not import torch / PyG / transformers.

### 5.3 No abstract base class yet

Models do not inherit from a common `Model` ABC. The interface emerges from repetition once at least two implementations exist for the same task. Convention (not enforced):

- Constructor accepts a hyperparameter dict (Hydra-friendly).
- `fit(X, y, sample_weight=None, **kwargs)`
- `predict(X)`
- `save(path)` / `load(path)` using the framework's native serialisation (joblib, torch state_dict, safetensors).

`registry.exporter` relies on this convention to write artefacts uniformly.

## 6. Data layer

The only place in the repo coupled to goldilocks-data's Parquet schema. Downstream code (`tasks/`, `models/`) consumes plain DataFrames / arrays / tensors and is decoupled from the source format.

| Module | Responsibility |
|---|---|
| `data/loaders.py` | Parquet IO, snapshot pinning, dataset hashing for reproducibility |
| `data/features.py` | Feature engineering on top of `goldilocks_core.infer_features` `StructureFeatures` |
| `data/embeddings.py` | LLM-derived structure embeddings via a frozen text encoder (CIF / POSCAR / textualised description -> vector); cached per `(structure_hash, encoder_checkpoint_hash)` so the LLM is called once per structure |
| `data/splits.py` | Train / val / test partitioning: random, element-disjoint, spacegroup-disjoint, stratified by convergence label |
| `data/datasets.py` | Framework-specific wrappers (numpy for sklearn / xgboost; torch `Dataset`; PyG `InMemoryDataset`) |

**Splits philosophy**: a model's "test-set accuracy" is meaningless if test rows share elements / chemistry with train. Default to **element-disjoint** for `kpoints` to test true generalisation across the periodic table; **spacegroup-disjoint** for tasks where geometry dominates (smearing, kpoints anisotropy slices).

**LLM-encoder embeddings live in `data/`, not `models/`**. The text encoder is **frozen** in Phase 1, so the embedding is a feature, not a model. Downstream classical / gnn / nn models consume it as a regular input column or as a graph-level attribute. The encoder checkpoint hash enters the manifest's `feature_schema` so inference reconstructs the same input. If the encoder is ever fine-tuned jointly with a downstream head (Phase 2+ at the earliest, only if frozen embeddings underperform), the combined object becomes a hybrid model and migrates to a hybrid model subpackage; we will not pre-create that subpackage.

## 7. Configs (Hydra)

*Skeleton in place; first training run validates the wiring.*

Four axes, composed via Hydra:

```
configs/
├── data/         # parquet snapshot path, feature subset, split kind
├── task/         # which task: kpoints / ecutwfc / ...
├── model/        # which model class: xgboost / mlp / cgcnn / ...
└── experiment/   # composes the four above + overrides
```

Run example: `gm train experiment=kpoints_xgb_v1`. Configs version-controlled in git; experiments reproducible from the config alone.

## 8. Tracking (MLflow)

Phase 1: local MLflow store (`mlruns/`), single user, offline-friendly. Adapter in `tracking/logger.py` so we can swap to wandb later without touching training code.

Each run logs:
- All Hydra config parameters
- Validation metrics + slice tables
- Dataset hash (so we know which Parquet snapshot trained the model)
- `goldilocks_models.__version__`
- git SHA at run time

## 9. Registry — handoff to `goldilocks-core`

A trained model becomes a versioned artefact directory:

```
artifacts/<task>/<version>/
├── model.<ext>          # joblib / pt / safetensors / onnx
├── feature_schema.json  # input columns and types
├── manifest.json        # see schema below
└── card.md              # model card: training data, limitations, reporting slices
```

### 9.1 Manifest schema (Phase 1 draft)

```json
{
  "task": "kpoints",
  "version": "0.1.0",
  "created_at": "2026-05-07T00:00:00Z",
  "git_sha": "...",
  "package_version": "goldilocks_models 0.1.0",
  "supported_scope": {
    "code": ["QE"],
    "calc_type": ["SCF"]
  },
  "dependencies": {
    "upstream_targets": ["pseudo"]
  },
  "feature_schema": { "...": "..." },
  "metrics": {
    "validation": {
      "mae_kindex": 1.2,
      "within_1": 0.89,
      "convergence_fraction": 0.94
    }
  },
  "data": {
    "parquet_hash": "sha256:...",
    "split": "element_disjoint",
    "n_train": 4000, "n_val": 500, "n_test": 500
  },
  "schedule": {
    "generator": "goldilocks_core.kmesh.build_kmesh_entries",
    "generator_version": "goldilocks_core 0.x.y",
    "max_index": 30
  },
  "model_uri": "model.joblib"
}
```

`goldilocks-core` reads `manifest.json` to load and use the model. We do not pip-publish; this decouples model versions from core releases.

### 9.2 Model-class-specific manifest fields

The §9.1 schema lists fields that every artefact carries. Some fields appear only for certain model classes:

| Field | Required for | Content |
|---|---|---|
| `schedule` | `kpoints` task (any model class) | generator + version + max_index (already shown in §9.1) |
| `graph_construction` | gnn-class models | cutoff radius, max neighbours, image convention, edge features |
| `tokeniser` | llm-class models | tokeniser checkpoint + max sequence length (Phase 3+) |

Example `graph_construction` block:

```json
"graph_construction": {
  "cutoff_radius_angstrom": 5.0,
  "max_neighbours": 12,
  "include_periodic_images": true,
  "image_convention": "minimum_image",
  "self_loops": false,
  "edge_features": ["distance", "bond_vector"]
}
```

These fields are omitted (or set to `null`) for model classes that do not need them. `goldilocks-core` reads them at inference time to reconstruct the graph identically to training, or to load the matching tokeniser.

## 10. CLI

*Sketched; not yet built.*

```
gm train experiment=<config>     # train per Hydra config; logs to MLflow
gm eval  experiment=<config>     # evaluate a registered model on a held-out split
gm predict --task <t> --version <v> --input <parquet>
gm register --run-id <id> --version <v>
```

Implementation: Typer + Hydra. Entry point declared in `pyproject.toml`'s `[project.scripts]` as `gm = goldilocks_models.cli.main:app`.

## 11. Repository layout

```
3-goldilocks-models/
├── pyproject.toml
├── README.md
├── docs/
│   └── PLAN.md                   (source of truth; this file)
├── configs/                      (Hydra)
│   ├── data/  task/  model/  experiment/
├── src/goldilocks_models/
│   ├── __init__.py
│   ├── data/                     loaders / features / splits / datasets / embeddings
│   ├── tasks/                    kpoints / ecutwfc / smearing / pseudo / xc / resources / explanation
│   ├── models/                   classical / nn / gnn / llm
│   ├── training/                 trainers / callbacks / losses
│   ├── evaluation/               metrics / slices / reports
│   ├── tracking/                 MLflow adapter
│   ├── registry/                 manifest / exporter
│   ├── cli/                      Typer
│   └── utils/
├── scripts/                      one-off scripts
├── notebooks/                    EDA, ablation
├── experiments/                  gitignored: ckpts, logs, MLflow store
└── tests/
```

## 12. Roadmap

| Phase | Active tasks | Driver |
|---|---|---|
| 1 | `kpoints`; classical + gnn lines, both with optional frozen-LLM encoder features | k-mesh sweeps in goldilocks-data |
| 1.5 | `pseudo`, `resources` | pseudo benchmark + resource regressor |
| 2 | `ecutwfc`, `smearing` | ecutwfc / smearing sweeps |
| 2+ | `kpoints` cross-code (VASP) | second-code data |
| 3+ | `xc`, `explanation` | benchmark sets + LLM data |

## 13. Open questions

| Question | Resolves when |
|---|---|
| `pseudo` task label definition (cost-to-converge vs reference agreement vs Pareto) | Phase 1.5 sweeps land |
| `ecutwfc` loss form | first ecutwfc-sweep dataset |
| `smearing` joint vs separate (kind / width) | first smearing-sweep dataset |
| `xc` benchmark set choice (Matbench vs WBM vs other) | Phase 3 design phase |
| Cross-HPC transfer for `resources` | second-cluster data |
| Lightning vs custom trainer for `nn` | first nn implementation |

## 14. References

- `2-goldilocks-data/PLAN.md` — sibling data contract
- `4-goldilocks-core/src/goldilocks_core/kmesh.py` — kindex schedule generation
- UKRI grant EP/Z530657/1
- PseudoDojo: <http://www.pseudo-dojo.org/>
- Materials Cloud MC3D: <https://www.materialscloud.org/discover/mc3d>
