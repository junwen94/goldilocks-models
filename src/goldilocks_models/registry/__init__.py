"""Registry: versioned artefact + manifest writer for handoff to
``goldilocks-core``.

Packages a trained model into a self-contained release directory that
``goldilocks-core`` can load at inference time. This is the
**out-the-door** path; ``goldilocks_models.tracking`` is the
**dev-time** path.

Output layout, one directory per (task, version):

    artifacts/<task>/<version>/
    ├── model.<ext>          # joblib / pt / safetensors / onnx
    ├── feature_schema.json  # input columns, types, encoder hashes
    ├── manifest.json        # PLAN.md §9.1 + §9.2 schema
    └── card.md              # model card: training data, limitations,
                             # reporting slices, intended use

The directory is gitignored; it is uploaded out-of-band (shared
filesystem, S3, PSDI deposition) and referenced by URL in the
manifest. ``goldilocks-core`` reads ``manifest.json`` and follows
``model_uri`` to the actual weights.

Boundary with neighbouring subpackages
--------------------------------------
* ``tracking`` writes the per-run development log (every parameter,
  every metric, every candidate model). ``registry`` writes only
  the curated, release-quality artefacts that downstream consumers
  should rely on.
* ``models/<family>/<class>.save(path)`` writes the model's weights;
  the exporter wraps that, adds the manifest / schema / card, and
  writes the directory atomically.
* ``evaluation.reports`` produces the metric-x-slice JSON / Markdown
  that the manifest and the model card consume; the registry does
  not recompute metrics.

Submodules
----------
manifest : Manifest dataclass and JSON serialiser. Implements the
           schema in PLAN.md §9.1 (universal fields) and §9.2
           (model-class-specific fields: ``schedule`` for ``kpoints``,
           ``graph_construction`` for gnn-class, ``tokeniser`` for
           llm-class). Validates required fields per
           (task, model_class) before writing.
exporter : Orchestrates artefact-directory creation: calls the
           model's ``save``, writes ``feature_schema.json``, asks
           ``manifest`` to render ``manifest.json``, fills the model
           card from a template, and writes the whole directory
           atomically (write to ``<dir>.tmp``, then rename).

Convention
----------
A registry write is **versioned and immutable**. Once
``artifacts/<task>/<version>/`` exists, it is not modified — bump the
version (semver) and write a new directory. ``goldilocks-core`` may
have older versions pinned in production; overwriting in place would
silently break those consumers.
"""
