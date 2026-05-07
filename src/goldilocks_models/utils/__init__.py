"""Utilities: cross-cutting helpers that do not fit any other subpackage.

The ``utils`` subpackage is intentionally narrow. If a helper is
specific to one subpackage's concern (data loading, model fitting,
evaluation), it belongs there, not here. Things that genuinely live
across the whole package — hashing, git inspection, path resolution —
land here.

Submodules
----------
hashing : Stable hashes for files, dicts, structures, and datasets.
          Used by the data layer (snapshot pinning), the embeddings
          cache (per-structure cache key), and the registry (manifest
          ``dataset_hash`` field). Algorithm: SHA-256 over a
          canonicalised byte representation.
git     : Git inspection helpers — current commit SHA, dirty-tree
          flag, branch name. Used by tracking (run tags) and registry
          (manifest ``git_sha`` field). Falls back gracefully when
          the working directory is not a git checkout (returns
          sentinel values rather than raising).

Convention
----------
``utils`` modules have **no project-internal imports**. They are
leaf-level helpers, importable from anywhere without circularity
risk. If a utility starts importing from ``goldilocks_models.data``
or similar, that is a sign the helper belongs in that subpackage
instead.
"""
