"""GNN models: crystal-graph neural networks via PyTorch Geometric.

This subpackage hosts neural networks that treat a crystal as a graph
(atoms = nodes, bonds / radius cutoffs = edges). Non-graph networks
(MLP, Set Transformer) live in ``goldilocks_models.models.nn``.

Stack: PyTorch + Lightning + PyTorch Geometric. Optional dependency
group ``gnn`` — install with ``uv sync --extra gnn``.

Companion-package note
----------------------
Some PyG operators require platform-specific compiled wheels:
``torch-scatter``, ``torch-sparse``, ``torch-cluster``. These are
*not* part of the ``gnn`` extra because their wheels depend on the
exact CPU / CUDA / torch-version combination. Install them manually
when needed::

    uv pip install torch-scatter torch-sparse torch-cluster \\
        -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

PyG >= 2.5 ships pure-PyTorch fallbacks for most common operators, so
basic CGCNN / SchNet flows work without the companions. Equivariant
models (NequIP / MACE / Allegro) typically need them.

-When to reach for gnn
----------------------
-* Tasks whose target is dominated by **local chemistry** (which atoms
-  bond to which) more than by global cell shape — pseudo recommendation
-  is the canonical example.
-* Long-range structural features (anisotropy, cell volume) interact
-  with local chemistry; ``kpoints`` likely benefits here too.
-* You want a learned structure embedding that respects E(3) symmetry
-  (rotation / translation / reflection invariance or equivariance).
+Role in Phase 1
+---------------
+Phase 1 trains a gnn-class model on the ``kpoints`` task in parallel
+with the classical baseline (CGCNN-style is the default starting
+architecture). The kpoints kindex target depends jointly on:
+
+* **local chemistry** (which atoms bond to which) — encoded naturally
+  by graph message passing,
+* long-range structural anisotropy (cell shape, reciprocal-lattice
+  spread), and
+* pseudopotential family (passed as an extra non-graph node feature
+  or as a global graph attribute).
+
+The classical baseline handles the second and third axes directly via
+tabular features but cannot represent local chemistry without
+hand-crafted descriptors. The gnn line is the channel for that.
+
+Beyond Phase 1, the same machinery serves tasks dominated by local
+chemistry such as ``pseudo`` and (later) ``ecutwfc`` / ``smearing`` —
+pseudopotential recommendation is the canonical GNN use case in this
+stack.


Architecture families (created on demand)
-----------------------------------------
cgcnn   : Crystal Graph CNN (Xie & Grossman 2018). Strong, simple
          baseline; treats edges by binned distance features.
schnet  : Continuous-filter convolutions (Schuett et al. 2017). Smooth
          distance dependence; well-tested on energies.
alignn  : Atomistic Line-Graph NN (Choudhary & DeCost 2021). Adds a
          bond-angle line graph; better when geometry matters strongly.
mace    : MACE-style equivariant message passing (Batatia et al. 2022).
          State-of-the-art on energies / forces; heavier install
          (needs torch-scatter and e3nn).

Convention
----------
Mirrors ``goldilocks_models.models.nn``: ``fit / predict / save / load``
plus a flat hyperparameter dict for Hydra. The dataset format differs —
gnn models expect a PyG ``InMemoryDataset`` (built in
``goldilocks_models.data.datasets``), not a plain numpy array. The
``fit(X, y, ...)`` wrapper accepts the dataset object as ``X`` and
hides this asymmetry from training / registry code.

``save(path)`` writes a directory containing ``state_dict.pt``,
``hyperparams.json``, and ``cutoff_graph_spec.json`` — the radius
cutoff, max-neighbour count, and periodic-image convention used to
build the input graph at training time. Inference must reconstruct
the graph with the same spec, so this file is part of the artefact
contract.
"""
