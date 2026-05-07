"""Data layer: load Parquet from goldilocks-data, build features, define splits.

This package is the only part of goldilocks-models that depends on the
goldilocks-data Parquet schema. Downstream code (tasks/, models/) consumes
plain DataFrames / arrays / tensors and is decoupled from the source format.

 Submodules
 ----------
 loaders   : Parquet IO, snapshot pinning, dataset hashing.
 features  : Feature engineering on top of StructureFeatures from goldilocks-core.
 embeddings: LLM-derived structure embeddings (frozen encoder), cached per structure.
 splits    : Train/val/test partitioning (random, element-disjoint, spacegroup-disjoint, etc.).
 datasets  : Task-specific Dataset wrappers (e.g. PyG / torch Dataset for GNNs).
"""
