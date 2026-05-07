"""kpoints: recommend k-point density from crystal structure.

Target          : kindex (1-indexed integer) — the row index in the
                  per-structure k-mesh schedule produced by
                  ``goldilocks_core.kmesh.build_kmesh_entries(structure)``.
                  The schedule is mesh-transition-regular: each kindex
                  corresponds to a unique Monkhorst-Pack triple, so the
                  mapping kindex <-> mesh is 1-to-1 within a structure.
                  Why not k_line_density (continuous)? Because mesh is
                  intrinsically discrete: any continuous 1-D coordinate
                  (k_line_density, k_distance, jarvis-style L) is
                  many-to-one onto mesh and introduces label-interval
                  ambiguity. The mesh-transition kindex avoids that
                  ambiguity at the cost of a per-structure semantic —
                  which is fine because structure features are inputs.
Label source    : the smallest kindex whose mesh is tagged "just_right"
                  in goldilocks-data's rolling convergence sweep, for
                  each (structure, pseudo) row.
Loss            : start with MAE on kindex (continuous regression, round
                  to nearest valid integer at inference). If the off-by-1
                  pattern matters, switch to ordinal cross-entropy or
                  earth-mover's distance over the kindex range, which
                  penalises off-by-many more than off-by-one.
Primary metric  : exact-match accuracy on kindex; "fraction within +/-1"
                  as a domain-friendly secondary metric. Convergence-
                  fraction (predicted mesh actually converges) is the
                  ultimate downstream metric and should also be tracked.
Reporting slices: crystal_system, is_metal_guess, contains_heavy,
                  cell_volume bins, anisotropy ratio (max(b_i)/min(b_i)),
                  pseudo_family.
Scope           : code={QE}, calc_type={SCF}
Status          : Phase 1 active.
Inputs          : structure features + pseudo_family. Structure features
                  must include the three reciprocal-lattice lengths
                  (b1, b2, b3) so the model can implicitly resolve the
                  per-structure schedule.
Dependencies    : pseudo  (the convergence label is per-pseudo, so the
                  model is conditional on the chosen pseudopotential
                  family; in Phase 1 core supplies a default pseudo
                  since the pseudo task is not yet trained).

Notes
-----
Inference handoff to goldilocks-core:
    predicted_kindex --> goldilocks_core.kmesh.build_kmesh_entries(structure)
                         [predicted_kindex - 1].mesh
The schedule is recomputed deterministically from the structure at
inference time; we do not ship the schedule with the model artefact.

Manifest fields:
    schedule_generator         : "goldilocks_core.kmesh.build_kmesh_entries"
    schedule_generator_version : version pin or git SHA of goldilocks-core
                                 at training time
    schedule_max_index         : the ``max_index`` argument used when
                                 generating candidate k-distances
                                 (default 30; affects the upper bound of
                                 the kindex range)

A change to the schedule generator (different max_index, different
candidate-distance formula, deduplication rule) shifts kindex semantics
and requires retraining.

Predicting beyond the structure's actual schedule length is invalid
output. Inference must clip predictions to ``len(build_kmesh_entries)``
and surface a "schedule too short" warning if the prediction saturates
at the upper bound.
"""
