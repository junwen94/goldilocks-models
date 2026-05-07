"""resource: predict wallclock, peak memory, and MPI task count for an SCF
job, given structure and all DFT parameters.

Target          : (wallclock_seconds, peak_RAM_GB, ntasks). Multi-output.
Label source    : actual job statistics extracted from AiiDA Node.extras
                  and QE output files (PwBaseWorkChain finalizers).
Loss            : MAE on log-transformed targets (wallclock and RAM are
                  heavy-tailed; log-space loss avoids huge jobs dominating).
Primary metric  : MAPE per output; tail-bucket MAPE (top 10% longest jobs
                  reported separately, since underestimating those wastes
                  the most HPC allocation).
Reporting slices: n_atoms bins, n_electrons bins, kpoints kindex bin,
                  ecutwfc bin, pseudo_family.
Scope           : code={QE}, calc_type={SCF}, HPC={SCARF} for Phase 1.
Status          : Phase 1.5 / Phase 2 placeholder.
Inputs          : structure features + pseudo + ecutwfc + smearing + kpoints.
                  Prefer log(k_pra) over raw k_pra as a feature — k_pra
                  grows as n^3 and log compresses the heavy tail.
Dependencies    : pseudo, ecutwfc, smearing, kpoints  (every upstream
                  target is a feature; resource is the leaf of the DAG).

Notes
-----
HPC scope matters: a wallclock model trained on SCARF Intel nodes will not
transfer cleanly to ARCHER2 / Iridis. The manifest's supported_scope must
also carry the HPC site once we go cross-cluster (Phase 2+).
"""
