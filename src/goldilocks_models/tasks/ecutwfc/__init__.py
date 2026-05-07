"""ecutwfc: recommend plane-wave energy cutoff for wavefunctions (Ry).

Target          : ecutwfc (Ry). ecutrho is derived as dual * ecutwfc with
                  dual fixed by pseudo type (NC=4, USPP/PAW=8-12) and is
                  not predicted here.
Label source    : ecutwfc convergence sweeps (not yet produced — Phase 2+).
Loss            : TBD.
Primary metric  : MAE (Ry); "fraction within +/-5 Ry of converged value".
Reporting slices: pseudo_family, contains_lanthanide, contains_actinide,
                  contains_heavy.
Scope           : code={QE}, calc_type={SCF}
Status          : Phase 2 placeholder — no training labels yet.
Inputs          : structure features + pseudo_family.
Dependencies    : pseudo  (ecutwfc convergence is strongly pseudo-specific:
                  hard NC pseudos need much higher cutoffs than soft PAW).
"""

