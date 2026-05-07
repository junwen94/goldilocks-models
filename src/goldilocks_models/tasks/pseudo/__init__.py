"""pseudo: recommend pseudopotential family + version + precision.

Target          : pseudo_family (categorical), e.g.
                    PseudoDojo/0.4/PBEsol/SR3plus/standard
                    PAW-JTH-1.0-PBEsol
Label source    : TBD — candidate definitions:
                    * "best pseudo per structure" by lowest cost-to-converge
                    * agreement with a reference (e.g. all-electron, SSSP)
                    * pareto frontier of cost vs accuracy.
                  Decision deferred until Phase 1.5 sweeps land.
Loss            : Cross-entropy.
Primary metric  : top-1 accuracy; top-3 accuracy as a softer signal.
Reporting slices: contains_lanthanide, contains_actinide, contains_heavy,
                  likely_magnetic, crystal_system.
Scope           : code={QE}, calc_type={SCF}
Status          : Phase 1.5 placeholder.
Inputs          : structure features only (no upstream targets).
Dependencies    : none (root of the DAG after structure).

Notes
-----
Choosing a PseudoDojo / PAW-JTH family commits to a specific XC functional
(PBEsol in Phase 1). The xc/ task is therefore inactive by default;
it only fires when the user constrains pseudo upstream and asks for an
xc recommendation under that constraint.
"""
