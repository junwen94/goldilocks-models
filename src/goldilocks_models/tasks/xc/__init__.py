"""xc: recommend exchange-correlation functional, conditional on user
constraints that fix pseudo upstream.

Target          : xc functional (categorical), e.g. PBE, PBEsol, SCAN,
                  r2SCAN, LDA, optB88-vdW, ...
Label source    : TBD — likely benchmark-set agreement (lattice constants,
                  formation energies, band gaps) on Matbench / WBM /
                  similar reference sets.
Loss            : Cross-entropy.
Primary metric  : top-1 accuracy on a held-out benchmark slice.
Reporting slices: bonding type (ionic / covalent / metallic / vdW),
                  band-gap class, contains_heavy.
Scope           : code={QE}, calc_type={SCF}
Status          : Phase 3+ placeholder. Inactive by default — choosing a
                  pseudo already fixes XC for PseudoDojo and PAW-JTH.
Inputs          : structure features + user-supplied pseudo constraint.
Dependencies    : user constraint (not an upstream task).
"""
