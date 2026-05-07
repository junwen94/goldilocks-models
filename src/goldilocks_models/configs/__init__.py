"""HOW TO USE THIS PACKAGE
export SNAPSHOT_ROOT=/path/to/your/snapshots
export GOLDILOCKS_CORE_SHA=$(cd ../4-goldilocks-core && git rev-parse HEAD)

gm train experiment=kpoints_xgb_baseline

# Override at command line:
gm train experiment=kpoints_xgb_baseline task.loss=ordinal_ce model.max_depth=12

# Multirun (sweep):
gm train --multirun experiment=kpoints_xgb_baseline model.max_depth=4,8,12
"""
