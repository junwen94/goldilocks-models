"""CLI: Typer-based command-line interface exposed as ``gm``.

Entry point declared in ``pyproject.toml``'s ``[project.scripts]``:

    gm = "goldilocks_models.cli.main:app"

Surface (PLAN.md §10):

    gm train experiment=<config>
    gm eval  experiment=<config>
    gm predict --task <t> --version <v> --input <parquet>
    gm register --run-id <id> --version <v>

Implementation
--------------
* Typer for argument parsing and the help / version surface.
* Hydra for composing experiment configs (``data`` x ``task`` x
  ``model`` x ``experiment`` overrides). Hydra is invoked from inside
  Typer commands rather than driving the entry point — Typer owns the
  ``gm`` namespace and Hydra owns the config-resolution machinery.
* ``rich`` for human-readable progress and tables in the terminal;
  log lines are also routed through ``tracking.logger`` to MLflow.

Phase 1 layout: a single ``main.py`` holds all four commands. If a
command grows beyond ~50 lines it moves to its own file
(``train.py``, ``eval.py``, ``predict.py``, ``register.py``) and
``main.py`` only composes the Typer app. The single-file form is the
default until that growth pressure arrives.

Submodules
----------
main : Typer ``app`` plus the four Phase 1 commands. The ``app``
       symbol is what ``[project.scripts]`` resolves to.
"""
