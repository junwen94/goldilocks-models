"""LLM models: language-model-based explainers and dialog wrappers.

Serves the ``goldilocks_models.tasks.explanation`` task — turning a
structure plus its recommended DFT inputs into a natural-language
explanation for a DFT user.

Stack: HuggingFace transformers + PEFT (LoRA / QLoRA) plus tokenisers,
datasets, accelerate. Optional dependency group ``llm`` — install with
``uv sync --extra llm``.

Phase positioning
-----------------
Phase 3+ placeholder. Phases 1 / 1.5 / 2 ship recommendations without
natural-language explanation; goldilocks-core surfaces the raw
recommendation and tabular metadata. This subpackage exists so prompt
assets, fine-tuning code, and evaluation harnesses have a home when a
coherent explanation dataset becomes available.

Deployment patterns (decision deferred to Phase 3 design)
---------------------------------------------------------
1. **Pure prompt engineering** — no fine-tuning; a prompt template
   plus an optional retrieval index over canonical-case explanations
   wrapped around a hosted or locally-served base model.
2. **Supervised fine-tuning (SFT)** with LoRA / QLoRA on
   (recommendation, expert explanation) pairs. Produces a small
   adapter on top of a base model.
3. **Preference optimisation** (DPO / IPO / KTO) on top of an SFT
   adapter, using human preference pairs over candidate explanations.

Planned modules (created on demand)
-----------------------------------
prompts     : versioned prompt templates (system / user / few-shot).
sft         : LoRA / QLoRA supervised fine-tuning runner.
preference  : DPO / IPO / KTO preference-optimisation runner.
generation  : inference-side wrapper exposing ``predict(X) -> str``
              with decoding-strategy hyperparameters
              (temperature, top-p, max tokens).
retrieval   : optional small case-base retrieval index for the
              pure-prompt pattern.

Convention
----------
Mirrors the other model subpackages (``__init__(**hyperparams)``,
``fit / predict / save / load``). For the pure-prompt pattern, ``fit``
is a no-op that records the prompt and retrieval hashes. ``save(path)``
/ ``load(path)`` writes a directory containing adapter weights
(safetensors), the tokeniser checkpoint, the prompt template, the
optional retrieval index, and a ``hyperparams.json`` sidecar.

The manifest's ``tokeniser`` block (see PLAN.md §9.2) records the
tokeniser checkpoint hash and maximum sequence length the artefact
was trained / prompted against — both must match at inference time.

Evaluation
----------
``explanation`` outputs are free-form text, so exact-match metrics do
not apply. The standard set (implemented in
``goldilocks_models.evaluation``):
  * faithfulness  — references only true facts about the
    recommendation and the structure.
  * completeness  — covers all upstream tasks that drove the
    recommendation.
  * calibration   — says so when uncertain.
LLM-as-judge against a stronger held-out model is the default cheap
proxy; periodic human evaluation is ground truth.
"""
