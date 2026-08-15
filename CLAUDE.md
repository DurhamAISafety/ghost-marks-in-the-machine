# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Research repo adapting Google DeepMind's **SynthID** text watermark to Python code, with Bayesian
detectors classifying watermarked vs. unwatermarked code. APPS dataset, CodeGemma-7b-it.

- **Paper (context):** https://docs.google.com/document/d/1zhsXX0CTYF6j4Q1P58gVwaK_BNUomDumJVAeCyhbii0/edit
- **Submission:** https://apartresearch.com/project/ghost-marks-in-the-machine-a-critical-review-of-synthid-for-code-provenance-monitoring-ov2c

## Commands

# Generation and detection are MUTUALLY EXCLUSIVE extras (see pyproject [tool.uv] conflicts):
# synthid-text hard-pins torch==2.4.0, so anything loading/training detectors is stuck there,
# while generation is free to run modern torch. Sync one OR the other, never both.
uv sync --extra generation      # CodeGemma generation + g-score: modern torch + transformers 4.x
uv sync --extra detection       # load/train/score Bayesian detectors: synthid-text -> torch 2.4.0
uv sync --extra dev             # generation + reporting + nlp (generation-side; excludes detection)

# Generation + g-score + execution over APPS (GPU, large memory; needs HF_TOKEN in .env)
python scripts/pipeline.py                          # -> outputs/results/results.json, report.html

# Train one Bayesian detector per n-gram length {2,5,10} (JAX)
python scripts/train_bayesian_detector.py --train   # -> outputs/models/bayesian_detector_ngram{n}.pkl
python scripts/train_bayesian_detector.py --score   # reads outputs/results/results.json

python scripts/visualize_results.py                 # plots into outputs/reports/
python apps/web_app.py                              # Flask UI on :5001 (falls back to a stub detector)
```

**No pytest suite** — the red-team runner and detection demos are standalone scripts, run directly:
`python scripts/run_red_team_tests.py`, `python scripts/interactive_detection.py`. No lint config.

Environment: `cp configs/.env.example .env` and set `HF_TOKEN` (gated CodeGemma access).

## Architecture

The end-to-end flow spans three files that must be read together:

1. **Generation** (`src/model_utils.py`) — PyTorch/transformers. `generate_code()` wraps CodeGemma
   with `SynthIDTextWatermarkLogitsProcessor`. The watermark scheme is the **n-gram context length
   `ngram_len` ∈ {None, 2, 5, 10}**, where `None` = no watermark (the negative class).
   `compute_g_score()` returns the mean g-value — the fast, training-free detection signal.
   Watermark keys and sampling params are centralised in `src/watermark_config.py`
   (`make_synthid_processor()`), shared by generation, detection and training — not hard-coded per file.

2. **Detection** (`src/bayesian_detector.py`) — **JAX/Flax**, vendored from GDM's SynthID repo. This is
   the second, learned signal: `BayesianDetector.train_best_detector(...)` fits a per-n-gram classifier
   on g-values (cross-entropy + L2). Note the framework split: generation is Torch, detection is JAX.

3. **Execution/grading** (`src/execution_utils.py`) — `run_code_safely()` runs generated code in a
   subprocess with a timeout and a **restricted `__builtins__` dict**. ⚠️ That dict omits `__import__`,
   so any generated code with an `import` is classified as a Runtime Error — a known confound in the
   error-rate results. It also grades correctness against **only the first** APPS test case.

`scripts/pipeline.py` is the orchestrator: filters APPS to `difficulty == 'interview'`, loops
`ngram_len ∈ {None,2,5,10}` × 3 attempts per problem, and writes `results.json` (the input every
downstream script expects) plus an HTML report via `src/report_generator.py`.

### Pickled-detector gotcha
Trained `.pkl`s bundle a **Torch** SynthID logits_processor inside the JAX detector. Loading on CPU
requires `CPU_Unpickler` (the single shared copy in `src/pickle_utils.py`, imported everywhere), which
remaps CUDA→CPU storage and rewrites old root-level module paths to `src.*`. Load detectors through
`WatermarkDetector` / that unpickler, never plain `pickle.load`.

`analysis/` is a separate, self-contained n-gram / TF-IDF analysis on natural-language SynthID data —
not wired into the code pipeline.

Large artifacts (models, results, HTML report, NLP data) are **not in git** — they're hosted on
HF: `Theosdoor/ghost-marks-artifacts`. `outputs/{models/*.pkl,results/*.json,reports/*.html}` are
gitignored; fetch them into the repo layout with `hf download Theosdoor/ghost-marks-artifacts
--repo-type dataset --local-dir .` before running anything that reads `results.json`.
