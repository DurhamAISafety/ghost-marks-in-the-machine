# Ghost Marks in the Machine

Evaluates SynthID code watermarking on the APPS dataset with Bayesian detectors for watermark identification.

**Link:** [Apart Research Submission](https://apartresearch.com/project/ghost-marks-in-the-machine-a-critical-review-of-synthid-for-code-provenance-monitoring-ov2c)

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/DurhamAISafety/ghost-marks-in-the-machine.git
cd ghost-marks-in-the-machine
uv sync

# Optional features
uv sync --extra dev           # All extras
uv sync --extra reporting     # Reporting only
uv sync --extra training      # Training only
uv sync --extra nlp           # NLP analysis tools
```

### Environment Setup

Copy `configs/.env.example` to `.env` and add your Hugging Face token:

```bash
cp configs/.env.example .env
# Edit .env and add your HF_TOKEN
```

## Quick Start

### 1. Generate Watermarked Code

```bash
python scripts/pipeline.py
```

Generates code with different watermark configurations (ngram_len: None, 2, 5, 10) and evaluates correctness. By default it iterates the **entire APPS `interview` split** (thousands of problems); edit `NGRAM_LENS` / add a slice in `scripts/pipeline.py` to cap the run.

**Output:** `outputs/results/results.json`, `outputs/reports/report.html`

**Note:** Requires GPU with significant memory.

### 2. Train Detectors

```bash
python scripts/train_bayesian_detector.py --train
```

Trains three separate Bayesian classifiers (one per ngram length).

**Output:** `outputs/models/bayesian_detector_ngram{2,5,10}.pkl`

### 3. Score Samples

```bash
python scripts/train_bayesian_detector.py --score
```

Evaluates samples using trained detectors.

**Output:** `outputs/results/bayesian_scores.json`

> **Note:** `--score` (and re-training) reads `outputs/results/results.json`. That file and the
> trained `.pkl` detectors are large and are **not** committed to git — download them first (see
> [Data & Artifacts](#data--artifacts)).

## Detection API

### Simple Detection

```python
from src.detector_utils import WatermarkDetector

detector = WatermarkDetector(ngram_len=5)
result = detector.detect(code, threshold=0.5)
print(f"Watermarked: {result['is_watermarked']}, Score: {result['score']:.4f}")
```

### Web Interface

```bash
python apps/web_app.py
```

Open http://localhost:5001 for an interactive UI with real-time detection. (Falls back to a
pseudo-random stub detector if the real model/weights can't be loaded — the response is flagged
with `"stub": true`.)

## Key Results

Detectors evaluated on 67 APPS `interview` problems (the scored subset; the paper reports the
watermarking scheme across 1000 APPS prompts):

| Detector | Watermarked Mean | Unwatermarked Mean | Separation |
|----------|------------------|-------------------|------------|
| Ngram=2  | 0.27 | 0.17 | +0.10 |
| Ngram=5  | 0.49 | 0.24 | **+0.25** |
| Ngram=10 | 0.47 | 0.21 | **+0.26** |

**Findings:** Longer n-grams (5, 10) provide stronger watermark signals and better discrimination.

### Visualize Results

```bash
python scripts/visualize_results.py
```

Generates performance plots in `outputs/reports/`.

## G-Score vs Bayesian Score

- **G-score**: Fast statistical measure from token-level watermark signals
- **Bayesian score**: Learned classifier trained on specific data, typically achieves better separation

## Project Structure

```
ghost-marks-in-the-machine/
├── src/                    # Core library modules
│   └── red_team/           # Adversarial attack transforms (importable)
├── scripts/                # Pipeline, runners, and demo scripts
├── apps/                   # Web application
│   ├── web_app.py
│   └── web_interface/
├── configs/                # Configuration files (.env.example)
├── outputs/                # Generated outputs (models, reports, results)
└── analysis/               # Standalone n-gram / TF-IDF analysis (not wired into the pipeline)
```

## Data & Artifacts

Trained detectors (`outputs/models/*.pkl`), generation results (`outputs/results/*.json`, the
HTML report), and the NLP analysis datasets are large and are hosted on Hugging Face rather than
in git: **[Theosdoor/ghost-marks-artifacts](https://huggingface.co/datasets/Theosdoor/ghost-marks-artifacts)** (public).

```bash
# Download models + results into the repo layout (requires the `hf` CLI: pip install huggingface-hub)
hf download Theosdoor/ghost-marks-artifacts --repo-type dataset --local-dir .
```

The paths mirror the repo layout, so this restores `outputs/` and the `analysis/` data files in place.

## Reproducibility Notes

- **Sample count:** `scripts/pipeline.py` runs over the full APPS `interview` split; the paper reports
  1000 prompts and the committed detector evaluation covers 67 problems. Pin the count in the script
  for an exact repro.
- **Environments:** a committed `uv.lock` lives on the `theo2` branch (pending merge) for pinned installs.
- **CI provenance check:** the GitHub Actions workflow that flags watermarked code on push/PR lives on
  the `ai_code_workflow` branch (`.github/workflows/check_files.yml`), pending merge to `main`.

## References

- [SynthID Text Repository](https://github.com/google-deepmind/synthid-text)
- [APPS Dataset](https://huggingface.co/datasets/codeparrot/apps)
- [CodeGemma Model](https://huggingface.co/google/codegemma-7b-it)
