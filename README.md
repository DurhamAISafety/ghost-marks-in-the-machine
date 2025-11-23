# Code Watermarking with SynthID

This repository evaluates SynthID code watermarking on the APPS dataset and provides tools for training Bayesian detectors to identify watermarked code.

## Overview

The project consists of two main components:

1. **Pipeline** (`scripts/pipeline.py`): Generates code with different watermarking configurations and evaluates correctness
2. **Bayesian Detector Training** (`scripts/train_bayesian_detector.py`): Trains machine learning classifiers to detect watermarked code

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd code-watermarking

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_token_here
```

## Pipeline

### What It Does

The pipeline evaluates SynthID text watermarking on code generation:

1. **Loads the APPS dataset** (interview difficulty coding problems)
2. **Generates code** with different watermark configurations:
   - `ngram_len=None` (no watermark - baseline)
   - `ngram_len=2` (watermarked with 2-gram context)
   - `ngram_len=5` (watermarked with 5-gram context)
   - `ngram_len=10` (watermarked with 10-gram context)
3. **Computes g-scores** for each generated sample to measure watermark strength
4. **Executes code** against test cases to verify correctness
5. **Generates results** as JSON and HTML reports

### Usage

```bash
python scripts/pipeline.py
```

**Note:** This requires significant GPU memory. Do not run locally without a GPU.

### Output Files

- `outputs/results/results.json`: Raw results with g-scores, execution status, and generated code
- `outputs/reports/report.html`: Visual HTML report of results

### Configuration

Edit `scripts/pipeline.py` to modify:

```python
NGRAM_LENS = [None, 2, 5, 10]  # Watermarking configurations to test
MODEL_NAME = "google/codegemma-7b-it"  # In src/model_utils.py
```

## Bayesian Detector Training

### What It Does

Trains specialized Bayesian classifiers to detect watermarked code:

1. **Loads samples** from `results.json`
2. **Separates by ngram_len**: Creates training/validation sets for each watermark type
3. **Trains 3 separate detectors**:
   - One for `ngram_len=5` (trained first)
   - One for `ngram_len=2`
   - One for `ngram_len=10`
4. **Saves each detector** as a separate `.pkl` file for reuse
5. **Scores samples** using the appropriate detector for each watermark type

### Why Separate Detectors?

Different `ngram_len` values create different watermark patterns. A detector trained on `ngram_len=5` samples won't properly detect `ngram_len=2` patterns. Training separate detectors ensures optimal accuracy for each watermark configuration.

### Training Detectors

```bash
python scripts/train_bayesian_detector.py --train
```

This will:
- Process your `outputs/results/results.json` file
- Train detectors for ngram lengths 5, 2, and 10 (in that order)
- Save three files:
  - `outputs/models/bayesian_detector_ngram5.pkl`
  - `outputs/models/bayesian_detector_ngram2.pkl`
  - `outputs/models/bayesian_detector_ngram10.pkl`

**Training requirements:**
- At least 5 watermarked and 5 unwatermarked samples per ngram_len
- Recommended: 100+ samples of each type for good performance
- GPU strongly recommended (CPU training is unstable)

### Scoring Samples

After training, score your samples:

```bash
python scripts/train_bayesian_detector.py --score
```

This will:
- Load the trained detectors
- Score each sample in `outputs/results/results.json` with the appropriate detector
- Output scores from 0-1 (0 = likely unwatermarked, 1 = likely watermarked)
- Save detailed results to `outputs/results/bayesian_scores.json`

### Output Format

`outputs/results/bayesian_scores.json` contains:
```json
[
  {
    "problem_id": 0,
    "ngram_len": "5",
    "bayesian_score": 0.8234,
    "g_score": 0.5314,
    "status": "Passed (Wrong Output)"
  }
]
```

## Detecting Watermarks in Custom Code

### Quick Start

To check if a Python code string is watermarked:

```python
from src.detector_utils import WatermarkDetector

# Create detector (automatically uses HF_TOKEN from .env)
detector = WatermarkDetector(ngram_len=5)

# Test some code
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

result = detector.detect(code, threshold=0.5)
print(f"Watermarked: {result['is_watermarked']}")
print(f"Score: {result['score']:.4f}")
print(f"Confidence: {result['confidence']}")
```

**Note:** The detector automatically loads your `HF_TOKEN` from `.env` for authentication.

### Advanced Usage

For better performance when checking multiple samples:

```python
from src.detector_utils import WatermarkDetector

# Create detector instance (reusable)
detector = WatermarkDetector(ngram_len=5)

# Check code
result = detector.detect(code, threshold=0.5)
print(f"Score: {result['score']:.4f}")
print(f"Watermarked: {result['is_watermarked']}")
print(f"Confidence: {result['confidence']}")
```

### API Reference

**`detect_watermark(code, ngram_len=5, threshold=0.5) -> bool`**
- Simple function to check if code is watermarked
- Returns `True` if watermarked, `False` otherwise

**`WatermarkDetector(ngram_len=5)`**
- Class for efficient multiple detections
- Methods:
  - `is_watermarked(code, threshold=0.5) -> bool`
  - `get_score(code) -> float` - Returns score 0-1
  - `detect(code, threshold=0.5) -> dict` - Detailed results

**Example:** Run `python scripts/example_detection.py` for interactive demos.

## Project Structure

```
code-watermarking/
├── src/                          # Core source code
│   ├── bayesian_detector.py      # Bayesian detector implementation
│   ├── detector_utils.py         # Simple detection API
│   ├── model_utils.py            # Model loading and generation utilities
│   ├── execution_utils.py        # Safe code execution
│   └── report_generator.py       # HTML report generation
├── scripts/                      # Executable scripts
│   ├── pipeline.py               # Main evaluation pipeline
│   ├── train_bayesian_detector.py # Detector training script
│   ├── visualize_results.py      # Visualization script
│   ├── example_detection.py      # Detection usage examples
│   ├── score_test_results.py     # Score test results
│   ├── test_samples.py           # Test sample scoring
│   ├── demo_detector.py          # Detector demo
│   ├── find_sample.py            # Find specific samples
│   └── gen_report.py             # Generate HTML report
├── outputs/                      # Generated outputs
│   ├── models/                   # Trained detector models
│   │   ├── bayesian_detector_ngram2.pkl
│   │   ├── bayesian_detector_ngram5.pkl
│   │   └── bayesian_detector_ngram10.pkl
│   ├── results/                  # JSON results
│   │   ├── results.json          # Main pipeline output
│   │   ├── test_results.json     # Test results
│   │   └── bayesian_scores.json  # Bayesian scores
│   └── reports/                  # HTML reports and visualizations
│       ├── report.html
│       ├── detector_performance_overall.png
│       └── detector_performance_by_status.png
├── .env                          # Environment variables (gitignored)
├── .gitignore
├── README.md
└── requirements.txt              # Python dependencies
```

## Workflow

### Full Evaluation Workflow

1. **Run the pipeline** to generate watermarked/unwatermarked code:
   ```bash
   python scripts/pipeline.py
   # Outputs: outputs/results/results.json, outputs/reports/report.html
   ```

2. **Train Bayesian detectors** on the generated data:
   ```bash
   python scripts/train_bayesian_detector.py --train
   # Outputs: outputs/models/bayesian_detector_ngram5.pkl, ngram2.pkl, ngram10.pkl
   ```

3. **Score samples** to evaluate detection accuracy:
   ```bash
   python scripts/train_bayesian_detector.py --score
   # Outputs: outputs/results/bayesian_scores.json
   ```

4. **Analyze results**: Compare g-scores vs Bayesian scores to see which detection method better separates watermarked from unwatermarked code.

## G-Score vs Bayesian Score

- **G-score**: Statistical measure computed directly from token-level watermark signals. Fast but may have limited separation between watermarked/unwatermarked samples.

- **Bayesian score**: Learned classifier trained on your specific data. Requires training but typically achieves better separation and detection accuracy.

## Bayesian Detector Results

### Scoring Methodology

**How detectors are tested:**

Each detector is trained on watermarked code with a specific `ngram_len` plus unwatermarked code. When testing:

- **Detector N only scores:**
  - Code watermarked with `ngram=N` (expects HIGH scores ~0.5-0.9)
  - Unwatermarked code (expects LOW scores ~0.2-0.4)

**Example:** The `ngram=5` detector is tested ONLY on `ngram_len=5` and `ngram_len=None` samples. This specialization ensures each detector focuses on recognizing its specific watermark pattern.

### Performance Summary

Tested on 67 coding problems from `outputs/results/test_results.json`:

| Detector | Watermarked Mean | Unwatermarked Mean | Separation |
|----------|------------------|-------------------|------------|
| Ngram=2  | 0.27 | 0.17 | **+0.10** |
| Ngram=5  | 0.49 | 0.24 | **+0.25** |
| Ngram=10 | 0.47 | 0.21 | **+0.26** |

**Key Findings:**
- ✅ All detectors successfully discriminate watermarked from unwatermarked code
- ✅ Performance is consistent across correct, wrong, and error-producing code
- ✅ **Ngram=5 and ngram=10 show the best separation** (~0.25 difference in means)
- ✅ Longer n-grams provide a stronger watermark signal

### Visualization

Run `visualize_results.py` to generate performance plots:

```bash
python scripts/visualize_results.py
```

This creates:
- `outputs/reports/detector_performance_overall.png`: Box plots showing overall discrimination
- `outputs/reports/detector_performance_by_status.png`: Performance breakdown by code correctness (Correct/Wrong/Error)



## Troubleshooting

### Out of Memory
- Reduce batch size in `pipeline.py`
- Use a machine with more GPU memory
- Process fewer samples at once

### Not Enough Training Data
- Run `scripts/pipeline.py` on more samples
- Reduce the `test_size` parameter in `scripts/train_bayesian_detector.py`

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're using the virtual environment

## References

- [SynthID Text Repository](https://github.com/google-deepmind/synthid-text)
- [APPS Dataset](https://huggingface.co/datasets/codeparrot/apps)
- [CodeGemma Model](https://huggingface.co/google/codegemma-7b-it)