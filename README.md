# Code Watermarking with SynthID

This repository evaluates SynthID code watermarking on the APPS dataset and provides tools for training Bayesian detectors to identify watermarked code.

## Overview

The project consists of two main components:

1. **Pipeline** (`pipeline.py`): Generates code with different watermarking configurations and evaluates correctness
2. **Bayesian Detector Training** (`train_bayesian_detector.py`): Trains machine learning classifiers to detect watermarked code

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

## Pipeline (`pipeline.py`)

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
python pipeline.py
```

**Note:** This requires significant GPU memory. Do not run locally without a GPU.

### Output Files

- `results.json`: Raw results with g-scores, execution status, and generated code
- `report.html`: Visual HTML report of results

### Configuration

Edit `pipeline.py` to modify:

```python
NGRAM_LENS = [None, 2, 5, 10]  # Watermarking configurations to test
MODEL_NAME = "google/codegemma-7b-it"  # In model_utils.py
```

## Bayesian Detector Training (`train_bayesian_detector.py`)

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
python train_bayesian_detector.py --train
```

This will:
- Process your `results.json` file
- Train detectors for ngram lengths 5, 2, and 10 (in that order)
- Save three files:
  - `bayesian_detector_ngram5.pkl`
  - `bayesian_detector_ngram2.pkl`
  - `bayesian_detector_ngram10.pkl`

**Training requirements:**
- At least 5 watermarked and 5 unwatermarked samples per ngram_len
- Recommended: 100+ samples of each type for good performance
- GPU strongly recommended (CPU training is unstable)

### Scoring Samples

After training, score your samples:

```bash
python train_bayesian_detector.py --score
```

This will:
- Load the trained detectors
- Score each sample in `results.json` with the appropriate detector
- Output scores from 0-1 (0 = likely unwatermarked, 1 = likely watermarked)
- Save detailed results to `bayesian_scores.json`

### Output Format

`bayesian_scores.json` contains:
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

## Project Structure

```
code-watermarking/
├── pipeline.py                    # Main evaluation pipeline
├── train_bayesian_detector.py    # Detector training script
├── visualize_results.py           # Visualization script for detector performance
├── bayesian_detector.py           # Bayesian detector implementation
├── model_utils.py                 # Model loading and generation utilities
├── execution_utils.py             # Safe code execution
├── report_generator.py            # HTML report generation
├── requirements.txt               # Python dependencies
├── results.json                   # Pipeline output (generated)
├── bayesian_detector_ngram*.pkl   # Trained detectors (generated)
└── bayesian_scores.json           # Detection scores (generated)
```

## Workflow

### Full Evaluation Workflow

1. **Run the pipeline** to generate watermarked/unwatermarked code:
   ```bash
   python pipeline.py
   # Outputs: results.json, report.html
   ```

2. **Train Bayesian detectors** on the generated data:
   ```bash
   python train_bayesian_detector.py --train
   # Outputs: bayesian_detector_ngram5.pkl, ngram2.pkl, ngram10.pkl
   ```

3. **Score samples** to evaluate detection accuracy:
   ```bash
   python train_bayesian_detector.py --score
   # Outputs: bayesian_scores.json
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

Tested on 67 coding problems from `test_results.json`:

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
python visualize_results.py
```

This creates:
- `detector_performance_overall.png`: Box plots showing overall discrimination
- `detector_performance_by_status.png`: Performance breakdown by code correctness (Correct/Wrong/Error)



## Troubleshooting

### Out of Memory
- Reduce batch size in `pipeline.py`
- Use a machine with more GPU memory
- Process fewer samples at once

### Not Enough Training Data
- Run `pipeline.py` on more samples
- Reduce the `test_size` parameter in `train_bayesian_detector.py`

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're using the virtual environment

## References

- [SynthID Text Repository](https://github.com/google-deepmind/synthid-text)
- [APPS Dataset](https://huggingface.co/datasets/codeparrot/apps)
- [CodeGemma Model](https://huggingface.co/google/codegemma-7b-it)