"""
Script to train and use the Bayesian detector on results.json data.

Usage:
    python train_bayesian_detector.py --train     # Train a new detector
    python train_bayesian_detector.py --score     # Score samples using trained detector
"""

import json
import torch
import numpy as np
import argparse
import pickle
from transformers import AutoTokenizer
from transformers import SynthIDTextWatermarkLogitsProcessor
from bayesian_detector import BayesianDetector

# Configuration
MODEL_NAME = "google/codegemma-7b-it"
WATERMARK_KEYS = [101, 202, 303, 404, 505, 606, 707, 808, 909]
RESULTS_FILE = "results.json"
DETECTOR_SAVE_PATH = "bayesian_detector.pkl"


def load_and_prepare_data(results_file):
    """Load results.json and separate into watermarked/unwatermarked samples."""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} problems from {results_file}")
    
    watermarked_codes = []
    unwatermarked_codes = []
    
    for problem in results:
        for result in problem['results']:
            code = result['generated_code']
            ngram_len = result['ngram_len']
            
            if code.strip():  # Skip empty code
                if ngram_len == 'None':
                    unwatermarked_codes.append(code)
                else:
                    watermarked_codes.append(code)
    
    print(f"Found {len(watermarked_codes)} watermarked samples")
    print(f"Found {len(unwatermarked_codes)} unwatermarked samples")
    
    return watermarked_codes, unwatermarked_codes


def tokenize_codes(tokenizer, codes, device):
    """Tokenize a list of code strings."""
    tokenized = []
    
    for code in codes:
        try:
            tokens = tokenizer.encode(code, return_tensors='pt', truncation=True, max_length=2048)
            tokenized.append(tokens.squeeze().cpu().numpy())
        except Exception as e:
            print(f"Warning: Failed to tokenize code: {e}")
            continue
    
    return tokenized


def train_detector(watermarked_codes, unwatermarked_codes):
    """Train the Bayesian detector."""
    
    print("\n=== Initializing Tokenizer and Processor ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tokenize codes
    print("\n=== Tokenizing Samples ===")
    watermarked_outputs = tokenize_codes(tokenizer, watermarked_codes, device)
    unwatermarked_outputs = tokenize_codes(tokenizer, unwatermarked_codes, device)
    
    print(f"Successfully tokenized {len(watermarked_outputs)} watermarked samples")
    print(f"Successfully tokenized {len(unwatermarked_outputs)} unwatermarked samples")
    
    if len(watermarked_outputs) < 10 or len(unwatermarked_outputs) < 10:
        raise ValueError("Need at least 10 samples of each type for training!")
    
    # Initialize logits processor
    print("\n=== Initializing Logits Processor ===")
    logits_processor = SynthIDTextWatermarkLogitsProcessor(
        keys=WATERMARK_KEYS,
        ngram_len=5,  # Representative value
        sampling_table_size=2**16,
        sampling_table_seed=0,
        context_history_size=1024,
        device=device
    )
    
    # Train detector
    print("\n=== Training Bayesian Detector ===")
    print("This may take several minutes...")
    
    detector, min_loss = BayesianDetector.train_best_detector(
        tokenized_wm_outputs=watermarked_outputs,
        tokenized_uwm_outputs=unwatermarked_outputs,
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        torch_device=device,
        test_size=0.3,  # 30% for validation
        pos_truncation_length=200,  # Watermarked truncation length
        neg_truncation_length=100,  # Unwatermarked truncation length
        max_padded_length=2300,
        n_epochs=50,
        learning_rate=2.1e-2,
        l2_weights=np.logspace(-3, -2, num=4),
        verbose=True
    )
    
    print(f"\n=== Training Complete ===")
    print(f"Minimum validation loss: {min_loss:.4f}")
    
    # Save detector
    print(f"\n=== Saving Detector ===")
    with open(DETECTOR_SAVE_PATH, 'wb') as f:
        pickle.dump({
            'detector': detector,
            'min_loss': min_loss,
            'watermark_keys': WATERMARK_KEYS
        }, f)
    print(f"Detector saved to {DETECTOR_SAVE_PATH}")
    
    return detector


def score_samples(detector=None):
    """Score samples from results.json using the trained detector."""
    
    # Load detector if not provided
    if detector is None:
        print(f"Loading detector from {DETECTOR_SAVE_PATH}...")
        with open(DETECTOR_SAVE_PATH, 'rb') as f:
            saved_data = pickle.load(f)
            detector = saved_data['detector']
    
    # Load results
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    
    print(f"\n=== Scoring Samples ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_scores = []
    
    for problem_idx, problem in enumerate(results):
        print(f"\nProblem {problem_idx + 1}/{len(results)} (ID: {problem['problem_id']})")
        
        for result in problem['results']:
            code = result['generated_code']
            ngram_len = result['ngram_len']
            
            if not code.strip():
                continue
            
            # Tokenize
            tokens = tokenizer.encode(code, return_tensors='pt', truncation=True, max_length=2048)
            tokens = tokens.to(device)
            
            # Score
            score = detector.score(tokens)
            score_value = float(score[0])
            
            print(f"  ngram_len={ngram_len:>4} | Score={score_value:.4f} | Status={result['status']}")
            
            all_scores.append({
                'problem_id': problem['problem_id'],
                'ngram_len': ngram_len,
                'bayesian_score': score_value,
                'g_score': result['g_score'],
                'status': result['status']
            })
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    none_scores = [s['bayesian_score'] for s in all_scores if s['ngram_len'] == 'None']
    wm_scores = [s['bayesian_score'] for s in all_scores if s['ngram_len'] != 'None']
    
    if none_scores:
        print(f"Unwatermarked (None): Mean={np.mean(none_scores):.4f}, Std={np.std(none_scores):.4f}")
    if wm_scores:
        print(f"Watermarked (2,5,10): Mean={np.mean(wm_scores):.4f}, Std={np.std(wm_scores):.4f}")
    
    # Save scores
    scores_file = "bayesian_scores.json"
    with open(scores_file, 'w') as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nScores saved to {scores_file}")


def main():
    parser = argparse.ArgumentParser(description='Train and use Bayesian detector')
    parser.add_argument('--train', action='store_true', help='Train a new detector')
    parser.add_argument('--score', action='store_true', help='Score samples with trained detector')
    
    args = parser.parse_args()
    
    if args.train:
        # Load and prepare data
        watermarked_codes, unwatermarked_codes = load_and_prepare_data(RESULTS_FILE)
        
        # Train detector
        detector = train_detector(watermarked_codes, unwatermarked_codes)
        
        # Optionally score after training
        print("\n" + "="*60)
        response = input("Would you like to score the samples now? (y/n): ")
        if response.lower() == 'y':
            score_samples(detector)
    
    elif args.score:
        # Score samples with existing detector
        score_samples()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
