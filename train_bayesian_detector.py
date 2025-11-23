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
import io
from transformers import AutoTokenizer
from transformers import SynthIDTextWatermarkLogitsProcessor
from bayesian_detector import BayesianDetector

# Configuration
MODEL_NAME = "google/codegemma-7b-it"
WATERMARK_KEYS = [101, 202, 303, 404, 505, 606, 707, 808, 909]
RESULTS_FILE = "results.json"
DETECTOR_SAVE_PATH = "bayesian_detector.pkl"


def load_and_prepare_data(results_file):
    """Load results.json and separate into watermarked/unwatermarked samples by ngram_len."""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} problems from {results_file}")
    
    # Organize by ngram_len
    watermarked_by_ngram = {2: [], 5: [], 10: []}
    unwatermarked_codes = []
    
    for problem in results:
        for result in problem['results']:
            code = result['generated_code']
            ngram_len = result['ngram_len']
            
            if code.strip():  # Skip empty code
                if ngram_len == 'None':
                    unwatermarked_codes.append(code)
                else:
                    # Parse ngram_len as int
                    ngram_val = int(ngram_len)
                    if ngram_val in watermarked_by_ngram:
                        watermarked_by_ngram[ngram_val].append(code)
    
    print(f"Found {len(unwatermarked_codes)} unwatermarked samples")
    for ngram in [2, 5, 10]:
        print(f"Found {len(watermarked_by_ngram[ngram])} watermarked samples with ngram_len={ngram}")
    
    return watermarked_by_ngram, unwatermarked_codes


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


def train_detector_for_ngram(ngram_len, watermarked_codes, unwatermarked_codes, tokenizer, device):
    """Train a Bayesian detector for a specific ngram_len."""
    
    print(f"\n{'='*70}")
    print(f"Training detector for ngram_len={ngram_len}")
    print(f"{'='*70}")
    
    # Tokenize codes
    print(f"\n=== Tokenizing Samples for ngram_len={ngram_len} ===")
    watermarked_outputs = tokenize_codes(tokenizer, watermarked_codes, device)
    unwatermarked_outputs = tokenize_codes(tokenizer, unwatermarked_codes, device)
    
    print(f"Successfully tokenized {len(watermarked_outputs)} watermarked samples")
    print(f"Successfully tokenized {len(unwatermarked_outputs)} unwatermarked samples")
    
    if len(watermarked_outputs) < 5 or len(unwatermarked_outputs) < 5:
        print(f"WARNING: Not enough samples for ngram_len={ngram_len}. Skipping.")
        return None
    
    # Initialize logits processor with specific ngram_len
    print(f"\n=== Initializing Logits Processor (ngram_len={ngram_len}) ===")
    logits_processor = SynthIDTextWatermarkLogitsProcessor(
        keys=WATERMARK_KEYS,
        ngram_len=ngram_len,
        sampling_table_size=2**16,
        sampling_table_seed=0,
        context_history_size=1024,
        device=device
    )
    
    # Train detector
    print(f"\n=== Training Bayesian Detector (ngram_len={ngram_len}) ===")
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
    
    print(f"\n=== Training Complete for ngram_len={ngram_len} ===")
    print(f"Minimum validation loss: {min_loss:.4f}")
    
    # Save detector
    save_path = f"bayesian_detector_ngram{ngram_len}.pkl"
    print(f"\n=== Saving Detector ===")
    with open(save_path, 'wb') as f:
        pickle.dump({
            'detector': detector,
            'min_loss': min_loss,
            'ngram_len': ngram_len,
            'watermark_keys': WATERMARK_KEYS
        }, f)
    print(f"Detector saved to {save_path}")
    
    return detector


def train_all_detectors(watermarked_by_ngram, unwatermarked_codes):
    """Train detectors for all ngram lengths."""
    
    print("\n=== Initializing Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    detectors = {}
    
    for ngram_len in [5, 2, 10]:  # Train ngram=5 first as requested
        watermarked_codes = watermarked_by_ngram[ngram_len]
        
        if len(watermarked_codes) == 0:
            print(f"\nNo watermarked samples for ngram_len={ngram_len}. Skipping.")
            continue
        
        detector = train_detector_for_ngram(
            ngram_len, 
            watermarked_codes, 
            unwatermarked_codes, 
            tokenizer, 
            device
        )
        
        if detector is not None:
            detectors[ngram_len] = detector
    
    return detectors


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def score_samples(detectors=None):
    """Score samples from results.json using the trained detectors."""
    
    # Load detectors if not provided
    if detectors is None:
        print(f"Loading detectors...")
        detectors = {}
        for ngram_len in [2, 5, 10]:
            detector_path = f"bayesian_detector_ngram{ngram_len}.pkl"
            try:
                with open(detector_path, 'rb') as f:
                    try:
                        # Use custom unpickler to force CPU mapping
                        saved_data = CPU_Unpickler(f).load()
                    except Exception as e:
                        print(f"  Error loading {detector_path}: {e}")
                        continue
                    
                    detectors[ngram_len] = saved_data['detector']
                    # Force device to CPU to avoid CUDA errors when running on CPU
                    if hasattr(detectors[ngram_len], 'logits_processor'):
                        detectors[ngram_len].logits_processor.device = torch.device('cpu')
                    
                    print(f"Loaded detector for ngram_len={ngram_len}")
            except FileNotFoundError:
                print(f"Warning: Could not find detector for ngram_len={ngram_len}")
    
    if not detectors:
        raise ValueError("No detectors loaded! Please train detectors first with --train")
    
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
            
            # Score with appropriate detector
            if ngram_len == 'None':
                # For unwatermarked, we can use any detector - let's use ngram=5 if available
                detector = detectors.get(5) or detectors.get(2) or detectors.get(10)
                score_value = 0.0 if detector is None else float(detector.score(tokens)[0])
            else:
                # Use detector trained for this specific ngram_len
                ngram_val = int(ngram_len)
                detector = detectors.get(ngram_val)
                if detector is None:
                    print(f"  Warning: No detector for ngram_len={ngram_val}, skipping")
                    continue
                score_value = float(detector.score(tokens)[0])
            
            print(f"  ngram_len={ngram_len:>4} | Bayesian Score={score_value:.4f} | G-Score={result['g_score']:.4f} | Status={result['status']}")
            
            all_scores.append({
                'problem_id': problem['problem_id'],
                'ngram_len': ngram_len,
                'bayesian_score': score_value,
                'g_score': result['g_score'],
                'status': result['status']
            })
    
    # Summary statistics by ngram_len
    print("\n=== Summary Statistics ===")
    none_scores = [s['bayesian_score'] for s in all_scores if s['ngram_len'] == 'None']
    
    if none_scores:
        print(f"Unwatermarked (None): Mean={np.mean(none_scores):.4f}, Std={np.std(none_scores):.4f}")
    
    for ngram in [2, 5, 10]:
        ngram_scores = [s['bayesian_score'] for s in all_scores if s['ngram_len'] == str(ngram)]
        if ngram_scores:
            print(f"Watermarked (ngram={ngram}): Mean={np.mean(ngram_scores):.4f}, Std={np.std(ngram_scores):.4f}")
    
    # Save scores
    scores_file = "bayesian_scores.json"
    with open(scores_file, 'w') as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nScores saved to {scores_file}")


def main():
    parser = argparse.ArgumentParser(description='Train and use Bayesian detectors')
    parser.add_argument('--train', action='store_true', help='Train detectors for all ngram lengths')
    parser.add_argument('--score', action='store_true', help='Score samples with trained detectors')
    
    args = parser.parse_args()
    
    if args.train:
        # Load and prepare data
        watermarked_by_ngram, unwatermarked_codes = load_and_prepare_data(RESULTS_FILE)
        
        # Train detectors for all ngram lengths
        detectors = train_all_detectors(watermarked_by_ngram, unwatermarked_codes)
        
        print(f"\n{'='*70}")
        print(f"Training complete! Saved {len(detectors)} detector(s)")
        print(f"{'='*70}")
        
        # Optionally score after training
        print("\n" + "="*60)
        response = input("Would you like to score the samples now? (y/n): ")
        if response.lower() == 'y':
            score_samples(detectors)
    
    elif args.score:
        # Score samples with existing detectors
        score_samples()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
