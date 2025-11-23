import json
import numpy as np
import pickle
import torch
import io
import sys
from pathlib import Path
from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bayesian_detector import BayesianDetector

# CPU Unpickler for loading CUDA-trained models on CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def load_detector(ngram_len):
    path = f"outputs/models/bayesian_detector_ngram{ngram_len}.pkl"
    print(f"Loading detector from {path}...")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
        
    with open(path, 'rb') as f:
        data = CPU_Unpickler(f).load()
    
    detector = data['detector']
    # Force device to CPU
    if hasattr(detector, 'logits_processor'):
        detector.logits_processor.device = torch.device('cpu')
    return detector

def get_summary_stats(scores):
    if not scores:
        return "N/A"
    return (f"Count: {len(scores)}, Mean: {np.mean(scores):.4f}, "
            f"Std: {np.std(scores):.4f}, Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")

def main():
    detectors = {}
    for n in [2, 5, 10]:
        d = load_detector(n)
        if d:
            detectors[n] = d

    if not detectors:
        print("No detectors loaded.")
        return

    try:
        with open("outputs/results/test_results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: outputs/results/test_results.json not found.")
        return

    # Store scores: scores[detector_ngram][data_type] = []
    # data_type is either the ngram_len (e.g. 2, 5, 10) or "None"
    scores_map = {n: {'watermarked': [], 'unwatermarked': []} for n in detectors}

    print(f"\nScoring {len(data)} problems...")
    
    for problem in data:
        results = problem.get('results', [])
        
        for res in results:
            ngram_len_str = res.get('ngram_len')
            code = res.get('generated_code', '')
            
            if not code.strip():
                continue
            
            # Determine which detectors to run
            # We run Detector N on Data N (Watermarked)
            # We run ALL Detectors on Data None (Unwatermarked)
            
            if ngram_len_str == "None":
                for n, detector in detectors.items():
                    try:
                        tokens = detector.tokenizer.encode(code, return_tensors='pt', truncation=True, max_length=2048)
                        score = float(detector.score(tokens)[0])
                        scores_map[n]['unwatermarked'].append(score)
                    except Exception as e:
                        pass
            else:
                try:
                    n = int(ngram_len_str)
                    if n in detectors:
                        detector = detectors[n]
                        tokens = detector.tokenizer.encode(code, return_tensors='pt', truncation=True, max_length=2048)
                        score = float(detector.score(tokens)[0])
                        scores_map[n]['watermarked'].append(score)
                except ValueError:
                    pass
                except Exception as e:
                    pass

    print("\n" + "="*80)
    print(f"{'Detector':<10} | {'Data Type':<15} | {'Stats'}")
    print("="*80)
    
    for n in sorted(detectors.keys()):
        # Watermarked
        wm_stats = get_summary_stats(scores_map[n]['watermarked'])
        print(f"Ngram={n:<4} | {'Watermarked':<15} | {wm_stats}")
        
        # Unwatermarked
        uwm_stats = get_summary_stats(scores_map[n]['unwatermarked'])
        print(f"Ngram={n:<4} | {'Unwatermarked':<15} | {uwm_stats}")
        print("-" * 80)

if __name__ == "__main__":
    main()
