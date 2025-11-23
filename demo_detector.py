import pickle
import io
import torch
import sys
from bayesian_detector import BayesianDetector

# CPU Unpickler for loading CUDA-trained models on CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def load_detector(path):
    print(f"Loading detector from {path}...")
    with open(path, 'rb') as f:
        data = CPU_Unpickler(f).load()
    
    detector = data['detector']
    # Force device to CPU
    if hasattr(detector, 'logits_processor'):
        detector.logits_processor.device = torch.device('cpu')
    return detector

def main():
    # Check for available detectors
    import os
    if os.path.exists("bayesian_detector_ngram5.pkl"):
        detector_path = "bayesian_detector_ngram5.pkl"
    elif os.path.exists("bayesian_detector_ngram2.pkl"):
        detector_path = "bayesian_detector_ngram2.pkl"
    else:
        print("Error: No detector files found (bayesian_detector_ngram*.pkl).")
        return

    try:
        detector = load_detector(detector_path)
    except Exception as e:
        print(f"Error loading detector: {e}")
        return

    # Example code to score
    code_sample = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    print(f"\nScoring sample code with {detector_path}:\n{code_sample}")
    
    # Tokenize
    # Note: The detector object has the tokenizer attached
    tokenizer = detector.tokenizer
    tokens = tokenizer.encode(code_sample, return_tensors='pt', truncation=True, max_length=2048)
    
    # Score
    # detector.score expects tokens
    try:
        score = detector.score(tokens)
        print(f"\nBayesian Score: {float(score[0]):.4f}")
        print("(0 = likely unwatermarked, 1 = likely watermarked)")
    except Exception as e:
        print(f"Error during scoring: {e}")

if __name__ == "__main__":
    main()
