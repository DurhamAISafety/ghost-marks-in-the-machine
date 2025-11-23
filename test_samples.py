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

def score_code(detector, code, label):
    print(f"\n--- Scoring {label} ---")
    print(code.strip())
    print("-" * 20)
    
    tokenizer = detector.tokenizer
    tokens = tokenizer.encode(code, return_tensors='pt', truncation=True, max_length=2048)
    
    try:
        score = detector.score(tokens)
        print(f"Bayesian Score: {float(score[0]):.4f}")
    except Exception as e:
        print(f"Error during scoring: {e}")

def main():
    detector_path = "bayesian_detector_ngram5.pkl"
    
    try:
        detector = load_detector(detector_path)
    except Exception as e:
        print(f"Error loading detector: {e}")
        return

    # Problem ID: 0
    
    unwatermarked_code = """
def solve():
    s = input()
    count = 0
    for i in s:
        if i == '[' or i == ']' or i == ':' or i == '|':
            count += 1
    if (count == 0) or (count % 3 != 0):
        print(-1)
    else:
        print(count//3*2)

solve()
"""

    watermarked_code = """
def solve():
    # define function here with no arguments, which reads input from stdin and prints output to stdout
    n = int(input())
    s = input()
    flag = False
    flag_2 = False
    ans = 0
    for i in range(n):
        if s[i] == '[' and flag == False:
            flag = True
        elif s[i] == ']' and s[-i-1] == '[' and flag == True:
            flag = False
        elif s[i] == ']' and flag == False:
            flag_2 = True
            flag = True
            ans +=2
        elif s[i] == '[' and flag_2 == True:
            flag = True
        elif s[i] == ':' and s[i+1] == ':' and flag_2 == True:
            flag_2 = False
        elif s[i] == '|' and flag == True:
            ans += 1
    if flag == True:
        print(-1)
    else:
        print(ans)

solve()
"""

    score_code(detector, unwatermarked_code, "Unwatermarked (ngram=None)")
    score_code(detector, watermarked_code, "Watermarked (ngram=5)")

if __name__ == "__main__":
    main()
