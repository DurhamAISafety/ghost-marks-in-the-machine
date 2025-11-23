import json
import pickle
import io
import torch
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bayesian_detector import BayesianDetector

# CPU Unpickler for loading CUDA-trained models on CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def load_detector(ngram_len):
    path = f"bayesian_detector_ngram{ngram_len}.pkl"
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

def get_status_category(status_str):
    if "Correct" in status_str:
        return "Correct Output"
    elif "Wrong Output" in status_str:
        return "Wrong Output"
    else:
        return "Error"

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
        with open("test_results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: test_results.json not found.")
        return

    records = []

    print(f"Processing {len(data)} problems...")
    
    for problem in data:
        results = problem.get('results', [])
        
        for res in results:
            ngram_len_str = res.get('ngram_len')
            code = res.get('generated_code', '')
            status_str = res.get('status', 'Error')
            status_cat = get_status_category(status_str)
            
            if not code.strip():
                continue
            
            # Logic:
            # If unwatermarked (None), test against ALL detectors.
            # If watermarked (N), test ONLY against detector N.
            
            if ngram_len_str == "None":
                for n, detector in detectors.items():
                    try:
                        tokens = detector.tokenizer.encode(code, return_tensors='pt', truncation=True, max_length=2048)
                        score = float(detector.score(tokens)[0])
                        records.append({
                            'Detector': f"Ngram={n}",
                            'Type': 'Unwatermarked',
                            'Status': status_cat,
                            'Score': score
                        })
                    except Exception as e:
                        pass
            else:
                try:
                    n = int(ngram_len_str)
                    if n in detectors:
                        detector = detectors[n]
                        tokens = detector.tokenizer.encode(code, return_tensors='pt', truncation=True, max_length=2048)
                        score = float(detector.score(tokens)[0])
                        records.append({
                            'Detector': f"Ngram={n}",
                            'Type': 'Watermarked',
                            'Status': status_cat,
                            'Score': score
                        })
                except ValueError:
                    pass
                except Exception as e:
                    pass

    df = pd.DataFrame(records)
    print(f"Collected {len(df)} records.")
    
    # Set style
    sns.set_theme(style="whitegrid")

    # Plot 1: Overall Performance
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Detector', y='Score', hue='Type', palette="Set2")
    plt.title('Overall Detector Performance: Watermarked vs Unwatermarked')
    plt.ylim(-0.05, 1.05)
    plt.savefig('detector_performance_overall.png')
    print("Saved detector_performance_overall.png")
    plt.close()

    # Plot 2: Performance by Status
    # We want a plot for each Status type
    g = sns.catplot(
        data=df, 
        x='Detector', 
        y='Score', 
        hue='Type', 
        col='Status', 
        kind='box', 
        palette="Set2",
        height=5, 
        aspect=0.8,
        col_order=['Correct Output', 'Wrong Output', 'Error']
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Detector Performance by Code Response Type')
    g.set(ylim=(-0.05, 1.05))
    plt.savefig('detector_performance_by_status.png')
    print("Saved detector_performance_by_status.png")
    plt.close()

if __name__ == "__main__":
    main()
