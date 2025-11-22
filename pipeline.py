# DO NOT RUN ON LOCAL w/o memory

import multiprocessing
import json
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import SynthIDTextWatermarkLogitsProcessor

# Import custom modules
from model_utils import load_model_and_tokenizer, generate_code, compute_g_score, WATERMARK_KEYS
from execution_utils import run_code_safely
from report_generator import generate_html_report

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Prevent tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
NGRAM_LENS = [None, 2, 5, 10]

# --- Main Pipeline ---

def main():
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Load model ONLY in main process
    model, tokenizer, device = load_model_and_tokenizer()

    # Delayed import for datasets
    from datasets import load_dataset
        
    print("Loading APPS dataset...")
    ds = load_dataset("codeparrot/apps", "all", split="test", trust_remote_code=True)
    
    print("Filtering for interview difficulty...")
    ds = ds.filter(lambda x: x['difficulty'] == 'interview')
    
    if len(ds) == 0:
        print("Error: No samples found with difficulty='interview'")
        return

    # Pre-initialize processor to save time
    processor_cache = {} # Cache by ngram_len
    
    all_results = []
    
    print(f"Processing {len(ds)} samples...")
    
    for i, sample in enumerate(ds):
        print(f"\n--- Processing Sample {i+1}/{len(ds)} (Problem ID: {sample['problem_id']}) ---")
        
        prompt = sample['question']
        input_output = sample['input_output']
        
        # Robust parsing
        try:
            if isinstance(input_output, str):
                io_data = json.loads(input_output)
            else:
                io_data = input_output
        except json.JSONDecodeError:
            print("Error parsing input_output JSON")
            io_data = {"inputs": [], "outputs": []}
            
        test_inputs = io_data.get('inputs', [])
        test_outputs = io_data.get('outputs', [])
        
        problem_results = []
        
        for n in NGRAM_LENS:
            print(f"  Processing ngram_len={n}...")
            
            for attempt in range(1, 6):
                # Generate
                try:
                    code = generate_code(model, tokenizer, device, prompt, ngram_len=n)
                except Exception as e:
                    print(f"    Generation failed: {e}")
                    current_result = {
                        "ngram_len": str(n),
                        "g_score": 0.0,
                        "status": "Generation Error",
                        "output": str(e),
                        "generated_code": "",
                        "attempts": attempt
                    }
                    problem_results.append(current_result)
                    continue
                    
                # Detect
                # Use cached processor
                if n not in processor_cache:
                    # Initialize if not in cache
                    processor_cache[n] = SynthIDTextWatermarkLogitsProcessor(
                        keys=WATERMARK_KEYS,
                        ngram_len=n if n is not None else 5,
                        sampling_table_size=2**16,
                        sampling_table_seed=0,
                        context_history_size=1024,
                        device=device
                    )
                
                score = compute_g_score(tokenizer, device, code, ngram_len=n, processor=processor_cache[n])
                
                # Execute
                status, output = run_code_safely(code, test_inputs, test_outputs)
                
                print(f"    Attempt {attempt}: G-Score: {score:.4f} | Status: {status}")
                
                current_result = {
                    "ngram_len": str(n),
                    "g_score": score,
                    "status": status,
                    "output": output,
                    "generated_code": code,
                    "attempts": attempt
                }
                
                problem_results.append(current_result)
                
                if status == "Passed (Correct)":
                    break
            
        # Structure for JSON
        problem_data = {
            "problem_id": sample['problem_id'],
            "prompt": prompt,
            "results": problem_results
        }
        all_results.append(problem_data)
        
        # Save intermediate results
        if (i + 1) % 5 == 0:
             with open("results_partial.json", "w") as f:
                json.dump(all_results, f, indent=2)
    
    # Save to JSON
    output_json = "results.json"
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_json}")
    
    # Generate HTML Report
    generate_html_report(all_results, "report.html")
    print(f"Report saved to report.html")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
