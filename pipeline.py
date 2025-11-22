# DO NOT RUN ON LOCAL w/o memory

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, SynthIDTextWatermarkingConfig
from transformers import SynthIDTextWatermarkLogitsProcessor
from datasets import load_dataset
import pandas as pd
import multiprocessing
import time
import sys
import io
import contextlib
import traceback
import json

# --- Configuration ---
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
WATERMARK_KEYS = [101, 202, 303, 404, 505, 606, 707, 808, 909]
NGRAM_LENS = [None, 2, 5, 10]
TIMEOUT_SECONDS = 5
OUTPUT_FILE = "results.csv"

# --- Helper Functions ---

def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"

    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.bfloat16 if device != "mps" else torch.float16
    )
    return model, tokenizer, device

def generate_code(model, tokenizer, device, prompt, ngram_len=None):
    # Wrap prompt to ensure Python code generation
    full_prompt = f"Please write a Python solution for the following problem:\n\n{prompt}\n\n```python\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    watermark_config = None
    if ngram_len is not None:
        watermark_config = SynthIDTextWatermarkingConfig(
            keys=WATERMARK_KEYS,
            ngram_len=ngram_len,
        )
    
    outputs = model.generate(
        **inputs,
        watermarking_config=watermark_config,
        do_sample=True,
        max_new_tokens=400,
        top_k=40,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract code part
    # We expect the model to continue after ```python\n
    if full_prompt in generated_text:
        code = generated_text.split(full_prompt)[1]
    else:
        # Fallback: try to find the prompt or just return everything
        code = generated_text
        
    # Clean Markdown
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    return code.strip()

def compute_g_score(tokenizer, device, text, ngram_len=5):
    detect_ngram = ngram_len if ngram_len is not None else 5
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    if input_ids.shape[1] == 0:
        return 0.0
    
    processor = SynthIDTextWatermarkLogitsProcessor(
        keys=WATERMARK_KEYS,
        ngram_len=detect_ngram,
        sampling_table_size=2**16,
        sampling_table_seed=0,
        context_history_size=1024,
        device=device
    )
    
    g_values = processor.compute_g_values(input_ids)
    mean_g = g_values.float().mean().item()
    return mean_g

# --- Safe Execution ---

def _run_code_process(code, input_str, queue):
    # Capturing stdout
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            # Basic safety check
            if "import os" in code or "import subprocess" in code or "import sys" in code:
                queue.put(("Failed", "Security Risk: Forbidden imports"))
                return

            # Create a local scope
            local_scope = {}
            
            # Mock input()
            if not isinstance(input_str, str):
                input_str = ""
                
            input_iter = iter(input_str.split('\n'))
            def mock_input(prompt=""):
                try:
                    return next(input_iter)
                except StopIteration:
                    return ""
            
            local_scope['input'] = mock_input
            
            # Execute
            exec(code, {}, local_scope)
            
            queue.put(("Passed", f.getvalue()))
    except Exception:
        queue.put(("Failed", traceback.format_exc()))

def run_code_safely(code, input_cases):
    input_str = input_cases[0] if input_cases else ""
    
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_code_process, args=(code, input_str, queue))
    p.start()
    p.join(TIMEOUT_SECONDS)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return "Timeout", "Execution exceeded time limit"
    
    if not queue.empty():
        return queue.get()
    return "Error", "No result returned"

# --- Main Pipeline ---

def main():
    # Load model ONLY in main process
    model, tokenizer, device = load_model_and_tokenizer()

    print("Loading APPS dataset...")
    ds = load_dataset("codeparrot/apps", "all", split="test", trust_remote_code=True)
    
    print("Filtering for interview difficulty...")
    ds = ds.filter(lambda x: x['difficulty'] == 'interview')
    
    if len(ds) == 0:
        print("Error: No samples found with difficulty='interview'")
        return

    sample = ds[0]
    prompt = sample['question']
    input_output = sample['input_output']
    
    if isinstance(input_output, str):
        io_data = json.loads(input_output)
    else:
        io_data = input_output
        
    test_inputs = io_data.get('inputs', [])
    
    print(f"Selected Problem ID: {sample['problem_id']}")
    
    results = []
    
    for n in NGRAM_LENS:
        print(f"\nProcessing ngram_len={n}...")
        
        final_result = None
        
        for attempt in range(1, 6):
            print(f"  Attempt {attempt}/5...")
            
            # Generate
            try:
                code = generate_code(model, tokenizer, device, prompt, ngram_len=n)
            except Exception as e:
                print(f"Generation failed: {e}")
                final_result = {
                    "ngram_len": str(n),
                    "g_score": 0.0,
                    "status": "Generation Error",
                    "output": str(e),
                    "generated_code": "",
                    "attempts": attempt
                }
                continue
                
            # Detect
            score = compute_g_score(tokenizer, device, code, ngram_len=n)
            
            # Execute
            status, output = run_code_safely(code, test_inputs)
            
            print(f"    G-Score: {score:.4f}")
            print(f"    Status: {status}")
            if status == "Failed":
                print(f"    Error: {output}")
            
            final_result = {
                "ngram_len": str(n),
                "g_score": score,
                "status": status,
                "output": output,
                "generated_code": code,
                "attempts": attempt
            }
            
            if status == "Passed":
                print(f"  Passed on attempt {attempt}!")
                break
        
        if final_result:
            results.append(final_result)
        
    # Structure for JSON: List of problems
    # Since we only run one problem here, it's a list with one element
    problem_data = {
        "problem_id": sample['problem_id'],
        "prompt": prompt,
        "results": results
    }
    
    # Save to JSON
    output_json = "results.json"
    with open(output_json, "w") as f:
        json.dump([problem_data], f, indent=2)
    print(f"\nResults saved to {output_json}")
    
    # Generate HTML Report
    generate_html_report([problem_data], "report.html")
    print(f"Report saved to report.html")

def generate_html_report(data, filename):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SynthID Evaluation Report</title>
        <style>
            body { font-family: sans-serif; margin: 20px; }
            .problem { border: 1px solid #ccc; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
            .prompt { background-color: #f5f5f5; padding: 10px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; border: 1px solid #ddd; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
            th { background-color: #f2f2f2; }
            .passed { color: green; font-weight: bold; }
            .failed { color: red; }
            pre { white-space: pre-wrap; margin: 0; max-height: 300px; overflow-y: auto; background: #f8f8f8; padding: 5px; }
        </style>
    </head>
    <body>
        <h1>SynthID Evaluation Report</h1>
    """
    
    for prob in data:
        html += f"""
        <div class="problem">
            <h2>Problem ID: {prob['problem_id']}</h2>
            <details>
                <summary><strong>Prompt (Click to expand)</strong></summary>
                <div class="prompt">{prob['prompt']}</div>
            </details>
            
            <table>
                <thead>
                    <tr>
                        <th>N-gram</th>
                        <th>Attempts</th>
                        <th>Status</th>
                        <th>G-Score</th>
                        <th>Output/Error</th>
                        <th>Generated Code</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for res in prob['results']:
            status_class = "passed" if res['status'] == "Passed" else "failed"
            html += f"""
                    <tr>
                        <td>{res['ngram_len']}</td>
                        <td>{res['attempts']}</td>
                        <td class="{status_class}">{res['status']}</td>
                        <td>{res['g_score']:.4f}</td>
                        <td><pre>{res['output']}</pre></td>
                        <td><pre>{res['generated_code']}</pre></td>
                    </tr>
            """
            
        html += """
                </tbody>
            </table>
        </div>
        """
        
    html += """
    </body>
    </html>
    """
    
    with open(filename, "w") as f:
        f.write(html)

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
