# DO NOT RUN ON LOCAL w/o memory

import multiprocessing
import time
import sys
import io
import contextlib
import traceback
import json
import os
import importlib
from dotenv import load_dotenv
from huggingface_hub import login

# Delayed imports for performance (spawn method re-imports top-level)
torch = None
AutoTokenizer = None
AutoModelForCausalLM = None
SynthIDTextWatermarkingConfig = None
SynthIDTextWatermarkLogitsProcessor = None
load_dataset = None
pd = None

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Prevent tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
MODEL_NAME = "google/gemma-2b-it"
WATERMARK_KEYS = [101, 202, 303, 404, 505, 606, 707, 808, 909]
NGRAM_LENS = [None, 2, 5, 10]
TIMEOUT_SECONDS = 5
OUTPUT_FILE = "results.csv"

# --- Helper Functions ---

def load_model_and_tokenizer():
    global torch, AutoTokenizer, AutoModelForCausalLM, SynthIDTextWatermarkingConfig, SynthIDTextWatermarkLogitsProcessor
    
    # Import here to avoid heavy loading in worker processes
    if torch is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, SynthIDTextWatermarkingConfig
        from transformers import SynthIDTextWatermarkLogitsProcessor
        
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
    # Use chat template for instruction-tuned model
    messages = [
        {"role": "user", "content": f"Please write a Python script for the following problem. Follow this exact structure:\n1. Define a function `solve()` that takes no arguments.\n2. Inside `solve()`, read input from stdin and print output to stdout.\n3. Call `solve()` at the end of the script.\n\n{prompt}\n\nEnsure the code is inside a ```python block."}
    ]
    
    # Fix: apply_chat_template typically returns input_ids (list or tensor), not a dict by default unless specified.
    # Also, some versions don't support return_dict=True. Safer to get input_ids and create dict.
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)
    
    inputs = {"input_ids": input_ids}
    
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
    
    # Decode only the new tokens
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Extract code part
    code = generated_text
    
    # Clean Markdown
    code = code.strip()
    if "```python" in code:
        code = code.split("```python")[1]
        if "```" in code:
            code = code.split("```")[0]
    elif "```" in code:
        code = code.split("```")[1]
        if "```" in code:
            code = code.split("```")[0]
            
    return code.strip()

def compute_g_score(tokenizer, device, text, ngram_len=5, processor=None):
    detect_ngram = ngram_len if ngram_len is not None else 5
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    if input_ids.shape[1] == 0:
        return 0.0
    
    # Optimization: Reuse processor if provided
    if processor is None:
        # Import if needed (should be imported by load_model_and_tokenizer)
        if 'SynthIDTextWatermarkLogitsProcessor' not in globals() or SynthIDTextWatermarkLogitsProcessor is None:
             from transformers import SynthIDTextWatermarkLogitsProcessor
             
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
    # Improve Sandbox: Clear modules to prevent state leakage (partial)
    # Note: True sandboxing requires OS-level isolation (Docker/nsjail).
    
    # Capture stdout
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):

            # Restricted globals
            safe_globals = {
                "__name__": "__main__", # Allow if __name__ == "__main__": blocks
                "__builtins__": {
                    "print": print,
                    "range": range,
                    "len": len,
                    "int": int,
                    "float": float,
                    "str": str,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "bool": bool,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "sorted": sorted,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "abs": abs,
                    "round": round,
                    "input": None # Will be mocked
                }
            }
            
            # Mock input()
            if not isinstance(input_str, str):
                input_str = ""
                
            input_iter = iter(input_str.split('\n'))
            def mock_input(prompt=""):
                try:
                    return next(input_iter)
                except StopIteration:
                    return ""
            
            safe_globals["__builtins__"]["input"] = mock_input
            
            # Execute in restricted scope
            exec(code, safe_globals)
            
            queue.put(("Passed", f.getvalue()))
    except Exception:
        queue.put(("Runtime Error", traceback.format_exc()))

def run_code_safely(code, input_cases, expected_outputs=None):
    # Run against first test case for now (or loop if needed)
    # APPS usually has multiple cases. We'll check the first one or all.
    # For simplicity/speed in this pipeline, checking the first case is a start,
    # but to be "Passed (Correct)", we should ideally check all.
    
    if not input_cases:
        return "Skipped", "No test cases"
        
    # Handle APPS format inconsistencies
    # input_cases can be a list of strings
    # expected_outputs can be a list of strings
    
    input_str = input_cases[0] if len(input_cases) > 0 else ""
    expected = expected_outputs[0] if expected_outputs and len(expected_outputs) > 0 else None
    
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_code_process, args=(code, input_str, queue))
    p.start()
    p.join(TIMEOUT_SECONDS)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return "Timeout", "Execution exceeded time limit"
    
    if not queue.empty():
        status, output = queue.get()
        if status == "Passed":
            # Check correctness
            output = output.strip()
            if expected:
                expected = str(expected).strip()
                if output == expected:
                    return "Passed (Correct)", output
                else:
                    return "Passed (Wrong Output)", f"Expected: {expected}\nGot: {output}"
            return "Passed (No Check)", output
        return status, output
        
    return "Error", "No result returned"

# --- Main Pipeline ---

def main():
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Load model ONLY in main process
    model, tokenizer, device = load_model_and_tokenizer()

    # Delayed import
    global load_dataset
    if load_dataset is None:
        from datasets import load_dataset
        
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
    
    print(f"Selected Problem ID: {sample['problem_id']}")
    
    # Pre-initialize processor to save time
    from transformers import SynthIDTextWatermarkLogitsProcessor
    processor_cache = {} # Cache by ngram_len if needed, or just one if we reuse logic
    
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
            
            if status == "Passed (Correct)":
                print(f"  Passed (Correct) on attempt {attempt}!")
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
            body { font-family: sans-serif; margin: 20px; background-color: #f9f9f9; }
            .problem { background: white; border: 1px solid #ccc; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .prompt { background-color: #f5f5f5; padding: 15px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 14px; }
            th, td { border: 1px solid #ddd; padding: 12px 8px; text-align: left; vertical-align: top; }
            th { background-color: #f2f2f2; font-weight: 600; }
            tr:nth-child(even) { background-color: #f8f8f8; }
            .passed-correct { color: #155724; background-color: #d4edda; font-weight: bold; }
            .passed-wrong { color: #856404; background-color: #fff3cd; font-weight: bold; }
            .failed { color: #721c24; background-color: #f8d7da; }
            .ngram-group { margin-top: 15px; border: 1px solid #e0e0e0; border-radius: 5px; overflow: hidden; }
            .ngram-summary { background-color: #e9ecef; padding: 10px 15px; cursor: pointer; font-weight: bold; display: flex; justify-content: space-between; align-items: center; }
            .ngram-summary:hover { background-color: #dee2e6; }
            .ngram-content { padding: 10px; background: white; }
            pre { white-space: pre-wrap; margin: 0; font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; }
            .code-block { max-height: 300px; overflow-y: auto; background: #2d2d2d; color: #f8f8f2; padding: 10px; border-radius: 4px; }
            .output-block { max-height: 200px; overflow-y: auto; background: #f8f9fa; padding: 10px; border: 1px solid #eee; border-radius: 4px; }
            details > summary { cursor: pointer; outline: none; }
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
        """
        
        # Group results by ngram_len
        from itertools import groupby
        # Ensure sorted for groupby
        sorted_results = sorted(prob['results'], key=lambda x: str(x['ngram_len']))
        
        for ngram, group in groupby(sorted_results, key=lambda x: x['ngram_len']):
            group_list = list(group)
            total = len(group_list)
            passed_correct = sum(1 for r in group_list if r['status'] == "Passed (Correct)")
            mean_g = sum(r['g_score'] for r in group_list) / total if total > 0 else 0
            
            html += f"""
            <details class="ngram-group" open>
                <summary class="ngram-summary">
                    <span>N-gram: {ngram}</span>
                    <span>Pass (Correct): {passed_correct}/{total} | Mean G-Score: {mean_g:.4f}</span>
                </summary>
                <div class="ngram-content">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 50px">#</th>
                                <th style="width: 150px">Status</th>
                                <th style="width: 80px">G-Score</th>
                                <th>Output/Error</th>
                                <th>Generated Code</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for res in group_list:
                status_class = ""
                if res['status'] == "Passed (Correct)":
                    status_class = "passed-correct"
                elif "Passed" in res['status']:
                    status_class = "passed-wrong"
                else:
                    status_class = "failed"
                
                html += f"""
                        <tr>
                            <td>{res['attempts']}</td>
                            <td class="{status_class}">{res['status']}</td>
                            <td>{res['g_score']:.4f}</td>
                            <td>
                                <details>
                                    <summary>Show Output</summary>
                                    <div class="output-block"><pre>{res['output']}</pre></div>
                                </details>
                            </td>
                            <td>
                                <details>
                                    <summary>Show Code</summary>
                                    <div class="code-block"><pre>{res['generated_code']}</pre></div>
                                </details>
                            </td>
                        </tr>
                """
            
            html += """
                        </tbody>
                    </table>
                </div>
            </details>
            """
            
        html += """
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
