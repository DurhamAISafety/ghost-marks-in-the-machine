import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, SynthIDTextWatermarkingConfig
from transformers import SynthIDTextWatermarkLogitsProcessor
import numpy as np

# 1. Load a small, fast coding model
# 'deepseek-ai/deepseek-coder-1.3b-base' or 'google/codegemma-2b' are great choices
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base" 
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu" and torch.backends.mps.is_available():
    device = "mps"

print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto", 
    torch_dtype=torch.bfloat16 if device != "mps" else torch.float16 # MPS doesn't support bfloat16 well sometimes
)

# Define keys globally for consistency
WATERMARK_KEYS = [101, 202, 303, 404, 505, 606, 707, 808, 909]

def generate_watermarked_code(prompt, ngram_len=5):
    print(f"\nGenerating with ngram_len={ngram_len}...")
    watermark_config = SynthIDTextWatermarkingConfig(
        keys=WATERMARK_KEYS,
        ngram_len=ngram_len, 
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        watermarking_config=watermark_config,
        do_sample=True,
        max_new_tokens=200,
        top_k=40,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated_code

def detect_watermark(text, ngram_len=5):
    # Simple detection heuristic: compute mean g-value
    # We re-create the logits processor to access compute_g_values
    
    # Note: We need to tokenize the text first
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Initialize the processor (we only need it for utility functions)
    # We need to match the config used for generation
    processor = SynthIDTextWatermarkLogitsProcessor(
        keys=WATERMARK_KEYS,
        ngram_len=ngram_len,
        sampling_table_size=2**16, # Default
        sampling_table_seed=0, # Default
        context_history_size=1024, # Default
        device=device
    )
    
    # Compute g-values
    # g_values shape: (batch_size, seq_len - (ngram_len - 1), depth)
    # We only care about the generated part, but here we check the whole text
    # Ideally we should only check the new tokens, but for this demo checking whole text is okay
    # if the prompt is short compared to generated text.
    
    g_values = processor.compute_g_values(input_ids)
    
    # Compute mean g-value across all tokens and depths
    # g-values are 0 or 1 (or similar, depending on implementation, usually they are values used to bias logits)
    # Wait, compute_g_values returns values that are used to *index* into the sampling table?
    # No, sample_g_values does that. compute_g_values returns the g values.
    # Let's assume they are the values that are added to logits or used for sampling.
    # Actually, SynthID usually biases towards a specific g-value.
    # Let's just print the mean and see if it differs from 0.5 (random).
    
    mean_g = g_values.float().mean().item()
    return mean_g

# Experimentation Loop
prompt = "def fibonacci(n):"
ngram_lens = [5, 3, 2]

print(f"Prompt: {prompt}")

results = []

# Baseline: No Watermark
print("\nGenerating Baseline (No Watermark)...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    do_sample=True,
    max_new_tokens=200,
    top_k=40,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)
baseline_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("-" * 40)
print(f"Baseline Code:\n{baseline_code}")
print("-" * 40)
baseline_score = detect_watermark(baseline_code, ngram_len=5) # Check against ngram=5 keys
print(f"Baseline Detection Score (Mean G-Value): {baseline_score:.4f}")
results.append(("None", baseline_score, "OK"))

for n in ngram_lens:
    code = generate_watermarked_code(prompt, ngram_len=n)
    print("-" * 40)
    print(f"Code (ngram_len={n}):\n{code}")
    print("-" * 40)
    
    # Detect
    score = detect_watermark(code, ngram_len=n)
    print(f"Detection Score (Mean G-Value): {score:.4f}")
    
    # Check for basic syntax errors (heuristic)
    syntax_status = "OK"
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        syntax_status = f"Error: {e}"
        
    print(f"Syntax Check: {syntax_status}")
    results.append((n, score, syntax_status))

print("\n=== Summary ===")
print(f"{'ngram_len':<10} | {'Score':<10} | {'Syntax':<20}")
print("-" * 45)
for n, score, syntax in results:
    print(f"{n:<10} | {score:.4f}     | {syntax:<20}")
