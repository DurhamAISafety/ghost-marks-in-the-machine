import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, SynthIDTextWatermarkingConfig
from transformers import SynthIDTextWatermarkLogitsProcessor

MODEL_NAME = "google/gemma-2b-it"
WATERMARK_KEYS = [101, 202, 303, 404, 505, 606, 707, 808, 909]

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
    # Use chat template for instruction-tuned model
    messages = [
        {"role": "user", "content": f"Please write a Python script for the following problem. The answer should be:\n\ndef solve():\n    # define function here with no arguments, which reads input from stdin and prints output to stdout\n\nsolve()\n\n{prompt}\n\nEnsure the code is inside a ```python block."}
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
