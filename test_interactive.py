"""
Simple test script for watermark detection.
You can run this in VSCode's interactive window (Jupyter).
"""

# %%
from src.detector_utils import WatermarkDetector

# %% 
# Create detector (automatically uses HF_TOKEN from .env)
print("Loading watermark detector...")
detector = WatermarkDetector(ngram_len=5)
print("✅ Detector loaded successfully!")

# %%
# Test 1: Simple function
code1 = """
def add(a, b):
    return a + b
"""

result1 = detector.detect(code1, threshold=0.5)
print(f"\nTest 1: Simple add function")
print(f"  Watermarked: {result1['is_watermarked']}")
print(f"  Score: {result1['score']:.4f}")
print(f"  Confidence: {result1['confidence']}")

# %%
# Test 2: More complex function
code2 = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
"""

result2 = detector.detect(code2, threshold=0.5)
print(f"\nTest 2: Fibonacci function")
print(f"  Watermarked: {result2['is_watermarked']}")
print(f"  Score: {result2['score']:.4f}")
print(f"  Confidence: {result2['confidence']}")

# %%
# Test 3: Your own code
# Replace with any Python code you want to test
your_code = """
# Paste your code here
def hello():
    print("Hello, world!")
"""

result3 = detector.detect(your_code, threshold=0.5)
print(f"\nTest 3: Your code")
print(f"  Watermarked: {result3['is_watermarked']}")
print(f"  Score: {result3['score']:.4f}")
print(f"  Confidence: {result3['confidence']}")

# %%
# Test different thresholds
print("\n" + "="*50)
print("Testing different thresholds:")
print("="*50)

test_code = "def multiply(x, y): return x * y"
score = detector.get_score(test_code)

print(f"\nCode: {test_code}")
print(f"Raw score: {score:.4f}\n")

for threshold in [0.3, 0.5, 0.7]:
    is_wm = score >= threshold
    print(f"  Threshold {threshold:.1f}: {'✓ Watermarked' if is_wm else '✗ Not watermarked'}")

# %%
print("\n✅ All tests complete!")
