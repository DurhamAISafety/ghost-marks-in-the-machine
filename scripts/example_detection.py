"""
Example usage of the watermark detection utility.

This script demonstrates how to use the detector_utils module
to check if Python code is watermarked.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector_utils import WatermarkDetector, detect_watermark, get_watermark_score


def example_simple_detection():
    """Example 1: Simple one-off detection."""
    print("=" * 70)
    print("Example 1: Simple Detection")
    print("=" * 70)
    
    # Some example code to test
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
"""
    
    # Simple detection
    is_watermarked = detect_watermark(code, ngram_len=5, threshold=0.5)
    print(f"\nCode is watermarked: {is_watermarked}")
    
    # Get the score
    score = get_watermark_score(code, ngram_len=5)
    print(f"Watermark score: {score:.4f}")
    print()


def example_detector_instance():
    """Example 2: Using a detector instance for multiple checks."""
    print("=" * 70)
    print("Example 2: Detector Instance (Efficient for Multiple Checks)")
    print("=" * 70)
    
    # Create a detector instance (more efficient for multiple checks)
    detector = WatermarkDetector(ngram_len=5)
    
    # Test multiple code samples
    code_samples = [
        "def add(a, b):\n    return a + b",
        "print('Hello, World!')",
        "for i in range(10):\n    print(i ** 2)",
    ]
    
    for i, code in enumerate(code_samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Code: {code[:50]}...")
        
        # Get detailed detection results
        result = detector.detect(code, threshold=0.5)
        
        print(f"Watermarked: {result['is_watermarked']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Confidence: {result['confidence']}")
    print()


def example_custom_threshold():
    """Example 3: Using different thresholds."""
    print("=" * 70)
    print("Example 3: Custom Thresholds")
    print("=" * 70)
    
    code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
    
    detector = WatermarkDetector(ngram_len=5)
    score = detector.get_score(code)
    
    print(f"\nCode snippet:")
    print(code[:100] + "...")
    print(f"\nDetection score: {score:.4f}")
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        is_watermarked = score >= threshold
        print(f"Threshold {threshold:.1f}: {'✓ Watermarked' if is_watermarked else '✗ Not watermarked'}")
    print()


def example_interactive():
    """Example 4: Interactive detection."""
    print("=" * 70)
    print("Example 4: Interactive Detection")
    print("=" * 70)
    print("\nEnter Python code to check for watermarks.")
    print("(Type 'END' on a new line when done, or Ctrl+C to quit)\n")
    
    try:
        detector = WatermarkDetector(ngram_len=5)
        
        lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        
        code = '\n'.join(lines)
        
        if code.strip():
            result = detector.detect(code, threshold=0.5)
            
            print("\n" + "=" * 70)
            print("Detection Results:")
            print("=" * 70)
            print(f"Watermarked: {result['is_watermarked']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Confidence: {result['confidence']}")
            print(f"Detector: ngram_len={result['ngram_len']}")
        else:
            print("\nNo code provided.")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("WATERMARK DETECTION EXAMPLES")
    print("=" * 70)
    print()
    
    try:
        # Run examples
        example_simple_detection()
        example_detector_instance()
        example_custom_threshold()
        
        # Optionally run interactive mode
        print("\nWould you like to try interactive mode? (y/n): ", end='')
        response = input().strip().lower()
        if response == 'y':
            print()
            example_interactive()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train a detector first:")
        print("  python scripts/train_bayesian_detector.py --train")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
