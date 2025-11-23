""" Check changed files to see if they have a SynthID watermark. """

# Standard library imports
import sys

# Local imports
from src.detector_utils import WatermarkDetector


def load_file(filepath: str) -> str:
    """ Loads in file from filepath. """
    with open(filepath, "r") as f:
        file_content = f.read()
    return file_content


def main():
    """ Main function. """
    # Create detector (loads .env automatically)
    detector = WatermarkDetector(ngram_len=5)

    files = sys.argv[1:]

    if not files:
        print("No changed files to check.")
        sys.exit(0)

    failed = []

    for filepath in files:
        print(f"Checking: {filepath}")
        try:
            # Load in the changed file
            code = load_file(filepath)

            # Detect watermarks in any Python code string
            result = detector.detect(code, threshold=0.5)

            if result['is_watermarked']:
                print(f"✗ Watermark detected: code likely AI generated,\n confidence: {result['confidence']}")
                failed.append(filepath)

            else:
                print(f"✓ Code unlikely AI generated,\n confidence: {result['confidence']}")

        except Exception as e:
            print(f"✗ Error: {e}")
            failed.append(filepath)

    print(f"\n{'='*40}")
    print(f"Results: {len(files) - len(failed)}/{len(files)} files passed")

    if failed:
        print(f"\nFailed files:")
        for f in failed:
            print(f"  - {f}")
         # Exit with error code to fail the CI
        sys.exit(1)
    # No SynthID watermark detected
    sys.exit(0)


if __name__ == "__main__":
    main()
