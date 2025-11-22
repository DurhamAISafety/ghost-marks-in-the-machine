""" Check changed files to see if they have a SynthID watermark. """

# Standard library imports
import sys

# Local imports
from test import test


def main():
    """ Main function. """
    files = sys.argv[1:]

    if not files:
        print("No changed files to check.")
        sys.exit(0)

    failed = []

    for filepath in files:
        print(f"Checking: {filepath}")
        try:
            result = test(filepath)
            if result:
                print("✓ Passed")
            else:
                print("✗ Failed")
                failed.append(filepath)
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
