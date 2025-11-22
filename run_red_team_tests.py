"""
Red Team Test Runner for SynthID Watermarking
Systematically tests watermark robustness against adversarial attacks

IMPORTANT: This requires GPU to run. Use sample_code from a previous generation or JSON file.
"""

import json
import torch
from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor
from red_team_attacks import generate_attack_suite, AdversarialTransformer
from model_utils import WATERMARK_KEYS, compute_g_score


def load_watermarked_samples(json_path: str = "results.json", max_samples: int = 5):
    """Load previously generated watermarked code samples"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    for problem in data[:max_samples]:
        for result in problem['results']:
            if result['status'] == 'Passed (Correct)' and result.get('generated_code'):
                samples.append({
                    'problem_id': problem['problem_id'],
                    'ngram_len': result['ngram_len'],
                    'code': result['generated_code'],
                    'original_g_score': result['g_score']
                })
                break  # Take first passing result per problem

    return samples


def run_red_team_tests(code: str, ngram_len: int, tokenizer, device,
                       original_score: float = None, verbose: bool = True):
    """
    Run full suite of adversarial attacks on watermarked code

    Args:
        code: Watermarked code to attack
        ngram_len: Original ngram length used for watermarking
        tokenizer: Tokenizer instance
        device: Device (cuda/cpu)
        original_score: Original detection score (optional)
        verbose: Print detailed results

    Returns:
        List of attack results with detection scores
    """
    # Generate all attacks
    attacks = generate_attack_suite(code)

    results = []

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {len(attacks)} adversarial attacks")
        print(f"Original G-Score: {original_score:.4f}" if original_score else "")
        print(f"{'='*80}\n")

    # Create processor for this ngram_len
    detect_ngram = int(ngram_len) if ngram_len and ngram_len != 'None' else 5

    processor = SynthIDTextWatermarkLogitsProcessor(
        keys=WATERMARK_KEYS,
        ngram_len=detect_ngram,
        sampling_table_size=2**16,
        sampling_table_seed=0,
        context_history_size=1024,
        device=device
    )

    for attack_name, modified_code, metadata in attacks:
        try:
            # Compute detection score on modified code
            score = compute_g_score(tokenizer, device, modified_code,
                                  ngram_len=detect_ngram, processor=processor)

            # Calculate score drop
            score_drop = (original_score - score) if original_score else 0

            result = {
                'attack_name': attack_name,
                'modified_code': modified_code,
                'detection_score': score,
                'score_drop': score_drop,
                'score_drop_pct': (score_drop / original_score * 100) if original_score and original_score > 0 else 0,
                'metadata': metadata,
                'watermark_detected': score > 0.5  # Threshold heuristic
            }

            results.append(result)

            if verbose:
                print(f"Attack: {attack_name:30s} | "
                      f"Score: {score:.4f} | "
                      f"Drop: {score_drop:+.4f} ({result['score_drop_pct']:+.1f}%) | "
                      f"Intensity: {metadata.get('intensity', 'N/A')} | "
                      f"Detected: {'✓' if result['watermark_detected'] else '✗'}")

        except Exception as e:
            if verbose:
                print(f"Attack: {attack_name:30s} | ERROR: {str(e)}")
            results.append({
                'attack_name': attack_name,
                'detection_score': None,
                'error': str(e),
                'metadata': metadata
            })

    return results


def analyze_breaking_point(results: list) -> dict:
    """Analyze at what point the watermark breaks"""

    # Filter to variable rename attacks only
    var_renames = [r for r in results if 'rename' in r['attack_name']
                   and r.get('metadata', {}).get('type') == 'variable_rename']

    if not var_renames:
        return {"breaking_point": "Unknown", "analysis": "No variable rename attacks found"}

    # Sort by intensity
    var_renames.sort(key=lambda x: x['metadata']['intensity'])

    # Find first attack where watermark is no longer detected
    breaking_point = None
    for attack in var_renames:
        if not attack.get('watermark_detected', True):
            breaking_point = attack
            break

    analysis = {
        'breaking_point_attack': breaking_point['attack_name'] if breaking_point else "Not broken",
        'breaking_point_intensity': breaking_point['metadata']['intensity'] if breaking_point else "N/A",
        'breaking_point_score': breaking_point['detection_score'] if breaking_point else "N/A",
        'all_rename_results': [
            {
                'num_renames': r['metadata']['intensity'],
                'score': r['detection_score'],
                'detected': r.get('watermark_detected', False)
            }
            for r in var_renames
        ]
    }

    return analysis


def main():
    """Main test execution"""
    import sys

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # For testing without loading full model, we can use just the tokenizer
    from transformers import AutoTokenizer

    MODEL_NAME = "google/gemma-2b-it"
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Option 1: Load from previous results JSON
    if len(sys.argv) > 1 and sys.argv[1] == "--from-json":
        json_file = sys.argv[2] if len(sys.argv) > 2 else "results.json"
        print(f"\nLoading watermarked samples from {json_file}...")

        try:
            samples = load_watermarked_samples(json_file, max_samples=3)
            print(f"Loaded {len(samples)} watermarked code samples\n")

            all_test_results = []

            for i, sample in enumerate(samples):
                print(f"\n{'#'*80}")
                print(f"# Testing Sample {i+1}/{len(samples)} - Problem ID: {sample['problem_id']}")
                print(f"# Original ngram_len: {sample['ngram_len']}")
                print(f"{'#'*80}")

                ngram_int = int(sample['ngram_len']) if sample['ngram_len'] != 'None' else 5

                results = run_red_team_tests(
                    code=sample['code'],
                    ngram_len=ngram_int,
                    tokenizer=tokenizer,
                    device=device,
                    original_score=sample['original_g_score'],
                    verbose=True
                )

                # Analyze breaking point
                analysis = analyze_breaking_point(results)

                print(f"\n--- Breaking Point Analysis ---")
                print(f"Breaking point: {analysis['breaking_point_attack']}")
                print(f"Intensity at break: {analysis['breaking_point_intensity']}")
                print(f"Score at break: {analysis['breaking_point_score']}")

                all_test_results.append({
                    'problem_id': sample['problem_id'],
                    'ngram_len': sample['ngram_len'],
                    'original_code': sample['code'],
                    'original_score': sample['original_g_score'],
                    'attacks': results,
                    'analysis': analysis
                })

            # Save comprehensive results
            output_file = "red_team_results.json"
            with open(output_file, 'w') as f:
                json.dump(all_test_results, f, indent=2)
            print(f"\n\nRed team results saved to {output_file}")

        except FileNotFoundError:
            print(f"ERROR: {json_file} not found. Please run pipeline.py first to generate watermarked samples.")
            sys.exit(1)

    # Option 2: Test on a single code snippet (manual mode)
    else:
        print("\nManual test mode - testing on sample code snippet")
        print("(Use --from-json <file> to test on previously generated samples)\n")

        # Example watermarked code (you can replace this)
        sample_code = """def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def solve():
    num = int(input())
    print(fibonacci(num))

solve()
"""

        ngram_len = 5
        print(f"Testing with ngram_len={ngram_len}")

        results = run_red_team_tests(
            code=sample_code,
            ngram_len=ngram_len,
            tokenizer=tokenizer,
            device=device,
            original_score=None,  # We don't know original score
            verbose=True
        )

        analysis = analyze_breaking_point(results)

        print(f"\n--- Breaking Point Analysis ---")
        print(f"Breaking point: {analysis['breaking_point_attack']}")
        print(f"Intensity: {analysis['breaking_point_intensity']}")

        # Save results
        output = [{
            'problem_id': 'manual_test',
            'ngram_len': ngram_len,
            'original_code': sample_code,
            'attacks': results,
            'analysis': analysis
        }]

        with open("red_team_results.json", 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to red_team_results.json")


if __name__ == "__main__":
    main()
