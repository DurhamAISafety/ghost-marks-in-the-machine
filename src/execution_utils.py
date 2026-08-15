import multiprocessing
import io
import contextlib
import traceback
import sys

TIMEOUT_SECONDS = 5

# SECURITY NOTE: this executes model-generated code with a full builtins
# namespace. The ONLY isolation is the subprocess + timeout below. It is NOT a
# security sandbox (full builtins are trivially escapable) and must not be run on
# untrusted/adversarial input without OS-level isolation (Docker/nsjail).
# A previously restricted __builtins__ whitelist was removed because it omitted
# __import__, so any generated solution using `import ...` (nearly all of them)
# raised ImportError and was misclassified as a Runtime Error, confounding the
# error-rate results.


def _run_code_process(code, input_str, queue):
    if not isinstance(input_str, str):
        input_str = ""

    stdout = io.StringIO()
    original_stdin = sys.stdin
    try:
        # Feed the test case via stdin so both input() and sys.stdin.read()
        # based solutions work; EOF raises EOFError, matching real judging.
        sys.stdin = io.StringIO(input_str)
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__name__": "__main__"})
        queue.put(("Passed", stdout.getvalue()))
    except Exception:
        queue.put(("Runtime Error", traceback.format_exc()))
    finally:
        sys.stdin = original_stdin


def _run_single_case(code, input_str):
    """Run code once against one input; return (status, output)."""
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


def run_code_safely(code, input_cases, expected_outputs=None):
    """Run generated code against ALL provided APPS test cases.

    Returns "Passed (Correct)" only if every case runs and matches its expected
    output. The first failing case short-circuits and is reported.
    """
    if not input_cases:
        return "Skipped", "No test cases"

    expected_outputs = expected_outputs or []

    output = ""
    for i, input_str in enumerate(input_cases):
        expected = expected_outputs[i] if i < len(expected_outputs) else None

        status, output = _run_single_case(code, input_str)
        if status != "Passed":
            # Runtime Error / Timeout / Error
            return status, output

        output = output.strip()
        if expected is None:
            continue  # nothing to check against

        expected = str(expected).strip()
        if output != expected:
            return "Passed (Wrong Output)", (
                f"Case {i + 1}/{len(input_cases)}\nExpected: {expected}\nGot: {output}"
            )

    if not any(e is not None for e in expected_outputs[: len(input_cases)]):
        return "Passed (No Check)", output
    return "Passed (Correct)", output


def _demo():
    """Self-check: imports work, stdin works, all cases must pass. Run: python -m src.execution_utils"""
    imports_code = "import math\nprint(int(math.sqrt(int(input()))))"
    assert run_code_safely(imports_code, ["16"], ["4"])[0] == "Passed (Correct)", "imports must run"

    multi = "print(int(input()) * 2)"
    assert run_code_safely(multi, ["2", "5"], ["4", "10"])[0] == "Passed (Correct)", "all cases pass"
    # second case wrong -> flagged, not silently passed on the first case
    assert run_code_safely(multi, ["2", "5"], ["4", "99"])[0] == "Passed (Wrong Output)", "checks every case"

    assert run_code_safely("1/0", ["1"], ["1"])[0] == "Runtime Error", "runtime errors caught"
    print("execution_utils self-check passed")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    _demo()
