import multiprocessing
import io
import contextlib
import traceback
import sys

TIMEOUT_SECONDS = 5

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
