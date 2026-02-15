import sys
import io
import contextlib

def execute_code_safe(code):
    """
    Executes the provided Python code in a restricted environment.
    Captures stdout and returns it.
    """
    # Create a buffer to capture stdout
    output_buffer = io.StringIO()
    
    # Define restricted globals
    # We allow basic math, string ops, etc.
    # But we block IO, OS, Networks generally by not importing them.
    # However, Python exec is not a true sandbox. 
    # This relies on the user approval step as the primary safety gate.
    
    safe_globals = {
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
            "abs": abs,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            # Add more safe builtins as needed
        }
    }
    
    # We capture stdout using contextlib
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals)
        return output_buffer.getvalue()
    except Exception as e:
        return f"Error executing code: {e}"

if __name__ == "__main__":
    # Test
    code = "print('Hello world'); x = 5 + 5; print(x)"
    print(execute_code_safe(code))
