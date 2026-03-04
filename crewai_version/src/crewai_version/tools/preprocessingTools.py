import subprocess, sys, traceback
from crewai.tools import tool

@tool("Run Python Script")
def run_python_script_tool(code: str) -> str:
    """
    Executes Python code in a temporary script file.
    Returns stdout if successful, stderr if failed.
    """
    filename = "executables/code_to_run.py"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            return stdout or "Execution successful (no output)."
        return f"Execution failed:\n{stderr}"

    except Exception:
        return f"Unexpected error:\n{traceback.format_exc()}"