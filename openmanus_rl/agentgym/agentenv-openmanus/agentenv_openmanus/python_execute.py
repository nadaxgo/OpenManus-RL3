import multiprocessing
import sys
from io import StringIO
from typing import Dict

from .base import BaseTool


class PythonExecute(BaseTool):
    """Execute Python code with timeout and safe globals."""

    name: str = "python_execute"
    description: str = (
        "Executes Python code string. Note: Only print outputs are visible, "
        "function return values are not captured. Use print statements to see results."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:  # pragma: no cover - execution error path
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def execute(self, code: str, timeout: int = 5) -> Dict:
        """Run the provided Python code with a timeout."""
        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"Execution timeout after {timeout} seconds",
                    "success": False,
                }
            return dict(result)
