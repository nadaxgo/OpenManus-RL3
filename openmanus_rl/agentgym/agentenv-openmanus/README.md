# OpenManus Environment

This environment exposes a simple server that executes Python code using a
lightweight `PythonExecute` tool. The tool implementation is included directly
in this package so no external OpenManus installation is required. It serves as
a minimal example showing how OpenManus functionalities can be integrated with
the AgentGym framework.

Start the server with:

```bash
python -m agentenv_openmanus.launch --port 8000
```
