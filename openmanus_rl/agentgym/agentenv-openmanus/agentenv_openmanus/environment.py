from __future__ import annotations
import asyncio
from typing import Dict
from .python_execute import PythonExecute

class OpenManusEnvServer:
    """Simple environment executing Python code via OpenManus's PythonExecute tool."""

    def __init__(self) -> None:
        self._max_id = 0
        self.envs: Dict[int, Dict[str, str]] = {}

    def create(self) -> int:
        env_id = self._max_id
        self._max_id += 1
        self.envs[env_id] = {"obs": ""}
        return env_id

    def reset(self, env_idx: int) -> str:
        if env_idx in self.envs:
            self.envs[env_idx]["obs"] = ""
        return self.envs.get(env_idx, {}).get("obs", "")

    def observation(self, env_idx: int) -> str:
        return self.envs.get(env_idx, {}).get("obs", "")

    async def _execute(self, code: str) -> Dict:
        tool = PythonExecute()
        return await tool.execute(code)

    def step(self, env_idx: int, action: str) -> Dict:
        """Execute python code and return observation, reward and done."""
        res = asyncio.run(self._execute(action))
        obs = res.get("observation", "")
        success = bool(res.get("success", False))
        self.envs[env_idx]["obs"] = obs
        reward = 1.0 if success else -1.0
        return {"state": obs, "reward": reward, "done": False, "info": None}

openmanus_env_server = OpenManusEnvServer()
