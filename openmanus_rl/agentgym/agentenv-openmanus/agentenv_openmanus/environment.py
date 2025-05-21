from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Tuple

from agentenv_gaia.tool_manager import ToolManager
from gaia.base import ToolResult

class OpenManusEnvServer:
    """Minimal in-process environment executing GAIA tools directly."""

    def __init__(self) -> None:
        self._max_id = 0
        self.envs: Dict[int, Dict[str, str]] = {}
        self.tool_manager = ToolManager()

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

    def _parse_action(self, action: str) -> Tuple[str, Dict[str, Any]]:
        """Parse an action string into tool name and parameters."""
        if "<tool_call>" in action and "</tool_call>" in action:
            raw = action.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
            try:
                payload = json.loads(raw)
                name = payload.get("name") or payload.get("tool_name")
                args = payload.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"input": args}
                return name, args
            except Exception:
                return action.strip(), {}

        m = re.search(r"Action:(.*)", action)
        if m:
            remain = m.group(1).strip()
            if "Action Input:" in remain:
                tool, inp = remain.split("Action Input:", 1)
                return tool.strip(), {"input": inp.strip()}
            return remain.strip(), {}
        return action.strip(), {}

    async def _execute(self, tool: str, params: Dict[str, Any]) -> Dict:
        return await self.tool_manager.execute_tool(tool, **params)

    def step(self, env_idx: int, action: str) -> Dict:
        """Execute a tool and return observation, reward and done."""
        tool, params = self._parse_action(action)
        res = asyncio.run(self._execute(tool, params))

        if isinstance(res, ToolResult):
            success = not bool(res.error)
            obs = str(res)
        else:
            success = bool(res.get("success", True))
            obs = res.get("observation", str(res))

        self.envs[env_idx]["obs"] = obs
        reward = 1.0 if success else -1.0
        done = tool == "terminate"
        return {"state": obs, "reward": reward, "done": done, "info": None}

openmanus_env_server = OpenManusEnvServer()
