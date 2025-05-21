from typing import Any, Mapping
import requests
from requests.exceptions import RequestException
from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput

class OpenManusEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage({
            "from": "human",
            "loss": None,
            "value": "You can execute python code using the <action> tag."
        }),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")
        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Mapping[str, Any]) -> Mapping[str, Any]:
        data = dict(data)
        data["env_idx"] = self.env_id
        res = requests.post(f"{self.env_server_base}/{path}", json=data, timeout=self.timeout)
        res.raise_for_status()
        return res.json()

    def _get(self, path: str) -> Mapping[str, Any]:
        res = requests.get(f"{self.env_server_base}/{path}?env_idx={self.env_id}", timeout=self.timeout)
        res.raise_for_status()
        return res.json()

    def observe(self) -> str:
        return self._get("observation")

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        action = _action[-1].strip()
        response = self._post("step", {"action": action})
        return StepOutput(state=response["state"], reward=response["reward"], done=response["done"])

    def reset(self, id: int) -> Any:
        return self._post("reset", {"session_id": id})

class OpenManusTask(BaseTask):
    env_client_cls = OpenManusEnvClient
    env_name = "OpenManus"

    def __init__(self, client_args: Mapping[str, Any] | Mapping[str, Any], n_clients: int, *args, **kwargs):
        super().__init__(client_args, n_clients, *args, **kwargs)
