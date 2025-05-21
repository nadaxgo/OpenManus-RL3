from typing import Any, Mapping
from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput
from agentenv_openmanus.environment import openmanus_env_server

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
        # Directly create the environment using the embedded server
        self.env_id = openmanus_env_server.create()

    def __len__(self):
        return self.data_len

    def observe(self) -> str:
        return openmanus_env_server.observation(self.env_id)

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        action = _action[-1].strip()
        result = openmanus_env_server.step(self.env_id, action)
        return StepOutput(state=result["state"], reward=result["reward"], done=result["done"])

    def reset(self, id: int) -> Any:
        return openmanus_env_server.reset(self.env_id)

class OpenManusTask(BaseTask):
    env_client_cls = OpenManusEnvClient
    env_name = "OpenManus"

    def __init__(self, client_args: Mapping[str, Any] | Mapping[str, Any], n_clients: int, *args, **kwargs):
        super().__init__(client_args, n_clients, *args, **kwargs)
