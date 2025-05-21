from typing import Any, Mapping

from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class OpenManusEnvClient(BaseEnvClient):
    """Lightweight built-in environment for demonstrations."""

    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are in a demo environment. Reply 'finish' to succeed.",
            }
        ),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Understood."}),
    )

    def __init__(self, env_server_base: str = "", data_len: int = 1, *args, timeout: int = 300, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.data_len = data_len
        self.timeout = timeout
        self.reset(0)

    def __len__(self) -> int:
        return self.data_len

    def observe(self) -> str:
        return self.state

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        action = action.split("Action:")[-1].strip().lower()
        self.steps += 1
        reward = 0.0
        done = False
        if "finish" in action:
            self.state = "Task completed."
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            self.state = "Task failed."
            done = True
        else:
            self.state = f"Received: {action}"
        self.done = done
        return StepOutput(state=self.state, reward=reward, done=done)

    def reset(self, idx: int = 0) -> Mapping[str, Any]:
        self.steps = 0
        self.max_steps = 3
        self.state = "Demo start. Reply 'finish' to end successfully."
        self.done = False
        return {"observation": self.state, "reward": 0.0, "done": False, "score": 0.0}


class OpenManusTask(BaseTask):
    env_client_cls = OpenManusEnvClient
    env_name = "OpenManusDemo"

    def __init__(self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)


