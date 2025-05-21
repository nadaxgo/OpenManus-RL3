# OpenManus Demo Environment

This package provides a minimal text environment used for
examples and testing within OpenManus-RL. It does not rely on any
external server. The agent succeeds by replying `finish` within a few
steps. The environment implements the same `BaseEnvClient` interface as
the other AgentGym environments so it can be loaded by
`openmanus_rl.llm_agent.openmanus.OpenManusAgent`.
