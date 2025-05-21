"""FastAPI server exposing the OpenManus python execution environment."""
from fastapi import FastAPI
from .environment import openmanus_env_server
from .model import StepQuery, StepResponse, ResetQuery

app = FastAPI()

@app.post("/create", response_model=int)
async def create_env() -> int:
    return openmanus_env_server.create()

@app.post("/step", response_model=StepResponse)
async def step_env(query: StepQuery) -> StepResponse:
    result = openmanus_env_server.step(query.env_idx, query.action)
    return StepResponse(**result)

@app.get("/observation", response_model=str)
async def observation(env_idx: int) -> str:
    return openmanus_env_server.observation(env_idx)

@app.post("/reset", response_model=str)
async def reset_env(query: ResetQuery) -> str:
    return openmanus_env_server.reset(query.env_idx)
