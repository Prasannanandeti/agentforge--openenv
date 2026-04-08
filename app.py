from fastapi import FastAPI, HTTPException
from env.environment import AgentForgeEnv
from env.models import Action

app = FastAPI(title="AgentForge API")

# Create environment instance

env = AgentForgeEnv()

@app.post("/reset")
def reset(task_id: str = "easy_1"):
    try:
        obs = env.reset(task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
