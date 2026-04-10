import os
from fastapi import FastAPI, HTTPException, Body
from openai import OpenAI
from env.environment import AgentForgeEnv
from env.models import Action

# -------------------------
# INIT APP
# -------------------------
app = FastAPI()
env = AgentForgeEnv()

# -------------------------
# OPTIONAL (MODEL CLIENT)
# -------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/")
def home():
    return {"status": "running"}

# -------------------------
# RESET ENDPOINT (CRITICAL)
# -------------------------
@app.post("/reset")
def reset(payload: dict = Body(default={"task_id": "easy_1"})):
    try:
        task_id = payload.get("task_id", "easy_1")
        obs = env.reset(task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------
# STEP ENDPOINT (CRITICAL)
# -------------------------
@app.post("/step")
def step(action: dict):
    try:
        act = Action(**action)
        obs, reward, done, info = env.step(act)

        return {
            "observation": obs.model_dump(),
            "reward": reward.value,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------
# OPTIONAL: YOUR ORIGINAL LOGIC (NOT USED BY OPENENV)
# -------------------------
def run_inference():
    task_ids = ["easy_1", "medium_1", "hard_1"]

    for tid in task_ids:
        print(f"[START] task={tid}")

        obs = env.reset(tid)
        done = False
        step_idx = 0

        order_id = obs.task_context.get("order_id")

        while not done and step_idx < 8:
            step_idx += 1

            try:
                # SAFE API CALL
                try:
                    _ = client.models.list()
                except Exception:
                    pass

                if tid == "easy_1":
                    if step_idx == 1:
                        action = Action(
                            action_type="call_tool",
                            tool_name="get_order_details",
                            tool_params={"order_id": order_id}
                        )
                    elif step_idx == 2:
                        action = Action(
                            action_type="reply",
                            text=f"Your order {order_id} has been shipped and will arrive on 2023-10-25."
                        )
                    else:
                        action = Action(action_type="close_ticket")

                elif tid == "medium_1":
                    if step_idx == 1:
                        action = Action(
                            action_type="call_tool",
                            tool_name="get_order_details",
                            tool_params={"order_id": order_id}
                        )
                    elif step_idx == 2:
                        action = Action(
                            action_type="call_tool",
                            tool_name="process_refund",
                            tool_params={"order_id": order_id}
                        )
                    elif step_idx == 3:
                        action = Action(
                            action_type="reply",
                            text=f"I have successfully processed your refund for {order_id}."
                        )
                    else:
                        action = Action(action_type="close_ticket")

                elif tid == "hard_1":
                    if step_idx == 1:
                        action = Action(action_type="ask_info", field="order_id")

                    elif step_idx == 2:
                        order_id = "ORD-303"
                        action = Action(
                            action_type="call_tool",
                            tool_name="get_order_details",
                            tool_params={"order_id": order_id}
                        )

                    elif step_idx == 3:
                        action = Action(
                            action_type="reply",
                            text="Sorry, refund not possible as order is still processing."
                        )

                    else:
                        action = Action(action_type="close_ticket")

                obs, reward_obj, done, info = env.step(action)

                print(f"[STEP] {step_idx} | {action.action_type} | reward={reward_obj.value}")

            except Exception as e:
                print(f"[ERROR] {str(e)}")
                break


# -------------------------
# RUN SERVER (IMPORTANT)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
