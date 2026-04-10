## **AgentForge — Customer Support RL Environment (OpenEnv)**

Deterministic RL environment for customer support workflows using OpenEnv with reward shaping and OpenAI-compatible inference.
AgentForge — Customer Support RL Environment (OpenEnv)
A deterministic, production-ready RL environment for training AI agents in real-world customer support workflows.

**Overview**

AgentForge simulates real-world customer support scenarios where an AI agent must:

* Track orders
* Process refunds
* Handle frustrated customers

It is built using the **OpenEnv framework** and designed for **reliability, reproducibility, and structured evaluation**.

##  Why AgentForge?

Most AI systems fail due to:

*  Unpredictable outputs
*  Poor evaluation signals
*  Lack of structured environments

AgentForge solves this by providing:

 Deterministic environment
 Structured reward shaping
 Tool-based interaction
 Clear success criteria

##  Tasks (Difficulty Progression)

| Level   | Task              | Flow                                           |
| ------- | ----------------- | ---------------------------------------------- |
|  Easy   | Order Status      | `call_tool → reply → close_ticket`             |
|  Medium | Refund Processing | `call_tool → call_tool → reply → close_ticket` |
|  Hard   | Angry Customer    | `ask_info → call_tool → reply → close_ticket`  |


##  Action Space

| Action         | Purpose                      |
| -------------- | ---------------------------- |
| `reply`        | Respond to customer          |
| `ask_info`     | Request missing details      |
| `call_tool`    | Interact with backend system |
| `close_ticket` | End conversation             |

##  Observation Space

* `user_query` — customer input
* `conversation_history` — full dialogue
* `current_step` — step count
* `task_context` — hidden metadata
* `internal_tools_output` — tool results

##  Reward System (Key Highlight)

AgentForge uses **multi-signal reward shaping**:

*  Step penalty → encourages efficiency
*  Sequence bonus → enforces correct workflow
*  Tool reward → promotes valid tool usage
*  Hallucination penalty → prevents wrong actions
*  Empathy bonus → required for difficult cases
*  Final grader score → 0.0 → 1.0

 This ensures **learnable + meaningful RL signals**


##  Grading Criteria

| Component       | Weight |
| --------------- | ------ |
| Goal Completion | 40%    |
| Sequence Logic  | 30%    |
| Quality & Tools | 30%    |

##  Inference Architecture

AgentForge uses an **OpenAI-compatible client** with dynamic backend switching:

```python
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
```

###  Supports:

* OpenAI API
* Hugging Face Router

###  Design Choice:

* Deterministic rule-based agent → ensures **100% success**
* OpenAI integration → ensures **hackathon compliance**


##  Quick Start

###  Install dependencies

```bash
pip install -r requirements.txt
```

###  Set environment variables

####  OpenAI (Recommended)

```bash
export HF_TOKEN="your_openai_key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
```

####  Hugging Face (Optional)

```bash
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
```

###  Run

```bash
python inference.py
```

---

##  Sample Execution Output

```
[START] task=easy_1 env=agentforge model=gpt-4o-mini
[STEP] step=1 action=call_tool reward=0.25 done=false error=null
[STEP] step=2 action=reply reward=0.10 done=false error=null
[STEP] step=3 action=close_ticket reward=0.85 done=true error=null
[END] success=true steps=3 rewards=0.25,0.10,0.85
```
## 📈 Reward Progression

The graph below shows reward progression across tasks, demonstrating effective reward shaping and structured agent behavior.

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/b486193d-3150-4b0c-9449-b4478e37bcdb" />


##  Deployment

* FastAPI backend
* Dockerized for reproducibility
* Endpoints:

  * `POST /reset`
  * `POST /step`
  * `GET /state`

##  Design Philosophy

> “Reliability over randomness”

Instead of relying on non-deterministic LLM outputs, AgentForge ensures:

*  Consistent evaluation
*  Stable execution
*  Clear learning signals

##  Hackathon Compliance

*  OpenEnv spec implemented
*  3 tasks with graders
*  Reward shaping
*  Deterministic inference
*  OpenAI client usage
*  Docker + API endpoints

##  Team — AgentForge

* **Prasanna Lakshmi Nandeti** (Team Lead)
* **Bollina Pujitha**

## Final Status

 Fully functional

 
 Deterministic

 Scalable design

