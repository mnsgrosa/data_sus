from fastapi import FastAPI
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from src.agentic.agent_schema.main_schema import AgentResponse
from src.utils.logger import MainLogger

from ..agentic.agent import StatisticalAgent
from . import config

app = FastAPI()
logger = MainLogger(__name__)

agent = StatisticalAgent()


class AgentRequest(BaseModel):
    prompt: str = Field(...)


def parse_agent_response(state: dict) -> AgentResponse:
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    content = last_message.content if last_message else ""

    tool_name = "unknown"
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_name = msg.tool_calls[0]["name"]
            break

    data_points = {}
    if state.get("data"):
        data_points = state["data"]
    elif state.get("summary"):
        data_points = state["summary"][-1]
    elif state.get("x"):
        data_points = state["stat_report"][-1]

    return AgentResponse(
        content=str(content),
        data=data_points if isinstance(data_points, dict) else {},
        tool_name=tool_name,
    )


@app.post("/prompt", response_model=AgentResponse)
async def chat_endpoint(user_input: str):  # Example signature
    result_state = agent.run(user_input)
    response = parse_agent_response(result_state)

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
