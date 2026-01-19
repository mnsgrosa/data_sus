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

    # 1. Extract Content
    content = last_message.content if last_message else ""

    # 2. Extract Tool Name (Look backwards for the last tool call)
    tool_name = "unknown"
    # Iterate backwards to find the most recent tool call
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_name = msg.tool_calls[0]["name"]
            break

    # 3. Extract Data (Look for the result of the tool)
    data_points = {}
    # You might need to adjust logic to find specific data depending on your flow
    # This example looks for the last 'summary' or 'stat_report' entry
    if state.get("summary"):
        data_points = state["summary"][-1]
    elif state.get("stat_report"):
        data_points = state["stat_report"][-1]

    # Handle cases where data is missing (AgentResponse requires 'data' field)
    # If the agent just chatted (error case), data might be empty.
    # You might need to relax the AgentResponse model to allow Optional[Dict]
    # or provide a default empty dict here.
    return AgentResponse(
        content=str(content),
        data=data_points if isinstance(data_points, dict) else {},
        tool_name=tool_name,
    )


@app.post("/prompt", response_model=AgentResponse)
async def chat_endpoint(user_input: str):  # Example signature
    agent = StatisticalAgent()

    # Run the agent
    result_state = agent.run(user_input)

    # Map the dict result to your Pydantic model
    response = parse_agent_response(result_state)

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
