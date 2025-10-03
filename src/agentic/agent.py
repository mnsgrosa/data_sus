import typing
from src.agentic.agent_tools.tools import (
    read_csv, 
    get_data_dict, 
    summarize_numerical_data,
    generate_statistical_report,
    generate_temporal_graphical_report
)
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from src.utils.logger import MainLogger


class ReportInfo(TypedDict):
    previous_messages: List[HumanMessage | AIMessage]
    report: Dict[str, Any]
    insights: List[str]


class StatisticalAgent(MainLogger):
    def __init__(self):
        self.tools = [read_csv, get_data_dict, summarize_numerical_data, generate_statistical_report, generate_temporal_graphical_report]
        self.llm_tool_caller = ChatOllama(model = "deepseek-r1:1.5b")
        self.llm_tool_caller.bind_tools(self.tools)
        self.graph = StateGraph()
        self._init_graph()
        self.history = []
        self.data = {}

    def _init_graph(self):
        self.graph.add_node("assistant", START)
        self.graph.add_node("llm_tool_caller", ToolNode(self.tools))
        self.graph.add_edge("assistant", "llm_tool_caller")
        self.graph.add_edge("llm_tool_caller", "assistant")

if __name__ == "__main__":
    agent = StatisticalAgent()