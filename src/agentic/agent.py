import typing
from src.agentic.agent_tools.tools import (
    store_csvs, 
    get_data_dict, 
    summarize_numerical_data,
    generate_statistical_report,
    generate_temporal_graphical_report
)
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.utils.logger import MainLogger


class ReportInfo(TypedDict):
    messages: List[HumanMessage | AIMessage]
    report: List[Dict[str, Any]]
    insights: List[str]
    struct: Dict[str, Any]
    summary: List[Dict[str, Any]]
    stat_report: List[Dict[str, Any]]

class StatisticalAgent(MainLogger):
    def __init__(self):
        super().__init__(__name__)
        self.tools = [store_csvs, get_data_dict, summarize_numerical_data, generate_statistical_report, generate_temporal_graphical_report]
        self.llm_tool_caller = ChatOllama(model = "deepseek-r1:1.5b")
        self.llm_tool_caller.bind_tools(self.tools)
        self.graph = StateGraph(ReportInfo)
        self._init_graph()
        self.history = []
        self.data = {}

    def _init_graph(self):
        self.graph.add_node("assistant", self.assistant)
        self.graph.add_node("llm_tool_caller", ToolNode(self.tools))
        self.graph.add_edge(START, "assistant")
        self.graph.add_edge("assistant", "llm_tool_caller")
        self.graph.add_edge("llm_tool_caller", "assistant")
        self.graph.add_edge("assistant", END)
        self.react_graph = self.graph.compile()

    def assistant(self, state: ReportInfo):
        textual_description_of_tool = """
            store_csvs() -> Dict[str, Any]: Fetches and stores the 'srag' dataset into the database.
            get_data_dict() -> Dict[str, Any]: Retrieves the data dictionary for the 'srag' dataset.
            summarize_numerical_data(year:str, column_name: str, mean: Optional[bool] = True, median: 
            Optional[bool] = True, std: Optional[bool] = True, min: Optional[bool] = True, max: Optional[bool] = True) -> Dict[str, Any]:
            ARGS:
                year: str: Year of the data to be analyzed. Valid values are "all", "2019", "2020", "2021", "2022", "2023", "2024", "2025".
                column: str: The column to summarize.
                mean: Optional[bool]: Whether to calculate the mean. Default is True.
                median: Optional[bool]: Whether to calculate the median. Default is True.
                std: Optional[bool]: Whether to calculate the standard deviation. Default is True.
                min: Optional[bool]: Whether to calculate the minimum. Default is True.
                max: Optional[bool]: Whether to calculate the maximum. Default is True.
            RETURNS:
                A summary of the data in the specified column.

            generate_statistical_report(year: str, state: Optional[str], start_analisys_period: str, end_analisys_period: str, granularity: Optional[str] = 'D') -> Dict[str, Any]:
            ARGS:
                year: str: Year of the data to be analyzed. Valid values are "all", "2019", "2020", "2021", "2022", "2023", "2024", "2025".
                state: Optional[str]: The state to filter the data. If None, data for all states will be used.
                start_analisys_period: str: The start date of the analysis period in the format 'YYYY-MM-DD'.
                end_analisys_period: str: The end date of the analysis period in the format 'YYYY-MM-DD'.
                granularity: Optional[str]: The granularity of the analysis. Valid values are 'D' (daily), 'W' (weekly), 'M' (monthly). Default is 'D'.
            RETURNS:
                A statistical report containing key metrics and insights.

            generate_temporal_graphical_report(year: str, state: Optional[str], start_analisys_period: str, end_analisys_period: str, granularity: Optional[str] = 'D') -> Dict[str, Any]:
            ARGS:
                year: str: Year of the data to be analyzed. Valid values are "all",
                "2019", "2020", "2021", "2022", "2023", "2024", "2025".
                state: Optional[str]: The state to filter the data. If None, data for all
                states will be used.
                start_analisys_period: str: The start date of the analysis period in the
                format 'YYYY-MM-DD'.
                end_analisys_period: str: The end date of the analysis period in the
                format 'YYYY-MM-DD'.
                granularity: Optional[str]: The granularity of the analysis. Valid values
                are 'D' (daily), 'W' (weekly), 'M' (monthly). Default is 'D'.
                year: str: Year of the data to be analyzed. Valid values are "all", "2019", "2020", "2021", "2022", "2023", "2024", "2025".
            RETURNS:
                A graphical report containing visualizations of key metrics over time.
        """

        report = state.get("report", {})
        previous_messages = state.get("previous_messages", [])
        insights = state.get("insights", [])
        struct = state.get("struct", {})
        sys_message = SystemMessage(content = f"""
            You are a statistical analist agent. You have access to the following tools:\n{textual_description_of_tool}
            and currently this is your chat history:\n{previous_messages} and the report you generated is:\n{report}
            Now, based on this informations you should operate the tools whenever prompted or give insights about the report.
        """)

        ans = self.llm_tool_caller.invoke([sys_message] + previous_messages)

        figures = []

        if hasattr(ans, 'tool_calls'):
            for tool_call in ans.tool_calls:
                if 'figure_id' in tool_call.get('result', {}):
                    fig_id = tool_call['result']['figure_id']
                    figures.append(FIGURE_STORE.get(fig_id))

                if 'report' in tool_call.get('result', {}):
                    state['report'].append(tool_call['result']['report'])
                
                if 'mean' in tool_call.get('result', {}) or 'median' in tool_call.get('result', {}) or 'std' in tool_call.get('result', {}) or 'min' in tool_call.get('result', {}) or 'max' in tool_call.get('result', {}):
                    state['summary'].append(tool_call['result'])

                if '_' in tool_call.get('result', {}).keys():
                    state['struct'] = tool_call['result']

        return {"messages": [ans + previous_messages], "figures": figures, "report": state['report'], "summary": state['summary'], "struct": state['struct'], "insights": insights + [ans.content]}

if __name__ == "__main__":
    agent = StatisticalAgent()

    message = [HumanMessage(content = "Fetch and store the 'srag' dataset into the database.")]

    print(agent.react_graph.invoke({"messages": message}))