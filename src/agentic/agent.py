import typing
import json
from src.agentic.agent_tools.tools import (
    store_csvs, 
    get_data_dict, 
    summarize_numerical_data,
    generate_statistical_report,
    generate_temporal_graphical_report
)
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END 
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.utils.logger import MainLogger
from pydantic import BaseModel, Field


class ReportInfo(TypedDict):
    messages: List[HumanMessage | AIMessage]
    report: List[Dict[str, Any]]
    insights: List[str]
    struct: Dict[str, Any]
    summary: List[Dict[str, Any]]
    stat_report: List[Dict[str, Any]]
    figures: List[Any]


class StatisticalAgent(MainLogger):
    def __init__(self):
        super().__init__(__name__)
        self.tools = [
            store_csvs, 
            get_data_dict, 
            summarize_numerical_data,
            generate_statistical_report,
            generate_temporal_graphical_report
        ]
        
        # Create a tool map for execution
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        self.llm_tool_caller = ChatOllama(model="qwen2.5:14b")
        self.llm_tool_caller = self.llm_tool_caller.bind_tools(self.tools)
        self.graph = StateGraph(ReportInfo)
        self._init_graph()
        self.history = []
        self.data = {}

    def _init_graph(self):
        self.graph.add_node("assistant", self.assistant)
        self.graph.add_node("tools", self.call_tools)  # Use custom tool caller
        self.graph.add_edge(START, "assistant")
        self.graph.add_conditional_edges(
            "assistant",
            self.should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        self.graph.add_edge("tools", "assistant")        
        self.react_graph = self.graph.compile()

    def _serialize_for_json(self, obj: Any) -> Any:
        """Recursively serialize objects for JSON compatibility"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, date
        
        # Handle Plotly figures specially - don't serialize them
        if hasattr(obj, '_data_class_name') or (hasattr(obj, '__module__') and 'plotly' in str(obj.__module__)):
            # Return a reference instead of the full figure
            return {"_plotly_figure": True, "_type": str(type(obj).__name__)}
        
        if isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_dict') and not hasattr(obj, '_data_class_name'):
            # Handle pandas DataFrames, Series, etc. but NOT Plotly figures
            return self._serialize_for_json(obj.to_dict())
        else:
            return obj

    def call_tools(self, state: ReportInfo) -> Dict[str, List]:
        """Custom tool execution that preserves dict returns"""
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_messages = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            # Type coercion for common issues
            if tool_name == "generate_statistical_report":
                # Convert month integers to strings
                if "starting_month" in tool_args:
                    tool_args["starting_month"] = str(tool_args["starting_month"])
                    self.logger.info(f"Coerced starting_month to string: {tool_args['starting_month']}")
                
                if "ending_month" in tool_args:
                    tool_args["ending_month"] = str(tool_args["ending_month"])
                    self.logger.info(f"Coerced ending_month to string: {tool_args['ending_month']}")
            
            self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            try:
                tool = self.tool_map[tool_name]
                result = tool.invoke(tool_args)
                
                # Ensure result is a dict
                if not isinstance(result, dict):
                    self.logger.warning(f"Tool {tool_name} returned non-dict: {type(result)}")
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except json.JSONDecodeError:
                            result = {"result": result}
                    else:
                        result = {"result": result}
                
                # Serialize the result for JSON compatibility
                serialized_result = self._serialize_for_json(result)
                
                self.logger.info(f"Tool {tool_name} serialized result keys: {serialized_result.keys()}")
                
                # Create ToolMessage with dict content
                tool_message = ToolMessage(
                    content=json.dumps(serialized_result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_message)
                
            except Exception as e:
                self.logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                error_result = {"error": str(e), "tool": tool_name}
                tool_message = ToolMessage(
                    content=json.dumps(error_result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_message)
        
        return {"messages": tool_messages}

    def should_continue(self, state: ReportInfo):
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "end"

    def assistant(self, state: ReportInfo):
        messages = state.get("messages", [])

        textual_description_of_tool = """
            store_csvs(year: str) -> Dict[str, Any]: Fetches and stores the 'srag' dataset into the database.
            
            get_data_dict() -> Dict[str, Any]: Retrieves the data dictionary for the 'srag' dataset.
            
            summarize_numerical_data(year: str, column: str, mean: bool, median: bool, std: bool, min: bool, max: bool) -> Dict[str, Any]:
            Summarizes the numerical data in the specified column of the DataFrame.

            ARGS:
                year: str: The year of the data to summarize. Can be a specific year or "all".
                column: str: The column to summarize.
                mean: Optional[bool]: Whether to include the mean in the summary. Default is True.
                median: Optional[bool]: Whether to include the median in the summary. Default is True.
                std: Optional[bool]: Whether to include the standard deviation in the summary. Default is True.
                min: Optional[bool]: Whether to include the minimum value in the summary. Default is True.
                max: Optional[bool]: Whether to include the maximum value in the summary. Default is True.

            RETURNS:
                Dict[str, Any]: A summary of the data in the specified column.
                
            Generates a statistical report about the following topics:
            - Number of deaths and death rate
            - Number of new cases
            - Number of cases in UTI
            - Number of hospitalized cases
            - Percentage of citizens that got vaccinated

            the user will ask the year and month to month analysis

            ARGS:
                year: Year that im looking into
                state: Optional[str]: The state to filter the data by. If None, no filtering is applied.
                starting_month: int: The starting month that the user asked for.
                ending_month: int: The ending month that the user asked for.
                granularity: str: The granularity of the report. Valid values are 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual).
            RETURNS:
                A summary of the data of total cases from that year
                
            generate_temporal_graphical_report(state: Optional[str], year: Optional[str], granularity: str) -> Dict[str, Any]:
            Generates a graphical report about influenza cases.

            ARGS:
                state: Optional[str]: Brazilian state code like 'PE', 'CE', 'SP'. Defaults to all states.
                year: Optional[str]: Year to analyze. Defaults to '2025'.
                granularity: str: The granularity of the report. Valid values are 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual).
            RETURNS:
                Dict with figure_id, description, and data points.
        """

        if len(messages) == 1 or not isinstance(messages[0], SystemMessage):
            sys_msg = SystemMessage(content=textual_description_of_tool)
            messages = [sys_msg] + messages

        response = self.llm_tool_caller.invoke(messages)

        return {"messages": messages + [response]}

    def process_tool_results(self, state: ReportInfo):
        """Process tool results and extract figures, reports, etc."""
        messages = state["messages"]
        figures = state.get("figures", [])
        report = state.get("report", [])
        summary = state.get("summary", [])
        struct = state.get("struct", {})

        for msg in reversed(messages[-10:]):  # Look at more messages
            if hasattr(msg, 'type') and msg.type == 'tool':
                result = msg.content
                
                # Parse the JSON string back to dict
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                        self.logger.info(f"Parsed tool result with keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse tool result as JSON: {str(result)[:100]}... Error: {e}")
                        continue
                
                if isinstance(result, dict):
                    # Handle figure results - check for figure_id instead of figure object
                    if 'figure_id' in result:
                        # Check if this figure is already in the list
                        existing_ids = [f.get('figure_id') for f in figures if isinstance(f, dict)]
                        if result['figure_id'] not in existing_ids:
                            # Store the complete result (without the large figure object)
                            figure_data = {
                                'figure_id': result['figure_id'],
                                'description': result.get('description', ''),
                                'data_points': result.get('data_points', []),
                                'total_points': result.get('total_points', 0),
                                'state': result.get('state', 'all'),
                                'year': result.get('year', ''),
                                'granularity': result.get('granularity', 'D'),
                                'figure_html': result.get('figure_html', '')
                            }
                            figures.append(figure_data)
                            self.logger.info(f"Added figure: {result['figure_id']} with {result.get('total_points', 0)} data points")
                    
                    # Handle report results
                    if 'report' in result:
                        report.append(result['report'])
                        self.logger.info(f"Added report")
                    
                    # Handle summary statistics
                    if any(k in result for k in ['mean', 'median', 'std', 'min', 'max']):
                        summary.append(result)
                        self.logger.info(f"Added summary stats: {list(result.keys())}")
                    
                    # Handle struct/data dictionary
                    if any(k.startswith('_') for k in result.keys()):
                        struct = result
                        self.logger.info(f"Added struct with {len(result)} keys")
        
        return {
            "figures": figures,
            "report": report,
            "summary": summary,
            "struct": struct
        }

    def run(self, user_message: str, initial_state: Optional[Dict[str, Any]] = None):
        """Convenience method to run the agent with a user message"""
        if initial_state is None:
            initial_state = {
                "messages": [],
                "report": [],
                "insights": [],
                "struct": {},
                "summary": [],
                "stat_report": [],
                "figures": []
            }
        
        initial_state["messages"].append(HumanMessage(content=user_message))
        
        result = self.react_graph.invoke(initial_state)
        
        processed = self.process_tool_results(result)
        result.update(processed)
        
        return result


if __name__ == "__main__":
    agent = StatisticalAgent()

    result = agent.run("Store the CSV data for year 2024")
    
    print("=== FINAL RESULT ===")
    print(f"Messages: {len(result['messages'])}")
    print(f"Figures: {len(result.get('figures', []))}")
    print(f"Reports: {len(result.get('report', []))}")
    print(f"Summary: {len(result.get('summary', []))}")
    print(f"Struct keys: {len(result.get('struct', {}))}")
    
    # Print last few messages for debugging
    print("\n=== LAST MESSAGES ===")
    for i, msg in enumerate(result['messages'][-3:]):
        print(f"\nMessage {i}:")
        print(f"Type: {type(msg).__name__}")
        if hasattr(msg, 'content'):
            content_preview = str(msg.content)[:200]
            print(f"Content preview: {content_preview}...")