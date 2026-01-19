import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from src.agentic.agent_tools.tools import (
    generate_graphical_report,
    generate_statistical_report,
    summarize_numerical_data,
)
from src.utils.logger import MainLogger

from .agent_schema.main_schema import JsonEncoder, ReportInfo

load_dotenv()


class StatisticalAgent(MainLogger):
    def __init__(self):
        super().__init__(__name__)
        self.initial_state = {
            "messages": [],
            "report": [],
            "struct": {},
            "summary": [],
            "stat_report": [],
            "figures": [],
            "data": {},
        }

        self.tools = [
            summarize_numerical_data,
            generate_statistical_report,
            generate_graphical_report,
        ]

        self.tool_map = {tool.name: tool for tool in self.tools}

        self.llm_tool_caller = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", temperature=0.2, max_tokens=None, max_retries=2
        )

        self.llm_tool_caller = self.llm_tool_caller.bind_tools(self.tools)
        self.graph = StateGraph(ReportInfo)
        self._init_graph()
        self.history = []
        self.data = {}

    def _init_graph(self):
        self.graph.add_node("assistant", self.assistant)
        self.graph.add_node("tools", self.call_tools)
        self.graph.add_node("process_results", self.process_tool_results)

        self.graph.set_entry_point("assistant")
        self.graph.add_edge(START, "assistant")
        self.graph.add_conditional_edges(
            "assistant", self.should_continue, {"tools": "tools", "end": END}
        )
        self.graph.add_edge("tools", "process_results")
        self.graph.add_edge("process_results", "assistant")
        self.react_graph = self.graph.compile()

    def _serialize_for_json(self, obj: Any) -> Any:
        try:
            encoder = JsonEncoder(data=obj)
            dump = encoder.model_dump()
            return dump.get("data", obj)
        except Exception as e:
            self.logger.error(f"Serialization failed: {e}")
            return str(obj)

    def call_tools(self, state: ReportInfo) -> Dict[str, List]:
        """Custom tool execution that preserves dict returns"""
        messages = state["messages"]
        last_message = messages[-1]

        tool_messages = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name == "generate_statistical_report"
                if "starting_month" in tool_args:
                    tool_args["starting_month"] = str(tool_args["starting_month"])
                    self.logger.info(
                        f"Coerced starting_month to string: {tool_args['starting_month']}"
                    )

                if "ending_month" in tool_args:
                    tool_args["ending_month"] = str(tool_args["ending_month"])
                    self.logger.info(
                        f"Coerced ending_month to string: {tool_args['ending_month']}"
                    )

            if tool_name == "summarize_numerical_data":
                if "columns" in tool_args:
                    tool_args["columns"] = [col for col in tool_args["columns"]]
                    self.info("Got the columns")
                if "years" in tool_args:
                    tool_args["years"] = list(tool_args["years"])
                    self.info(f"Got years")

            self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            try:
                tool = self.tool_map[tool_name]
                result = tool.invoke(tool_args)

                # Ensure result is a dict
                if not isinstance(result, dict):
                    self.logger.warning(
                        f"Tool {tool_name} returned non-dict: {type(result)}"
                    )
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except json.JSONDecodeError:
                            result = {"result": result}
                    else:
                        result = {"result": result}

                serialized_result = self._serialize_for_json(result)

                self.logger.info(
                    f"Tool {tool_name} serialized result keys: {serialized_result.keys()}"
                )

                tool_message = ToolMessage(
                    content=json.dumps(serialized_result),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                tool_messages.append(tool_message)

            except Exception as e:
                self.logger.error(
                    f"Error executing tool {tool_name}: {e}", exc_info=True
                )
                error_result = {"error": str(e), "tool": tool_name}
                tool_message = ToolMessage(
                    content=json.dumps(error_result),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                tool_messages.append(tool_message)

        return {"messages": messages + tool_messages}

    def should_continue(self, state: ReportInfo):
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    def assistant(self, state: ReportInfo):
        messages = state.get("messages", [])

        textual_description_of_tool = """
            The available columns for the data are: EVOLUCAO, UTI, DT_NOTIFIC, SG_UF_NOT, VACINA_COV, HOSPITAL, SEM_NOT
            store_csvs(year: str) -> Dict[str, Any]: Fetches and stores the 'srag' dataset into the database.

            get_data_dict() -> Dict[str, Any]: Retrieves the data dictionary for the 'srag' dataset. When the user wants to talk about anything about the columns
            consult this dict, it will guide you throughout the questions about the columns
            Summarizes the data in the specified column of the DataFrame.

            ARGS:
                columns: List[str]: A list of columns to summarize.
                years: List[int]: List of desired years of data to summarizem, if user doesnt specify pass [2019, 2020, 2021, 2022, 2023, 2024, 2025].

            RETURNS:
                Dict[str, Dict[str, Any]] -> Dict with the informations about the categorical variables from the desired column and years

            def generate_statistical_report(
                year: str,
                starting_month: int,
                ending_month: int,
                state: Optional[str] = 'all',
                granularity: str = 'D'
                ) -> Dict[str, Any]:
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
                starting_month: str: The starting month that the user asked for.
                ending_month: str: The ending month that the user asked for.
            RETURNS:
                A summary of the data of total cases from that year

            generate_temporal_graphical_report(state: Optional[str], year: Optional[str], granularity: str) -> Dict[str, Any]:
            Generates a graphical report about influenza cases.

            Summarizes the data in the specified column of the DataFrame.

    ARGS:
        columns: List[str]: A list of columns to summarize.
        years: List[int]: List of desired years of data to summarizem, if user doesnt specify pass [2019, 2020, 2021, 2022, 2023, 2024, 2025].

    RETURNS:
        Dict[str, Dict[str, Any]] -> Dict with the informations about the categorical variables from the desired column and years
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
        data = state.get("data", {})

        for msg in reversed(messages[-10:]):
            if hasattr(msg, "type") and msg.type == "tool":
                result = msg.content

                self.info(f"RESULT CONTENT: {result}")

                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError as e:
                        continue

                if isinstance(result, dict):
                    self.info("THIS IS A DICT!!!")
                    years = ["2021", "2022", "2023", "2024", "2025"]
                    if set(result.keys()).issubset(years) and result:
                        data.update(result)
                        self.info(f"Added data with keys: {result}")

                    if "total_cases" in result and result["total_cases"]:
                        self.info("Report detected")
                        data.update(result)
                        self.info(f"Report added succesfully: {result}")

                    if "x" in result and "y" in result and result["x"]:
                        self.info("Data points detected")
                        data.update(result)
                        self.info(f"Data points updated succesfully: {result}")

                    if any(
                        k in result for k in ["mean", "median", "std", "min", "max"]
                    ):
                        summary.append(result)

                    if any(k.startswith("_") for k in result.keys()):
                        struct = result

        return {
            "figures": figures,
            "report": report,
            "summary": summary,
            "struct": struct,
            "data": data,
        }

    def run(self, user_message: str, initial_state: Optional[Dict[str, Any]] = None):
        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        if initial_state is None:
            initial_state = {
                "messages": [],
                "report": [],
                "struct": {},
                "summary": [],
                "stat_report": [],
                "figures": [],
                "data": {},
            }

        initial_state["messages"].append(HumanMessage(content=user_message))

        result = self.react_graph.invoke(initial_state)
        return result
