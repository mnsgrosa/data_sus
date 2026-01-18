import json
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agentic.agent_tools.tools import (
    generate_statistical_report,
    generate_temporal_graphical_report,
    summarize_numerical_data,
)
from src.utils.logger import MainLogger

from .agent_schema.main_schema import AgentResponse, JsonEncoder, ReportInfo

load_dotenv()


class StatisticalAgent(MainLogger):
    def __init__(self):
        super().__init__(__name__)
        self.initial_state = {
            "messages_dicts": [],
            "report_dict": [],
            "insights_dict": [],
            "struct": {},
            "summary_dict": [],
            "stat_report_dict": [],
            "figures_dict": [],
        }
        self.tools = [
            summarize_numerical_data,
            generate_statistical_report,
            generate_temporal_graphical_report,
        ]

        self.checkpointer = MemorySaver()

        self.tool_map = {tool.name: tool for tool in self.tools}

        self.llm_tool_caller = ChatGoogleGenerativeAI(
            model="gemini-3-pro", temperature=0.2, max_tokens=None, max_retries=2, output_format = AgentResponse
        )

        self.llm_tool_caller = self.llm_tool_caller.bind_tools(self.tools)
        self.graph = StateGraph(ReportInfo)
        self._init_graph()
        self.history = []
        self.data = {}

    def _init_graph(self):
        self.graph.add_node("assistant", self.assistant)
        self.graph.add_node("tools", self.call_tools)
        self.graph.set_entry_point("assistant")
        self.graph.add_edge(START, "assistant")
        self.graph.add_conditional_edges(
            "assistant", self.should_continue, {"tools": "tools", "end": END}
        )
        self.graph.add_edge("tools", "assistant")
        self.react_graph = self.graph.compile(checkpointer=self.checkpointer)

    def _serialize_for_json(self, obj: Any) -> Any:
        encoder = JsonEncoder(data=obj)
        return encoder.model_dump()["data"]

    def call_tools(self, state: ReportInfo) -> Dict[str, List]:
        messages = state["messages_dict"]
        last_message = messages[-1]

        if not hasattr(last_message.get("assistant"), "tool_calls"):
            self.logger.warning("No tool calls found in last message")
            return {"messages_dict": messages}

        tool_messages = []

        for tool_call in last_message["assistant"].tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            try:
                if tool_name not in self.tool_map:
                    raise ValueError(f"Unknown tool: {tool_name}")

                tool = self.tool_map[tool_name]

                result = tool.invoke(tool_args)

                if not isinstance(result, dict):
                    self.logger.warning(
                        f"Tool {tool_name} returned non-dict: {type(result)}"
                    )
                    result = {"result": str(result)}

                serialized_result = self._serialize_for_json(result)

                tool_message = ToolMessage(
                    content=json.dumps(serialized_result, ensure_ascii=False),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                tool_messages.append(tool_message) -> GraphReportResponse

                self.logger.info(f"Tool {tool_name} executed successfully")

            except Exception as e:
                self.logger.error(
                    f"Error executing tool {tool_name}: {e}", exc_info=True
                )

                error_result = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "tool": tool_name,
                }

                tool_message = ToolMessage(
                    content=json.dumps(error_result),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                tool_messages.append(tool_message)

        return {"messages_dict": messages}

    def should_continue(self, state: ReportInfo):
        messages = state["messages_dict"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    def assistant(self, state: ReportInfo):
        messages = state.get("messages", [])

        textual_description_of_tool = """
            The available columns for the data are: EVOLUCAO, UTI, DT_NOTIFIC, SG_UF_NOT, VACINA_COV, HOSPITAL, SEM_NOT
            RULES:
            - Use the provided tools to fetch, generate graphical reports and a statistical summary
            - Always interact with the output from the last tool call
            - You can not make up data, if you don't know the answer just say you dont know

            TOOLS:
            1) generate_statistical_report (
            request: SummarizerRequest(Dict with keys: years:List[int], columns[str])
            ) -> SummarizerResponse:
            DESCRIPTION:
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

            2)generate_statistical_report(
                request: StatReportRequest[
                    year: int,
                    starting_month: int,
                    ending_month: int,
                    state: str,
                    granularity: str
                ]
            )
            DESCRIPTION:
            Generates a statistical report about the following topics:
                    - Number of deaths and death rate
                    - Number of new cases
                    - Number of cases in UTI
                    - Number of hospitalized cases
                    - Percentage of citizens that got vaccinated

                    the user will ask the year and month to month analysis

                    ARGS:
                        request: StatReportRequest[
                        year: Year that im looking into
                        state: Optional[str]: The state to filter the data by. If None, no filtering is applied.
                        starting_month: str: The starting month that the user asked for.
                        ending_month: str: The ending month that the user asked for.
                        ]
                    RETURNS:
                        A summary of the data of total cases from that year

            generate_temporal_graphical_report(
            GraphReportRequest[
                year: Optional[int] = None
                granularity: str = Field(...)
                state: Optional[str] = None
            ]):
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
        messages = state["messages_dict"]
        figures = state.get("figures", [])
        report = state.get("report", [])
        summary = state.get("summary", [])
        struct = state.get("struct", {})

        for msg in reversed(messages[-10:]):
            if hasattr(msg, "type") and msg.type == "tool":
                result = msg.content

                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                        self.logger.info(
                            f"Parsed tool result with keys: {result.keys() if isinstance(result, dict) else 'not a dict'}"
                        )
                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            f"Failed to parse tool result as JSON: {str(result)[:100]}... Error: {e}"
                        )
                        continue

                if isinstance(result, dict):
                    if "figure_id" in result:
                        existing_ids = [
                            f.get("figure_id") for f in figures if isinstance(f, dict)
                        ]
                        if result["figure_id"] not in existing_ids:
                            figure_data = {
                                "figure_id": result["figure_id"],
                                "description": result.get("description", ""),
                                "data_points": result.get("data_points", []),
                                "total_points": result.get("total_points", 0),
                                "state": result.get("state", "all"),
                                "year": result.get("year", ""),
                                "granularity": result.get("granularity", "D"),
                                "figure_html": result.get("figure_html", ""),
                            }
                            figures.append(figure_data)
                            self.logger.info(
                                f"Added figure: {result['figure_id']} with {result.get('total_points', 0)} data points"
                            )

                    if "report" in result:
                        report.append(result["report"])
                        self.logger.info("Added report")

                    if any(
                        k in result for k in ["mean", "median", "std", "min", "max"]
                    ):
                        summary.append(result)
                        self.logger.info(f"Added summary stats: {list(result.keys())}")

                    if any(k.startswith("_") for k in result.keys()):
                        struct = result
                        self.logger.info(f"Added struct with {len(result)} keys")

        return {
            "figures": figures,
            "report": report,
            "summary": summary,
            "struct": struct,
        }

    def run(self, user_message: str) -> AgentResponse:
        prompt = HumanMessage(content=user_message)

        if self.initial_state["messages_dict"] is None:
            self.initial_state["messages_dict"].append(
                {"human": prompt, "assistant": None}
            )

        result = self.react_graph.invoke(self.initial_state)

        processed = self.process_tool_results(result)
        # Enforce a LLM output so it receives the same format as initial state schema
        # self.initial_state[]
        result.update(processed)

        return result
