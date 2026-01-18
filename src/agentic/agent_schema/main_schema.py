from datetime import date, datetime
from typing import Any, Dict, List, TypedDict

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, model_serializer


class ReportInfo(TypedDict):
    messages_dict: List[HumanMessage | AIMessage]
    report: List[Dict[str, Any]]
    struct: Dict[str, Any]
    summary: List[Dict[str, Any]]
    stat_report: List[Dict[str, Any]]
    figures: List[Any]


class JsonEncoder(BaseModel):
    data: Any

    @model_serializer(mode="wrap")
    def _serialize(self, serializer, info):
        return self._process(self.data)

    def _process(self, obj: Any) -> Any:
        if hasattr(obj, "_data_class_name") or (
            hasattr(obj, "__module__") and "plotly" in str(obj.__module__)
        ):
            return {"_plotly_figure": True, "_type": type(obj).__name__}

        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (pd.Timestamp, datetime, date)):
            return obj.isoformat()

        if isinstance(obj, dict):
            return {
                str(k) if isinstance(k, tuple) else k: self._process(v)
                for k, v in obj.items()
            }

        if isinstance(obj, (list, tuple)):
            return [self._process(item) for item in obj]

        if hasattr(obj, "to_dict") and not hasattr(obj, "_data_class_name"):
            try:
                return self._process(obj.to_dict())
            except Exception:
                return str(obj)

        return obj


class AgentResponse(BaseModel):
    content: str = Field(..., description="LLM generated output about tool calling")
    data: Dict[str, int] = Field(
        ..., description="Datapoints outputed by tools if any tool was called"
    )
    tool_name: str = Field(..., description="Name of tool used")
