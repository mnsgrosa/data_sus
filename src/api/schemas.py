from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PromptResponse(BaseModel):
    content: str = Field(..., description="LLM response about what it collected")
    data: Optional[Dict[str, int]] = Field(..., description="Datapoint collection")
