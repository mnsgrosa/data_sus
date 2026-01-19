from typing import List, Optional

from pydantic import BaseModel, Field


class SummarizerRequest(BaseModel):
    years: List[int] = Field(..., description="Selected years by user")
    columns: List[str] = Field(..., description="Selected columns by user")


class Summary(BaseModel):
    median: float = Field(..., description="Median from selected column")
    freq: float = Field(..., description="Frequency from selected column")


class Columns(BaseModel):
    column: Summary = Field(..., description="Summarization from selected column")


class SummarizerResponse(BaseModel):
    years: Columns = Field(..., description="Collumn summary from respective year")


class StatReportRequest(BaseModel):
    year: int = Field(..., description="Year prompted by user")
    starting_month: int = Field(
        ..., description="Starting month for report generation from prompted year"
    )
    ending_month: int = Field(
        ..., description="Ending month for report generation from prompted year"
    )
    state: Optional[str]
    granularity: str = Field(..., description="Signature for granulairity selection")


class StatReportResponse(BaseModel):
    death_count: int = Field(...)
    death_rate: float = Field(...)
    total_cases: int = Field(...)
    cases_hospitalized: int = Field(...)
    perc_uti: float = Field(...)
    perc_vaccinated: float = Field(...)


class GraphReportRequest(BaseModel):
    year: Optional[int] = None
    granularity: str = Field(...)
    state: Optional[str] = None


class GraphReportResponse(BaseModel):
    x: List[str] = Field(...)
    y: List[float] = Field(...)
    total_points: int = Field(...)
    state: str = Field(...)
    granularity: str = Field(...)
