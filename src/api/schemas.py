from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class StoreRequestSchema(BaseModel):
    years: Optional[List[int]] = Field([], description="Year of the data to store")

class StoreResponseSchema(BaseModel):
    status: str = Field(..., description="Status of the response")
    message: str = Field(..., description="Message regarding the storage operation")

class FetchRequestSchema(BaseModel):
    years: List[int] = Field(..., description="Years to fetch data for")
    
class FetchResponseSchema(BaseModel):
    status: str = Field(..., description="Status of the response")
    data: Dict[str, Any] = Field(..., description="Data returned in the response")

class SummaryRequestSchema(BaseModel):
    years: List[int] = Field(..., description="Years of the summary")
    columns: List[str] = Field(..., description="Column names")

class InnerSummaryItemSchema(BaseModel):
    median: Optional[float] = Field(None, description="Median value of the column")
    freq: List[int] = Field(default_factory=list, description="Frequency distribution of the column values")

class SummaryResponseSchema(BaseModel):
    status: str = Field(..., description="Status of the response")
    summaries: Dict[int, Dict[str, InnerSummaryItemSchema]] = Field(
        ..., 
        description="Summaries by year and column"
    )

class ReportRequestSchema(BaseModel):
    year: int = Field(..., description="Year for the report")
    starting_month: int = Field(..., description="Starting month")
    ending_month: int = Field(..., description="Ending month")
    state: Optional[str] = Field('all', description="State filter")
    granularity: str = Field('ME', description="Granularity of the report")

class ReportResponseSchema(BaseModel):
    status: str = Field(..., description="Status of the response")
    death_count: int = Field(..., description="Total death count")
    death_rate: float = Field(..., description="Death rate")
    total_cases: int = Field(..., description="Total cases")
    cases_hospitalized: int = Field(..., description="Total hospitalized cases")
    perc_uti: float = Field(..., description="Percentage of ICU cases")
    perc_vaccinated: float = Field(..., description="Percentage of vaccinated cases")

class GraphicalReportRequestSchema(BaseModel):
    year: int = Field(..., description="Year for the graphical report")
    granularity: str = Field('ME', description="Granularity of the report")
    state: Optional[str] = Field(None, description="State filter")

class GraphicalReportResponseSchema(BaseModel):
    x: List[str] = Field(..., description="X-axis values")
    y: List[int] = Field(..., description="Y-axis values")
    total_points: int = Field(..., description="Total data points")
    state: Optional[str] = Field("all", description="State filter")
    granularity: str = Field('ME', description="Granularity of the report")
    message: str = Field(..., description="Message regarding the report generation")