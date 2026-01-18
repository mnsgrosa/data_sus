from typing import Annotated, Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from src.utils.logger import MainLogger

from .tools_utils.db.data_storage import SragDb
from .tools_utils.tools_helper import fetch_data
from .tools_utils.tools_schema import (
    GraphReportRequest,
    GraphReportResponse,
    StatReportRequest,
    StatReportResponse,
    SummarizerRequest,
    SummarizerResponse,
)

logger = MainLogger(__name__)
BASE_URL = "https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}


def get_db():
    return SragDb()


@tool(response_format=SummarizerResponse | None)
def summarize_numerical_data(
    summary_request: SummarizerRequest,
) -> SummarizerResponse | None:
    """
    Summarizes the data in the specified column of the DataFrame.

    ARGS:
        SummarizerRequest[
        columns: List[str]: A list of columns to summarize.
        years: List[int]: List of desired years of data to summarizem, if user doesnt specify pass [2019, 2020, 2021, 2022, 2023, 2024, 2025].
        ]
    RETURNS:
        SummarizerResponse[year[str], Dict[column[str], int | float]] -> Dict with the informations about the categorical variables from the desired column and years
    """
    valid_years = [2021, 2022, 2023, 2024, 2025]
    valid_columns = [
        "EVOLUCAO",
        "UTI",
        "DT_NOTIFIC",
        "SG_UF_NOT",
        "VACINA_COV",
        "HOSPITAL",
        "DT_SIN_PRI",
        "DT_ENTUTI",
        "DT_SAIDUTI",
    ]
    if not set(summary_request.years).issubset(valid_years):
        logger.error("At least one of requested years isn't available")
        return None
    if not set(summary_request).issubset(valid_columns):
        logger.error("At least one of requested columns doesn't exist")
        return None

    db = get_db()

    logger.info(f"Starting data summary tool for: {summary_request}")
    returnable_data = {}

    columns = [column.upper() for column in summary_request.columns]

    for year in summary_request.years:
        year_data = db.get_data(year)
        if year_data is None:
            year_data = fetch_data(year)
        if year_data:
            year_data.fillna(-1, inplace=True)
            column_dict = {}
            for column in columns:
                year_dict = {}
                response = pd.Categorical(year_data[column], ordered=True)
                year_dict["median"] = np.median(response.codes)
                year_dict["freq"] = year_data[column].value_counts().to_numpy().tolist()
                column_dict[column] = year_dict
            returnable_data[year] = column_dict
    if not returnable_data:
        logger.info("No data was retrieved")
        return None
    logger.info(f"Data retrieved from:{returnable_data.keys()}")
    return SummarizerResponse(**returnable_data)


@tool(response_format=StatReportResponse | None)
def generate_statistical_report(
    request: StatReportRequest,
) -> StatReportResponse | None:
    """
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
    """
    logger.info(f"Starting report generation for: {request}")
    if request.year not in [2021, 2022, 2023, 2024, 2025]:
        return None

    if request.granularity not in ["D", "ME", "SE", "M"]:
        return None

    report = {}

    db = get_db()

    data = db.get_data(request.year)
    if data is None:
        data = fetch_data(request.year)
    if data is None or data.empty:
        return None

    if request.state and request.state.lower() != "all":
        data = data[data["SG_UF_NOT"] == request.state.upper()]

    try:
        year_int = int(request.year)

        data["DT_NOTIFIC"] = pd.to_datetime(data["DT_NOTIFIC"])
        mask = (
            (data["DT_NOTIFIC"].dt.year == year_int)
            & (data["DT_NOTIFIC"].dt.month >= request.starting_month)
            & (data["DT_NOTIFIC"].dt.month <= request.ending_month)
        )

        filtered_data = data[mask]

        logger.info(filtered_data)

        death_count = int(filtered_data[filtered_data["EVOLUCAO"] == 2].shape[0])
        total_count = int(filtered_data.shape[0])
        death_rate = (death_count / total_count) * 100 if total_count > 0 else 0

        report["death_count"] = int(death_count)
        report["death_rate"] = float(death_rate)
        report["total_cases"] = int(total_count)

        logger.info(f"{death_count}, {death_rate}, {total_count}")

        casos_internados = filtered_data[filtered_data["HOSPITAL"] == 1].shape[0]
        report["cases_hospitalized"] = int(casos_internados)

        logger.info(f"{casos_internados}")

        uti_count = filtered_data[filtered_data["UTI"] == 1].shape[0]
        perc_uti = (uti_count / total_count) * 100 if total_count > 0 else 0
        report["perc_uti"] = perc_uti

        logger.info(f"{perc_uti}")

        vaccinated_count = filtered_data[filtered_data["VACINA_COV"] == 1].shape[0]
        perc_vaccinated = (
            (vaccinated_count / total_count) * 100 if total_count > 0 else 0
        )
        report["perc_vaccinated"] = float(perc_vaccinated)

        logger.info(f"{perc_vaccinated}")

        logger.info(f"{perc_uti}")

        return StatReportResponse(**report)
    except Exception as e:
        logger.error(f"Error converting data: {e}")
        return None


@tool(response_format=GraphReportResponse | None)
def generate_temporal_graphical_report(
    request: GraphReportRequest,
) -> GraphReportResponse | None:
    """
    Generates a graphical report about the influenza cases in the selected state if not provided state defaults to None,

    ARGS:
        GraphReportRequest[
        state: str: Brazillian state options with the two first carachters like: 'PE', 'CE', 'SP'
        granularity: str: The granularity of the report. Valid values are 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual). Defaults to 'D'
        year: year that the user prompted, if not provided default to 2025
        ]
    RETURNS:
        A dictionary with the figure_id, description from the plot and the data points
    """
    if request.year not in [2021, 2022, 2023, 2024, 2025]:
        return None

    if request.granularity not in ["D", "ME", "SE", "M"]:
        return None
    db = get_db()

    data = db.get_data(request.year)
    if data is None:
        data = fetch_data(request.year)

    if data is None or data.empty:
        return None

    try:
        logger.info("Grouping the data")
        grouped = (
            data.fillna(0).groupby(by=["DT_NOTIFIC", "SG_UF_NOT"]).count().reset_index()
        )
        grouped["DT_NOTIFIC"] = pd.to_datetime(grouped["DT_NOTIFIC"])
        if request.state:
            grouped = (
                grouped[grouped["SG_UF_NOT"] == request.state]
                .set_index("DT_NOTIFIC")
                .resample(request.granularity)
                .count()
                .reset_index()
            )
        else:
            grouped = (
                grouped.set_index("DT_NOTIFIC")
                .resample(request.granularity)
                .count()
                .reset_index()
            )
    except Exception as e:
        logger.error(f"Error grouping data: {e}")
        return None

    try:
        logger.info("Creating the graph")

        x = grouped["DT_NOTIFIC"].dt.strftime("%Y-%m-%d").tolist()
        y = grouped["year"].tolist()

        return GraphReportResponse(
            **{
                "x": x,
                "y": y,
                "total_points": len(x),
                "state": request.state or "all",
                "granularity": request.granularity,
            }
        )
    except Exception as e:
        logger.error(f"Error while creating the graph: {e}")
        return None
