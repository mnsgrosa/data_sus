from typing import List, Optional

import numpy as np
import pandas as pd
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


@tool
def summarize_numerical_data(years: List[int], columns: List[str]) -> dict | None:
    """
    Summarizes the data in the specified column of the DataFrame.

    ARGS:
        years: List[int]: List of desired years of data to summarizem, if user doesnt specify pass [2019, 2020, 2021, 2022, 2023, 2024, 2025].
        columns: List[str]: A list of columns to summarize.
    RETURNS:
        Dict[str, float] -> Dict with the informations about the categorical variables from the desired column and years
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
    if not set(years).issubset(valid_years):
        logger.error("At least one of requested years isn't available")
        return {
            "error": "One or more requested years are unavailable. Valid years: 2021-2025."
        }

    if not set(columns).issubset(valid_columns):
        logger.error("At least one of requested columns doesn't exist")
        return {"error": f"Invalid columns requested. Valid options: {valid_columns}"}
    db = get_db()
    returnable_data = {}

    columns = [column.upper() for column in columns]

    for year in years:
        year_data = db.get_data(year)
        if year_data is None:
            year_data = fetch_data([year])
        if year_data is not None and not year_data.empty:
            year_data.fillna(-1, inplace=True)
            column_dict = {}
            for column in columns:
                if column in year_data.columns:
                    year_dict = {}
                    response = pd.Categorical(year_data[column], ordered=True)
                    year_dict["median"] = int(np.median(response.codes))
                    year_dict["freq"] = year_data[column].value_counts().to_dict()
                    column_dict[column] = year_dict
            returnable_data[year] = column_dict
    if not returnable_data:
        logger.info("No data was retrieved")
        return None
    logger.info(f"Data retrieved from:{returnable_data.keys()}")
    return returnable_data


@tool
def generate_statistical_report(
    year: int,
    starting_month: int,
    ending_month: int,
    state: Optional[str] = "all",
    granularity: Optional[str] = "ME",
) -> dict | None:
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
    if year not in [2021, 2022, 2023, 2024, 2025]:
        return {"error": f"Year {year} is not available. Try 2021-2025."}

    if granularity not in ["D", "ME", "SE", "M"]:
        return {"error": f"Invalid granularity '{granularity}'."}

    report = {}

    db = SragDb()
    df = db.get_data(year)

    if df is None or df.empty:
        print(f"No data found for {year}. Triggering fetch...")  # or use logger

        # Call the scraper we just fixed
        # Note: fetch_data expects a list of ints
        fetch_result = fetch_data([year])

        if fetch_result is None:
            return {
                "error": f"Could not fetch data for year {year} from external source."
            }

        # 3. Retry getting data after fetch
        df = db.get_data(year)
        if df is None or df.empty:
            return {
                "error": "Fetched data but database is still returning empty. Check insertion logic."
            }

    data = db.get_data(year)
    if data is None or data.empty:
        return {"error": f"No data available in the database for year {year}."}

    if state and state.lower() != "all":
        data = data[data["SG_UF_NOT"] == state.upper()]
        if data.empty:
            return {"error": f"No data found for state {state} in {year}."}

    if state and state.lower() != "all":
        data = data[data["SG_UF_NOT"] == state.upper()]

    df_state = df[df["SG_UF_NOT"] == state]

    if df_state.empty:
        return {
            "error": f"Data available for {year}, but no records found for state {state}."
        }

    try:
        year_int = int(year)

        data["DT_NOTIFIC"] = pd.to_datetime(data["DT_NOTIFIC"])
        mask = (
            (data["DT_NOTIFIC"].dt.year == year_int)
            & (data["DT_NOTIFIC"].dt.month >= starting_month)
            & (data["DT_NOTIFIC"].dt.month <= ending_month)
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

        return report
    except Exception as e:
        logger.error(f"Error converting data: {e}")
        return None


@tool
def generate_graphical_report(
    year: Optional[int], granularity: Optional[str] = "ME", state: Optional[str] = None
) -> dict | None:
    """
    Generates a graphical report about the covid cases at selected state if not provided state defaults to None,

    ARGS:-
        state: str: Brazillian state options with the two first carachters like: 'PE', 'CE', 'SP'
        granularity: str: The granularity of the report. Valid values are 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual). Defaults to 'D'
        year: year that the user prompted, if not provided default to 2025

    RETURNS:
        A dictionary with the figure_id, description from the plot and the data points
    """
    if year not in [2021, 2022, 2023, 2024, 2025]:
        return None

    if granularity not in ["D", "ME", "SE", "M"]:
        return None
    db = get_db()

    data = db.get_data(year)
    if data is None:
        data = fetch_data([year])

    if data is None or data.empty:
        return None

    try:
        logger.info("Grouping the data")
        # Ensure date column is datetime
        data["DT_NOTIFIC"] = pd.to_datetime(data["DT_NOTIFIC"])

        # Filter first if state is provided (optimization)
        if state and state.lower() != "all":
            data = data[data["SG_UF_NOT"] == state.upper()]
            if data.empty:
                return {"error": f"No data to plot for state {state}."}

        # Use a simpler groupby/resample logic
        # Note: 'year' column usually doesn't exist in raw data unless added,
        # normally we count rows (size)
        grouped = (
            data.set_index("DT_NOTIFIC")
            .resample(granularity)
            .size()  # Count occurrences
            .reset_index(name="count")
        )

        logger.info("Creating the graph payload")

        x = grouped["DT_NOTIFIC"].dt.strftime("%Y-%m-%d").tolist()
        y = grouped["count"].tolist()

        return {
            "x": x,
            "y": y,
            "total_points": len(x),
            "state": state or "all",
            "granularity": granularity,
            "figure_id": f"cases_{state or 'all'}_{year}",  # Added ID for agent processing
        }
    except Exception as e:
        logger.error(f"Error while creating the graph: {e}")
        return {"error": f"Failed to generate graph: {str(e)}"}
