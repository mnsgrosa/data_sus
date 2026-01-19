from typing import List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.utils.logger import MainLogger

from .tools_utils.db.data_storage import SragDb
from .tools_utils.tools_helper import fetch_data

logger = MainLogger(__name__)


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
        logger.info(f"Year data: {year_data}")
        if year_data.empty:
            fetch_data([year])
            year_data = db.get_data(year)
            logger.info(f"Fetched from function:{year_data}")
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
    logger.info(f"Data:{returnable_data}")
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

    ARGS:
        year: Year that im looking into
        state: Optional[str]: The state to filter the data by. If None, no filtering is applied.
        starting_month: str: The starting month that the user asked for.
        ending_month: str: The ending month that the user asked for.
        granularity: Optional[str]: Time granularity (not currently used in logic)

    RETURNS:
        A summary of the data of total cases from that year
    """
    # Validate inputs
    if year not in [2021, 2022, 2023, 2024, 2025]:
        return {"error": f"Year {year} is not available. Try 2021-2025."}

    if granularity not in ["D", "ME", "SE", "M"]:
        return {"error": f"Invalid granularity '{granularity}'."}

    # Validate month range
    if not (1 <= starting_month <= 12 and 1 <= ending_month <= 12):
        return {"error": "Months must be between 1 and 12."}

    if starting_month > ending_month:
        return {"error": "starting_month cannot be greater than ending_month."}

    db = get_db()
    data = db.get_data(year)

    if data is None or data.empty:
        logger.info(f"No data found for {year}. Triggering fetch...")
        fetch_data([year])
        fetch_result = db.get_data(year)

        if fetch_result is None:
            return {
                "error": f"Could not fetch data for year {year} from external source."
            }

        # Retry after fetch
        data = db.get_data(year)
        if data is None or data.empty:
            return {"error": f"Fetched data but database still empty for year {year}."}

    logger.info(f"Found data:{data}")

    # Filter by state if specified
    if state and state.lower() != "all":
        state_upper = state.upper()
        data = data[data["SG_UF_NOT"] == state_upper]
        if data.empty:
            return {"error": f"No data found for state {state} in {year}."}

    try:
        data["DT_NOTIFIC"] = pd.to_datetime(data["DT_NOTIFIC"], errors="coerce")

        data = data.dropna(subset=["DT_NOTIFIC"])

        mask = (
            (data["DT_NOTIFIC"].dt.year == year)
            & (data["DT_NOTIFIC"].dt.month >= starting_month)
            & (data["DT_NOTIFIC"].dt.month <= ending_month)
        )

        filtered_data = data[mask]

        if filtered_data.empty:
            logger.error("Filetered data is empty")
            return {
                "error": f"No data found for months {starting_month}-{ending_month} in {year}."
            }

        logger.info(f"Filtered data shape: {filtered_data.shape}")

        total_count = len(filtered_data)
        death_count = int((filtered_data["EVOLUCAO"] == 2).sum())
        death_rate = (death_count / total_count * 100) if total_count > 0 else 0.0

        cases_hospitalized = int((filtered_data["HOSPITAL"] == 1).sum())

        uti_count = int((filtered_data["UTI"] == 1).sum())
        perc_uti = (uti_count / total_count * 100) if total_count > 0 else 0.0

        vaccinated_count = int((filtered_data["VACINA_COV"] == 1).sum())
        perc_vaccinated = (
            (vaccinated_count / total_count * 100) if total_count > 0 else 0.0
        )

        report = {
            "year": year,
            "state": state or "all",
            "month_range": f"{starting_month}-{ending_month}",
            "total_cases": total_count,
            "death_count": death_count,
            "death_rate": round(death_rate, 2),
            "cases_hospitalized": cases_hospitalized,
            "uti_count": uti_count,
            "perc_uti": round(perc_uti, 2),
            "vaccinated_count": vaccinated_count,
            "perc_vaccinated": round(perc_vaccinated, 2),
        }

        logger.info(f"Report generated successfully: {report}")
        return report

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return {"error": f"Failed to generate report: {str(e)}"}


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
        fetch_data([year])
        data.get_data(year)

    if data is None or data.empty:
        return None

    try:
        logger.info("Grouping the data")
        data["DT_NOTIFIC"] = pd.to_datetime(data["DT_NOTIFIC"])

        if state and state.lower() != "all":
            data = data[data["SG_UF_NOT"] == state.upper()]
            if data.empty:
                return {"error": f"No data to plot for state {state}."}

        grouped = (
            data.set_index("DT_NOTIFIC")
            .resample(granularity)
            .size()
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
            "figure_id": f"cases_{state or 'all'}_{year}",
        }
    except Exception as e:
        logger.error(f"Error while creating the graph: {e}")
        return {"error": f"Failed to generate graph: {str(e)}"}
