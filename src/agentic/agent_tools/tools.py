import httpx
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Tuple, Annotated, Optional, Dict, Any
from langchain_core.tools import tool
from src.utils.logger import MainLogger
from src.agentic.agent_tools.tools_helper import extract_data_dictionary

logger = MainLogger(__name__)
BASE_URL = "https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

def get_db():
    return SragDb()

@tool
def store_csvs(years: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Get the 'srag' dataset from the datasus website about acute respiratory diseases including covid-19 
    and stores the data into de sqlite database.
    
    Whenever the user asks to get data, fetch data, store data it refers to this function
    ARGS:
        years: Optional[List[int]]: Year chosen by the user. the options are [2021, 2022, 2023, 2024, 2025]
        defaults to None, if None it will fetch all the years available
    RETURNS:
        pandas dataframe as dict.
    """
    logger.info("Starting the csv reader tool")
    with httpx.Client() as client:
        response = client.post('http://api:8000/store', json = {"years": years if years else []}, headers = headers, timeout=30000.0)
        response.raise_for_status()
        if response.status_code != 200:
            logger.error(f"Failed to fetch data: {response.text}")
            return {"status": "error", "message": "Failed to fetch data"}
        logger.info("Successfully fetched and stored the data")
    return response.json()
        
@tool
def get_data_dict() -> Dict[str, Any]:
    """
    Reads the data dictionary for the 'srag' dataset.

    RETURNS:
        A dictionary representing the data info.
    """
    logger.info("Starting to read the data dictionary")
    url = "https://opendatasus.saude.gov.br/dataset/39a4995f-4a6e-440f-8c8f-b00c81fae0d0/resource/3135ac9c-2019-4989-a893-2ed50ebd8e68/download/dicionario-de-dados-2019-a-2025.pdf"
    logger.info("Extracting the structure from pdf")

    structs = extract_data_dictionary(url)
    
    logger.info("Successfully read the data dictionary")
    return structs

@tool
def summarize_numerical_data(columns: List[str], years: List[int]) -> Dict[str, Dict[str, Any]]:
    """
    Summarizes the data in the specified column of the DataFrame.

    ARGS:
        columns: List[str]: A list of columns to summarize.
        years: List[int]: List of desired years of data to summarizem, if user doesnt specify pass [2019, 2020, 2021, 2022, 2023, 2024, 2025].

    RETURNS:
        Dict[str, Dict[str, Any]] -> Dict with the informations about the categorical variables from the desired column and years
    """
    logger.info(f"Starting to summarize data from: {columns}")
    with httpx.Client() as client:
        response = client.post('http://api:8000/summary', headers = headers, data = {"columns": columns, "years": years}, timeout=30000.0)
        response.raise_for_status()
        if response.status_code != 200:
            logger.error(f"Failed to summarize data: {response.text}")
            return {"status": "error", "message": "Failed to summarize data"}
        logger.info("Successfully summarized the data")
    ans = response.json()
    return ans.get("summaries", {})

@tool
def generate_statistical_report(
    year: int,
    starting_month: int,
    ending_month: int,
    state: Optional[str] = 'all',
    granularity: str = 'ME'
) -> Dict[str, Any]:
    """
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
    """
    logger.info("Starting statistical report generation")
    with httpx.Client() as client:
        post_data = {
            "year": year,
            "starting_month": starting_month,
            "ending_month": ending_month,
            "state": state,
            "granularity": granularity
        }
        response = client.get('http://api:8000/report', data = post_data, headers = headers, timeout=30000.0)
        response.raise_for_status()
        if response.status_code != 200:
            logger.error(f"Failed to generate report: {response.text}")
            return {"status": "error", "message": "Failed to generate report"}
        logger.info("Successfully generated the report")
        
        
@tool
def generate_temporal_graphical_report(year: Optional[int],  granularity: str = 'ME', state: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a graphical report about the influenza cases in the selected state if not provided state defaults to None,

    ARGS:
        state: str: Brazillian state options with the two first carachters like: 'PE', 'CE', 'SP'
        granularity: str: The granularity of the report. Valid values are 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual). Defaults to 'D'
        year: year that the user prompted, if not provided default to 2025
    RETURNS:
        A dictionary with the figure_id, description from the plot and the data points
    """
    logger.info("Starting graphical report generation")
    with httpx.Client() as client:
        post_data = {
            "year": year if year else 2025,
            "granularity": granularity,
            "state": state if state else None
        }
        response = client.post('http://api:8000/graphical_report', json = post_data, headers = headers, timeout=30000.0)
        response.raise_for_status()
        if response.status_code != 200:
            logger.error(f"Failed to generate graphical report: {response.text}")
            return {"status": "error", "message": "Failed to generate graphical report"}
        logger.info("Successfully generated the graphical report")
    return response.json()