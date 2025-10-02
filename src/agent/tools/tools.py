import httpx
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Tuple, Annotated, Optional
from langchain_core.tools import tool
from src.utils.logger import MainLogger
from src.agent.tools.tools_helper import extract_tables_from_pdf
import matplotlib.pyplot as plt
import plotly.express as px

logger = MainLogger(__name__)
BASE_URL = "https://opendatasus.saude.gov.br/dataset"

@tool
def read_csv(year: int) -> pd.DataFrame:
    """
    Reads the 'srag' data about acute respiratory diseases including covid-19 and returns it

    ARGS:
        year: int: Year of the data to be downloaded. Valid values are 2019, 2020, 2021, 2022, 2023, 2024, 2025
    RETURNS:
        A pandas DataFrame containing the data from the CSV file.
    """
    logger.info("Starting the csv reader tool")

    if year not in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        logger.error(f"Invalid year provided: {year}")
        raise ValueError("Year must be one of the following: 2019, 2020, 2021, 2022, 2023, 2024, 2025")

    response = httpx.get(BASE_URL)
    logger.info(f"Fetched data from {BASE_URL} with status code {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', class_ = 'dropdown-item')

    s3_link = None

    for link in links:
        if 's3' in link.get('href', '') and str(year) in link.get('href', ''):
            s3_link = link.get('href')
            break

    if not s3_link:
        raise ValueError(f"No S3 link found for the year {year}")

    df = pd.read_csv(s3_link)
    logger.info(f"Successfully read CSV data from {s3_link}")
    return df

@tool
def store_data_dict():
    """
    Reads the data dictionary for the 'srag' dataset.

    RETURNS:
        A pandas DataFrame containing the data dictionary.
    """
    logger.info("Starting to read the data dictionary")
    url = "https://opendatasus.saude.gov.br/dataset/39a4995f-4a6e-440f-8c8f-b00c81fae0d0/resource/3135ac9c-2019-4989-a893-2ed50ebd8e68/download/dicionario-de-dados-2019-a-2025.pdf"
    logger.info("Extracting the structure from pdf")

    structs = extract_tables_from_pdf(url)
    
    logger.info("Successfully read the data dictionary")
    return structs

@tool
def summarize_data(csv:pd.DataFrame):

@tool
def plot_data(csv:pd.DataFrame):
    """
    Plots the data from the provided DataFrame.

    ARGS:
        csv: pd.DataFrame: DataFrame containing the data to be plotted.
    RETURNS:
        A plot of the data.
    """
    pass
