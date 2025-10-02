import httpx
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Tuple, Annotated, Optional
from langchain_core.tools import tool
from src.utils.logger import MainLogger
from src.agent.tools.tools_helper import extract_data_dictionary
import matplotlib.pyplot as plt

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
def summarize_numerical_data(csv:pd.DataFrame, column: str, mean: Optional[bool] = True, median: Optional[bool] = True, std: Optional[bool] = True, min: Optional[bool] = True, max: Optional[bool] = True):
    """
    Summarizes the numerical data in the specified column of the DataFrame.

    ARGS:
        csv: pd.DataFrame: DataFrame containing the data to be summarized.
        column: str: The column to summarize.

    RETURNS:
        A summary of the data in the specified column.
    """
    logger.info(f"Starting to summarize data for column: {column}")
    if column not in csv.columns:
        logger.error(f"Column {column} not found in DataFrame")
        raise ValueError(f"Column {column} not found in DataFrame")

    returnable_data = {}

    if mean:
        mean_value = csv[column].mean()
        returnable_data['mean'] = mean_value
        logger.info(f"Mean of {column}: {mean_value}")
    if median:
        median_value = csv[column].median()
        returnable_data['median'] = median_value
        logger.info(f"Median of {column}: {median_value}")
    if std:
        std_value = csv[column].std()
        returnable_data['std'] = std_value
        logger.info(f"Standard Deviation of {column}: {std_value}")
    if min:
        min_value = csv[column].min()
        returnable_data['min'] = min_value
        logger.info(f"Minimum of {column}: {min_value}")
    if max:
        max_value = csv[column].max()
        returnable_data['max'] = max_value
        logger.info(f"Maximum of {column}: {max_value}")

    return returnable_data

@tool
def generate_statistical_report(csv:pd.DataFrame, state: Optional[str], start_analisys_period: str, end_analisys_period: str, granularity: str = 'D'):
    """
    Generates a statistical report about the following topics:
    - Number of deaths and death rate
    - Number of new cases
    - Number of cases in UTI
    - Number of hospitalized cases
    - Percentage of citizens that got vaccinated

    ARGS:
        csv: pd.DataFrame: DataFrame containing the data to be summarized.
        state: Optional[str]: The state to filter the data by. If None, no filtering is applied.
        start_analisys_period: str: The start date for the analysis period in 'YYYY-MM-DD' format.
        end_analisys_period: str: The end date for the analysis period in 'YYYY-MM-DD' format.
        granularity: str: The granularity of the report. Valid values are 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual).
    RETURNS:
        A summary of the data in the specified column.
    """
    if csv is None:
        logger.error("DataFrame is None")
        raise ValueError("DataFrame cannot be None")

    if granularity not in ['D', 'W', 'ME', 'Q', 'A']:
        logger.error(f"Invalid granularity provided: {granularity}")
        raise ValueError("Granularity must be one of the following: 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual)")

    data = csv.copy()
    report = {}

    if state:
        data = data[data['SG_UF_NOT'] == state]

    data['DT_NOTIFIC'] = pd.to_datetime(data['DT_NOTIFIC'])
    mask = (data['DT_NOTIFIC'] >= start_analisys_period) & (data['DT_NOTIFIC'] <= end_analisys_period)
    filtered_data = data.loc[mask]

    death_count = filtered_data[filtered_data['EVOLUCAO'] == 2].set_index('DT_NOTIFIC').resample(granularity).count().shape[0]
    total_count = filtered_data.shape[0]
    death_rate = (death_count / total_count) * 100 if total_count > 0 else 0

    report['death_count'] = death_count
    report['death_rate'] = death_rate

    novos_casos = filtered_data[filtered_data['SEM_NOT'] == 1].set_index('DT_NOTIFIC').resample(granularity).count()
    report['new_cases'] = novos_casos['SEM_NOT'].to_dict()

    casos_uti = filtered_data[filtered_data['UTI'] == 1].set_index('DT_NOTIFIC').resample(granularity).count()
    report['cases_uti'] = casos_uti['UTI'].to_dict()

    casos_internados = filtered_data[filtered_data['HOSPITAL'] == 1].set_index('DT_NOTIFIC').resample(granularity).count()
    report['cases_hospitalized'] = casos_internados['HOSPITAL'].to

    perc_uti = (casos_uti.shape[0] / total_count) * 100 if total_count > 0 else 0
    report['perc_uti'] = perc_uti

    vaccinated = filtered_data[filtered_data['VACINA_COV'] == 1].set_index('DT_NOTIFIC').resample(granularity).count()
    vaccinated = (vaccinated.shape[0] / total_count) * 100 if total_count > 0 else 0
    report['perc_vaccinated'] = vaccinated

    return report

@tool
def generate_temporal_graphical_report(csv:pd.DataFrame, column: str, state: Optional[str], granularity: str):
    """
    Generates a graphical report for the specified column of the DataFrame.

    ARGS:
        csv: pd.DataFrame: DataFrame containing the data to be plotted.
        column: str: The column to plot.
        chart_type: str: The type of chart to generate. Valid values are 'bar', 'line', 'scatter', 'histogram'.
    RETURNS:
        A plot of the data.
    """
    if csv is None:
        logger.error("DataFrame is None")
        raise ValueError("DataFrame cannot be None")

    if granularity not in ['D', 'W', 'ME', 'Q', 'A']:
        logger.error(f"Invalid granularity provided: {granularity}")
        raise ValueError("Granularity must be one of the following: 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual)")

    if column not in csv.columns:
        logger.error(f"Column {column} not found in DataFrame")
        raise ValueError(f"Column {column} not found in DataFrame")

    data = csv.copy()

    grouped = data.groupby(by = ['DT_NOTIFIC', 'SG_UF_NOT']).count().reset_index()
    if state:
        grouped = grouped[grouped['SG_UF_NOT'] == state].set_index('DT_NOTIFIC').resample(granularity).count()
    else:
        grouped = grouped.set_index('DT_NOTIFIC').resample(granularity).count()

    fig = px.line(grouped, x = 'DT_NOTIFIC', y = column, title = f'Temporal Graphical Report of {column}')
    return fig