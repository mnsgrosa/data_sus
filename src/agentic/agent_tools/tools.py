import httpx
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Tuple, Annotated, Optional, Dict, Any
from langchain_core.tools import tool
from src.utils.logger import MainLogger
from src.agentic.agent_tools.tools_helper import extract_data_dictionary
from src.db.data_storage import SragDb
from pydantic import BaseModel, Field
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import re
import datetime

logger = MainLogger(__name__)
BASE_URL = "https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024"

def get_db():
    return SragDb()

@tool
def store_csvs(year: str) -> Dict[str, Any]:
    """
    Get the 'srag' dataset from the datasus website about acute respiratory diseases including covid-19 
    and stores the data into de sqlite database.
    
    Whenever the user asks to get data, fetch data, store data it refers to this function
    ARGS:
        year: str: Year chosen by the user. the options are [2021', '2022', '2023', '2024', '2025']
    RETURNS:
        pandas dataframe as dict.
    """
    
    logger.info("Starting the csv reader tool")
    with httpx.Client() as client:
        response = client.get(BASE_URL, timeout = 300000)
    logger.info(f"Fetched data from {BASE_URL} with status code {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    dropdown = soup.find_all('a', class_ = 'dropdown-item')
    items = [item['href'] for item in dropdown]

    s3_link = None
    db = SragDb()

    if year == 'all':
        try:
            for link in items:
                if 's3' in link:
                    s3_link = link
                    option = re.findall(r'\d{4}', link)[0]
                    df = pd.read_csv(s3_link, sep = ';', low_memory = False)
                    df = df[['SG_UF_NOT', 'DT_NOTIFIC', 'UTI', 'VACINA_COV', 'HOSPITAL', 'EVOLUCAO']]
                    df['year'] = [int(option)] * len(df)
                    logger.info(f"Data {df}")
                    insertion_result = db.insert(df.to_dict(orient = 'records'))
                    logger.info(f"Data for year {option} fetched and processed")
                    if not insertion_result:
                        logger.error(f"Failed to insert data into the database for year {option}")
                        raise Exception(f"Failed to insert data into the database for year {option}")

            logger.info(f"Successfully read CSV data from {s3_link}")
            return {"status": "success", "message": "Data inserted successfully"}
        except Exception as e:
            logger.error(f"Error while fetching or processing data: {e}")
            raise
    
    try:
        for link in items:
            if 's3' in link:
                s3_link = link
                option = re.findall(r'\d{4}', link)[0]
                if year == option:
                    df = pd.read_csv(s3_link, sep = ';', low_memory = False)
                    df = df[['SG_UF_NOT', 'DT_NOTIFIC', 'UTI', 'VACINA_COV', 'SEM_NOT', 'HOSPITAL', 'EVOLUCAO']] 
                    df['year'] = [int(year)] * len(df)
                    logger.info(f"Data {df}")
                    insertion_result = db.insert(df.to_dict(orient = 'records'))
                    logger.info(f"Data for year {year} fetched and processed")
                    if not insertion_result:
                        logger.error(f"Failed to insert data into the database for year {year}")
                        raise Exception(f"Failed to insert data into the database for year {year}")

                    logger.info(f"Successfully read CSV data from {s3_link} and {year}")
                    return {"status": "success", "message": "Data inserted successfully"}
    except Exception as e:
        logger.error(f"Error while fetching or processing data: {e} and {year}")
        raise
        
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
    returnable_data = {}


    db = SragDb()
    columns = [column.upper() for column in columns]

    try:
        for year in years:
            logger.info(f'Year: {year}')
            if year not in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
                logger.error('Year is not part of the dataset')
            data = db.get_data(year)
            data.fillna(-1, inplace = True)
            columns_dict = {}
            for column in columns:
                year_dict = {} 
                response = pd.Categorical(data[column], ordered = True)
                year_dict['median'] = np.median(response.codes)
                year_dict['freq'] = data[column].value_counts()
                columns_dict[column] = year_dict
            returnable_data[year] = columns_dict 
        return returnable_data
    except Exception as e:
        logger.error(f'Error summarizing: {e}')
        raise e
        return returnable_dict

@tool
def generate_statistical_report(
    year: int,
    starting_month: int,
    ending_month: int,
    state: Optional[str] = 'all',
    granularity: str = 'D'
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
    if year not in [2021, 2022, 2023, 2024, 2025]:
        logger.error("Invalid year provided")
        raise ValueError("Invalid year provided")

    if granularity not in ['D', 'W', 'ME', 'Q', 'A']:
        logger.error(f"Invalid granularity provided: {granularity}")
        raise ValueError("Granularity must be one of: 'D', 'W', 'ME', 'Q', 'A'")

    report = {}
    db = get_db()
    data = db.get_data(int(year))

    if data is None:
        logger.error("No data found for the specified year")
        raise ValueError("No data found for the specified year")

    if state != 'all':
        data = data[data['SG_UF_NOT'] == state]
        
    try:
        year_int = int(year)

        data['DT_NOTIFIC'] = pd.to_datetime(data['DT_NOTIFIC'])
        mask = (data['DT_NOTIFIC'].dt.year == year_int) & \
            (data['DT_NOTIFIC'].dt.month >= starting_month) & \
            (data['DT_NOTIFIC'].dt.month <= ending_month)

        filtered_data = data[mask]

        logger.info(filtered_data)

        death_count = int(filtered_data[filtered_data['EVOLUCAO'] == 2].shape[0])
        total_count = int(filtered_data.shape[0])
        death_rate = (death_count / total_count) * 100 if total_count > 0 else 0

        report['death_count'] = int(death_count)
        report['death_rate'] = float(death_rate)
        report['total_cases'] = int(total_count)

        logger.info(f"{death_count}, {death_rate}, {total_count}")
        
        casos_internados = filtered_data[filtered_data['HOSPITAL'] == 1].shape[0]
        report['cases_hospitalized'] = int(casos_internados)

        logger.info(f"{casos_internados}")

        uti_count = filtered_data[filtered_data['UTI'] == 1].shape[0]
        perc_uti = (uti_count / total_count) * 100 if total_count > 0 else 0
        report['perc_uti'] = perc_uti

        logger.info(f"{perc_uti}")

        vaccinated_count = filtered_data[filtered_data['VACINA_COV'] == 1].shape[0]
        perc_vaccinated = (vaccinated_count / total_count) * 100 if total_count > 0 else 0
        report['perc_vaccinated'] = float(perc_vaccinated)

        logger.info(f"{perc_vaccinated}")

        uti_cases = filtered_data[filtered_data['UTI'] == 1].shape[0]
        report['perc_uti'] = (uti_cases / total_count) * 100 if uti_cases > 0 else 0 

        logger.info(f"{uti_cases}")

        return {'report': report}
    except Exception as e:
        logger.error(f'Error converting data: {e}')
        raise e
        
@tool
def generate_temporal_graphical_report(year: Optional[int],  granularity: str = 'D', state: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a graphical report about the influenza cases in the selected state if not provided state defaults to None,

    ARGS:
        state: str: Brazillian state options with the two first carachters like: 'PE', 'CE', 'SP'
        granularity: str: The granularity of the report. Valid values are 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual). Defaults to 'D'
        year: year that the user prompted, if not provided default to 2025
    RETURNS:
        A dictionary with the figure_id, description from the plot and the data points
    """
    logger.info('Starting temporal graphical report')
    if year not in range(2021, 2025):
        logger.error("Invalid year provided")
        raise ValueError("Invalid year provided")

    if granularity not in ['D', 'W', 'ME', 'Q', 'A']:
        logger.error(f"Invalid granularity provided: {granularity}")
        raise ValueError("Granularity must be one of the following: 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual)")

    db = get_db()

    data = db.get_data(int(year))
    if data is None:
        logger.error("No data found for the specified year")
        raise ValueError("No data found for the specified year")

    try:
        logger.info('Grouping the data')    
        grouped = data.fillna(0).groupby(by = ['DT_NOTIFIC', 'SG_UF_NOT']).count().reset_index()
        grouped['DT_NOTIFIC'] = pd.to_datetime(grouped['DT_NOTIFIC'])
        if state:
            grouped = grouped[grouped['SG_UF_NOT'] == state].set_index('DT_NOTIFIC').resample(granularity).count()
        else:
            grouped = grouped.set_index('DT_NOTIFIC').resample(granularity).count().reset_index()
    except Exception as e:
        logger.error(f'Error grouping data: {e}')
        raise e

    try:
        logger.info('Creating the graph')
        
        x = grouped['DT_NOTIFIC'].tolist()
        y = grouped['year'].tolist()

        return {
            "x": x,  
            "y": y,
            "total_points": len(x),
            "state": state or "all",
            "year": year,
            "granularity": granularity
        }
    except Exception as e:
        logger.error(f'Error while creating the graph: {e}')
        raise e