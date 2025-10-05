import httpx
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Tuple, Annotated, Optional, Dict, Any
from langchain_core.tools import tool
from src.utils.logger import MainLogger
from src.agentic.agent_tools.tools_helper import extract_data_dictionary
from src.db.data_storage import SragDb
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
        year: str: Year chosen by the user. the options are ['all', '2021', '2022', '2023', '2024', '2025']
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
                    df = df[['SG_UF_NOT', 'DT_NOTIFIC', 'UTI', 'VACINA_COV', 'HOSPITAL']]
                    df['year'] = [int(option)] * len(df)
                    logger.info(f"Data {df}")
                    insertion_result = db.insert(df.to_dict(orient = 'records'))
                    logger.info(f"Data for year {option} fetched and processed")
                    if not insertion_result:
                        logger.error(f"Failed to insert data into the database: {e}")
                        raise Exception("Failed to insert data into the database")
                        return {"status": "error", "message": "Failed to insert data into the database"}

            logger.info(f"Successfully read CSV data from {s3_link}")
            return {"status": "success", "message": "Data inserted successfully"}
        except Exception as e:
            logger.error(f"Error while fetching or processing data: {e}")
            raise e
            return {"status": "error", "message": str(e)}
    
    try:
        for link in items:
            if 's3' in link:
                s3_link = link
                option = re.findall(r'\d{4}', link)[0]
                if year == option:
                    df = pd.read_csv(s3_link, sep = ';', low_memory = False)
                    df = df[['SG_UF_NOT', 'DT_NOTIFIC', 'UTI', 'VACINA_COV', 'HOSPITAL']]
                    df['year'] = [int(year)] * len(df)
                    logger.info(f"Data {df}")
                    insertion_result = db.insert(df.to_dict(orient = 'records'))
                    logger.info(f"Data for year {year} fetched and processed")
                    if not insertion_result:
                        logger.error("Failed to insert data into the database")
                        raise Exception("Failed to insert data into the database")
                        return {"status": "error", "message": "Failed to insert data into the database"}

                    logger.info(f"Succesfully read CSV data from {s3_link} and {year}")
                    return {"status": "success", "message": "Data inserted successfully"}
    except Exception as e:
        logger.error(f"Error while fetching or processing data: {e} and {year}")
        raise e
        return {"status": "error", "message": str(e)}
    
        
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
def summarize_numerical_data(year:str, column: str, mean: Optional[bool] = True, median: Optional[bool] = True, std: Optional[bool] = True, min: Optional[bool] = True, max: Optional[bool] = True) -> Dict[str, Any]:
    """
    Summarizes the numerical data in the specified column of the DataFrame.

    ARGS:
        year: str: The year of the data to summarize. Can be a specific year or "all".
        column: str: The column to summarize.
        mean: Optional[bool]: Whether to include the mean in the summary. Default is True.
        median: Optional[bool]: Whether to include the median in the summary. Default is True.
        std: Optional[bool]: Whether to include the standard deviation in the summary. Default is True.
        min: Optional[bool]: Whether to include the minimum value in the summary. Default is True.
        max: Optional[bool]: Whether to include the maximum value in the summary. Default is True.

    RETURNS:
        Dict[str, Any]: A summary of the data in the specified column.
    """
    logger.info(f"Starting to summarize data for column: {column}")
    returnable_data = {}

    if year not in ["all", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
        logger.error("Invalid year provided")
        raise ValueError("Invalid year provided")

    db = get_db()

    data = db.get_data(year)

    if data is None:
        logger.error("No data found for the specified year")
        raise ValueError("No data found for the specified year")
        return {}

    if mean:
        mean_value = data[column].mean()
        returnable_data['mean'] = mean_value
        logger.info(f"Mean of {column}: {mean_value}")
    if median:
        median_value = data[column].median()
        returnable_data['median'] = median_value
        logger.info(f"Median of {column}: {median_value}")
    if std:
        std_value = data[column].std()
        returnable_data['std'] = std_value
        logger.info(f"Standard Deviation of {column}: {std_value}")
    if min:
        min_value = data[column].min()
        returnable_data['min'] = min_value
        logger.info(f"Minimum of {column}: {min_value}")
    if max:
        max_value = data[column].max()
        returnable_data['max'] = max_value
        logger.info(f"Maximum of {column}: {max_value}")

    return returnable_data

@tool
def generate_statistical_report(year: str, state: Optional[str], starting_month: str, ending_month: str, granularity: Optional[str] = 'D') -> Dict[str, Any]:
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
        granularity: str: The granularity of the report. Valid values are 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual).
    RETURNS:
        A summary of the data of total cases from that year
    """
    if year not in ["all", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
        logger.error("Invalid year provided")
        raise ValueError("Invalid year provided")

    if granularity not in ['D', 'W', 'ME', 'Q', 'A']:
        logger.error(f"Invalid granularity provided: {granularity}")
        raise ValueError("Granularity must be one of the following: 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual)")

    report = {}

    db = get_db()

    data = db.get_data(int(year))
    if data is None:
        logger.error("No data found for the specified year")
        raise ValueError("No data found for the specified year")
        return report

    if state:
        data = data[data['SG_UF_NOT'] == state]

    data['DT_NOTIFIC'] = pd.to_datetime(data['DT_NOTIFIC'])
    mask = (data['DT_NOTIFIC'] >= datetime.date(f'{year}-{starting_month}')) & (data['DT_NOTIFIC'] <= datetime.date(f'{year}-{ending_month}'))
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

    return {'report': report.to_dict('list')}

@tool
def generate_temporal_graphical_report(state: Optional[str], year: Optional[str],  granularity: str = 'D') -> Dict[str, Any]:
    """
    Generates a graphical report about the influenza cases in the selected state if not provided state defaults to 'all',

    ARGS:
        state: str: Brazillian state options with the two first carachters like: 'PE', 'CE', 'SP'
        granularity: str: The granularity of the report. Valid values are 'D' (daily), 'W' (weekly), 'ME' (monthly), 'Q' (quarterly), 'A' (annual). Defaults to 'D'
        year: year that the user prompted, if not provided default to 2025
    RETURNS:
        A dictionary with the figure_id, description from the plot and the data points
    """
    logger.info('Starting temporal graphical report')
    if year not in ["all", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
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
        data['DT_NOTIFIC'] = pd.to_datetime(data['DT_NOTIFIC'])
        grouped = data.fillna(0).groupby(by = ['DT_NOTIFIC', 'SG_UF_NOT']).count().reset_index()
        if state:
            grouped = grouped[grouped['SG_UF_NOT'] == state].set_index('DT_NOTIFIC').resample(granularity).count()
        else:
            grouped = grouped.set_index('DT_NOTIFIC').resample(granularity).count()
            grouped = grouped.iloc[:, 0]
    except Exception as e:
        logger.error(f'Error grouping data: {e}')
        raise e

    try:
        logger.info('Creating the graph')
        grouped = grouped.reset_index()
        
        grouped['DT_NOTIFIC'] = grouped['DT_NOTIFIC'].dt.strftime('%Y-%m-%d')
        
        x = grouped['DT_NOTIFIC'].tolist()
        y = grouped['SG_UF_NOT'].tolist()

        # Return without the full figure object to avoid serialization issues
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