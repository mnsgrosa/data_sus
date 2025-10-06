from fastapi import FastAPI
from schemas import (
    FetchRequestSchema,
    FetchResponseSchema,
    SummaryRequestSchema,
    SummaryResponseSchema,
    ReportRequestSchema,
    ReportResponseSchema,
    GraphicalReportRequestSchema,
    GraphicalReportResponseSchema
)
from src.db.data_storage import SragDb
import httpx
import pnadas as pd
from bs4 import BeautifulSoup
import re


app = FastAPI()
db = SragDb()

@app.post("/fetch", response_model = StoreResponseSchema)
def fetch_data(request: FetchRequestSchema):
    with httpx.Client() as client:
        response = client.get("https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024", timeout=30000.0)
        response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    dropdown = soup.find_all('a', class_ = 'dropdown-item')
    items = [item['href'] for item in dropdown]
    s3_link = None

    if request.years:
        for link in items:
            if 's3' in link::
                s3_link = link
                option = re.search(r'(\d{4})', link)[0]
                if option and int(option) in request.years:
                    df = pd.read_csv(s3_link, sep = ';', low_memory = False)
                    df = df[['SG_UF_NOT', 'DT_NOTIFIC', 'UTI', 'VACINA_COV', 'HOSPITAL', 'EVOLUCAO']]
                    df['year'] = [int(option)] * len(df)
                    insertion_result = db.store_data(df.to_dict(orient = 'records'))
                    if not insertion_result:
                        return {"status": "error", "message": "Failed to insert data into the database"}
    else:
        for item in items:
            if 's3' in item:
                s3_link = item
                option = re.search(r'(\d{4})', item)[0]
                if option:
                    df = pd.read_csv(s3_link, sep = ';', low_memory = False)
                    df = df[['SG_UF_NOT', 'DT_NOTIFIC', 'UTI', 'VACINA_COV', 'HOSPITAL', 'EVOLUCAO']]
                    df['year'] = [int(option)] * len(df)
                    insertion_result = db.store_data(df.to_dict(orient = 'records'))
                    if not insertion_result:
                        return {"status": "error", "message": "Failed to insert data into the database"}

    return {"status": "success", "message": "Data fetched and stored successfully with all available years."}

@app.get("/summary", response_model = SummaryResponseSchema)
def summarize_data(request: SummaryRequestSchema):
    logger.info(f"Starting data summary for: {request}")
    returnable_data = {}

    columns = [column.upper() for column in request.columns]
    years = request.years or [2021, 2022, 2023, 2024, 2025]

    for year in years:
        year_data = db.get_data(year)
        year_data.fillna(-1, inplace = True)
        column_dict = {}
        for column in columns:
            year_dict = {} 
            response = pd.Categorical(data[column], ordered = True)
            year_dict['median'] = np.median(response.codes)
            year_dict['freq'] = data[column].value_counts()
            columns_dict[column] = year_dict
        returnable_data[year] = columns_dict 
    if not returnable_data:
        return {"status": "error", "summaries": "{}"}
    return {"status": "success", "summaries": returnable_data}

@app.get("/report", response_model = ReportResponseSchema)
def generate_report(request: ReportRequestSchema):
    logger.info(f"Starting report generation for: {request}")
    if request.year not in [2021, 2022, 2023, 2024, 2025]:
        return {"status": "error", 
        "report": "Invalid year provided.",
        "death_count": 0,
        "death_rate": 0.0,
        "total_cases": 0,
        "cases_hospitalized": 0,
        "perc_uti": 0.0,
        "perc_vaccinated": 0.0
        }

    if request.granularity not in ['D', 'ME', 'SE', 'M']:
        return {"status": "error", 
        "report": "Invalid granularity provided.",
        "death_count": 0,
        "death_rate": 0.0,
        "total_cases": 0,
        "cases_hospitalized": 0,
        "perc_uti": 0.0,
        "perc_vaccinated": 0.0
        }
    
    report = {}
    data = db.get_data(request.year)
    if data is None or data.empty:
        return {"status": "error", 
        "report": "No data found for the specified year.",
        "death_count": 0,
        "death_rate": 0.0,
        "total_cases": 0,
        "cases_hospitalized": 0,
        "perc_uti": 0.0,
        "perc_vaccinated": 0.0
        }

    if request.state and request.state.lower() != 'all':
        data = data[data['SG_UF_NOT'] == request.state.upper()]

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

        logger.info(f"{uti_cases}")

        return {'status':'success', 
        'death_count': int(death_count),
        'death_rate': float(death_rate),
        'total_cases': int(total_count),
        'cases_hospitalized': int(casos_internados),
        'perc_uti': float(perc_uti),
        'perc_vaccinated': float(perc_vaccinated),
        }
    except Exception as e:
        logger.error(f'Error converting data: {e}')
        raise e

@app.get("/graphical_report", response_model = GraphicalReportResponseSchema)
def generate_graphical_report(request: GraphicalReportRequestSchema):
    if request.year not in [2021, 2022, 2023, 2024, 2025]:
        return {"status": "error", 
        "x": [],
        "y": [],
        "total_points": 0,
        "state": request.state or "all",
        "granularity": request.granularity,
        "message": "Invalid year provided."
        }

    if request.granularity not in ['D', 'ME', 'SE', 'M']:
        return {"status": "error", 
        "x": [],
        "y": [],
        "total_points": 0,
        "state": request.state or "all",
        "granularity": request.granularity,
        "message": "Invalid granularity provided."
        }

    data = db.get_data(request.year)
    if data is None or data.empty:
        return {"status": "error", 
        "x": [],
        "y": [],
        "total_points": 0,
        "state": request.state or "all",
        "granularity": request.granularity,
        "message": "No data found for the specified year."
        }
    
    try:
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