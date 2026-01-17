import re
from typing import List

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter

from src.utils.logger import MainLogger

from .db.data_storage import SragDb

logger = MainLogger()


def fetch_data(request: List[int]):
    logger.info(f"Starting data fetch for: {request}")
    with httpx.Client() as client:
        response = client.get(
            "https://dadosabertos.saude.gov.br/dataset/srag-2021-a-2024",
            timeout=30000.0,
        )
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    dropdown = soup.find_all("a", class_="dropdown-item")
    items = [item["href"] for item in dropdown] if dropdown else None
    s3_link = None

    if items is None:
        logger.info("Scrapper haven't found any year in its db")
        return None

    db = SragDb()

    if request:
        logger.info(f"Fetching data for years: {request}")
        try:
            for link in items:
                if "s3" in link:
                    s3_link = link
                    option = re.search(r"(\d{4})", link)[0]
                    if option and int(option) in request:
                        df = pd.read_csv(s3_link, sep=";", low_memory=False)
                        df = df[
                            [
                                "SG_UF_NOT",
                                "DT_NOTIFIC",
                                "UTI",
                                "VACINA_COV",
                                "SEM_NOT",
                                "HOSPITAL",
                                "EVOLUCAO",
                            ]
                        ]
                        df["year"] = [int(option)] * df.shape[0]
                        insertion_result = db.insert(df.to_dict(orient="records"))
                        return df

        except Exception as e:
            logger.error(f"Error fetching data: {e}")

            return None
    else:
        for item in items:
            if "s3" in item:
                s3_link = item
                option = re.search(r"(\d{4})", item)[0]
                if option:
                    df = pd.read_csv(s3_link, sep=";", low_memory=False)
                    df = df[
                        [
                            "SG_UF_NOT",
                            "DT_NOTIFIC",
                            "UTI",
                            "VACINA_COV",
                            "SEM_NOT",
                            "HOSPITAL",
                            "EVOLUCAO",
                        ]
                    ]
                    df["year"] = [int(option)] * df.shape[0]
                    insertion_result = db.insert(df.to_dict(orient="records"))
                    if not insertion_result:
                        return None
                    return df

    return None


def extract_data_dictionary(url: str):
    logger = MainLogger(__name__)
    tables = (
        DocumentConverter().convert(url).document.export_to_dict().get("tables", [])
    )

    colunas, caracteristicas, tipos = [], [], []
    tipos_colunas = ["varchar", "date", "number"]

    for table in tables:
        for conteudo in table["data"]["table_cells"]:
            texto = conteudo.get("text", "")
            if texto in [
                "EVOLUCAO",
                "UTI",
                "DT_NOTIFIC",
                "SEM_NOT",
                "SG_UF_NOT",
                "VACINA_COV",
                "HOSPITAL",
                "SEM_NOT",
            ]:
                colunas.append(texto)
            elif (
                tipos_colunas[0] in texto.lower()
                or tipos_colunas[1] in texto.lower()
                or tipos_colunas[2] in texto.lower()
            ):
                tipos.append(texto)
            elif "campo" in texto.lower():
                caracteristicas.append(texto)
        logger.info(f"{texto}")

    structs = {}
    for i in range(len(colunas)):
        structs[colunas[i]] = {
            "tipo": tipos[i] if i < len(tipos) else None,
            "caracteristicas": caracteristicas[i] if i < len(caracteristicas) else None,
        }

    logger.info(f"{structs}")
    return structs
