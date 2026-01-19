import re
from typing import List

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter

from src.utils.logger import MainLogger

from .db.data_storage import SragDb

logger = MainLogger()

BASE_URL = "https://dadosabertos.saude.gov.br"


def fetch_data(request: List[int]):
    logger.info(f"Starting data fetch for: {request}")
    with httpx.Client() as client:
        response = client.get(
            "https://dadosabertos.saude.gov.br/dataset/srag-2021-a-2024",
            timeout=30000.0,
        )
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    dropdown = soup.find_all("a", class_="br-button-primary")

    if not dropdown:
        logger.error(
            "Scraper found no items with class 'dropdown-item'. Check website structure."
        )
        return None

    items = [BASE_URL + item["href"][0] for item in dropdown] if dropdown else []
    links = []

    for item in items:
        with httpx.Client() as client:
            response = client.get(
                item,
                timeout=30000.0,
            )
        response.raise_for_status()
        new_soup = BeautifulSoup(response.text, "html.parser")
        links = new_soup.find_all("a", class_="br-button-primary")
        links = [link if "s3" in links else None for link in links] if dropdown else []

    db = SragDb()
    processed_dfs = []

    if request:
        found_any = False
        logger.info(f"Fetching data for years: {request}")
        for link in items:
            if "s3" in link:
                s3_link = link
                match = re.search(r"(\d{4})", link)
                if not match:
                    continue

                option = match.group(0)

                if int(option) in request:
                    try:
                        logger.info(f"Processing year: {option}")
                        df = pd.read_csv(s3_link, sep=";", low_memory=False)
                        # ... (column filtering remains the same) ...

                        df["year"] = [int(option)] * df.shape[0]

                        # Insert data
                        db.insert(df.to_dict(orient="records"))

                        processed_dfs.append(df)
                        found_any = True

                        # FIX 2: Do NOT return here. Let the loop continue for other years.
                    except Exception as e:
                        logger.error(f"Error processing year {option}: {e}")

        if not found_any:
            logger.error(f"No matching links found for requested years: {request}")
            return None

        # Return the last df or a concatenation, depending on your agent's needs
        return pd.concat(processed_dfs) if processed_dfs else None


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
