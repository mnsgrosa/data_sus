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


def find_correct_soup(dataset_link: str, years: list[str]):
    with httpx.Client() as client:
        response = client.get(dataset_link, timeout=300)
    soup = BeautifulSoup(response.text, "html.parser")

    title = (
        soup.find("h1", class_="text-weight-bold mt-3").text.lower() if soup else None
    )

    if title and "ficha" in title:
        return None

    if title and "csv" in title:
        match = [title for year in years if year in title]
        return soup.find_all("a") if match else None


def find_s3(soup):
    s3_links = [
        link["href"] for link in soup if link.get("href") and "s3" in link["href"]
    ]
    return s3_links[0]


def fetch_data(request: List[int]):
    logger.info(f"Starting data fetch for: {request}")

    str_years = list(map(str, request))

    with httpx.Client() as client:
        response = client.get(
            "https://dadosabertos.saude.gov.br/dataset/srag-2021-a-2024",
            timeout=30000.0,
        )
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    buttons = soup.find_all("a", class_="br-button primary")

    logger.info(f"Found candidates:{buttons}")

    if not buttons:
        logger.error(
            "Scraper found no items with class 'dropdown-item'. Check website structure."
        )
        return None

    links = [
        BASE_URL + item["href"]
        for item in buttons
        if item.get("href") and "dataset" in item["href"]
    ]

    logger.info(f"Links found:{links}")

    s3_links = []

    for link in links:
        link_soup = find_correct_soup(link, str_years)
        if link_soup:
            s3_links.append(find_s3(link_soup))

    if s3_links is None:
        logger.info("No s3 link found")
        return None

    logger.info(f"S3 links found:{s3_links}")

    db = SragDb()
    processed_dfs = []

    for s3 in s3_links:
        year = s3.split("/")[5]
        logger.info(f"Processing: {s3} from {year}")
        df = pd.read_csv(s3, sep=";", low_memory=False)
        df["year"] = [year] * df.shape[0]
        insertion_result = db.insert(df.to_dict(orient="records"))
        if insertion_result:
            logger.info(f"Succesfully inserted: {year}")
            processed_dfs.append(year)
        else:
            logger.info(f"Couldn't insert: {year}")
