import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI

from src.utils.logger import MainLogger

from .schemas import PromptResponse

app = FastAPI()
logger = MainLogger(__name__)


@app.post("/prompt", response_model=PromptResponse)
def prompt_agent(prompt: str) -> str: ...


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
