import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List

import pandas as pd

from src.utils.logger import MainLogger


class SragDb(MainLogger):
    def __init__(self):
        super().__init__(__name__)
        self.db_name = "data_sus"
        self.schema_name = "srag"
        self.conn = sqlite3.connect("data_sus.db", check_same_thread=False)
        self._check_schema()
        self.db_file = "data_sus.db"

    @contextmanager
    def get_cursor(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_file, check_same_thread=False)

        cursor = None
        try:
            self.info("Creating cursor")
            cursor = self.conn.cursor()
            yield cursor
            self.info("Committing changes")
            self.conn.commit()
        except Exception as e:
            self.error(f"Error during transaction: {e}. Rolling back changes.")
            if self.conn:
                self.conn.rollback()
            raise e
        finally:
            if cursor:
                self.info("Closing cursor")
                cursor.close()

    def _check_schema(self):
        self.info("Checking the schema")
        with self.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_sus (
                    year INTEGER,
                    SG_UF_NOT TEXT,
                    EVOLUCAO INTEGER,
                    DT_NOTIFIC DATETIME,
                    SEM_NOT INTEGER,
                    UTI INTEGER,
                    VACINA_COV INTEGER,
                    HOSPITAL INTEGER,
                    UNIQUE (year, SG_UF_NOT, EVOLUCAO, DT_NOTIFIC, SEM_NOT, UTI, VACINA_COV, HOSPITAL)
                );
            """)
        return

    def insert(self, data: List[Dict[str, Any]]):
        if data is None:
            self.error("Empty data")
            return False

        insertion_query = """
            INSERT INTO data_sus (year, SG_UF_NOT, EVOLUCAO, DT_NOTIFIC,  SEM_NOT, UTI, VACINA_COV, HOSPITAL)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """

        try:
            self.info("Starting insertion")
            data_to_insert = [
                (
                    d["year"],
                    d["SG_UF_NOT"],
                    d["EVOLUCAO"],
                    d["DT_NOTIFIC"],
                    d["SEM_NOT"],
                    d["UTI"],
                    d["VACINA_COV"],
                    d["HOSPITAL"],
                )
                for d in data
            ]

            with self.get_cursor() as cursor:
                cursor.executemany(insertion_query, data_to_insert)
            self.info("Insertion done")
            return True
        except Exception as e:
            self.error(f"Error while inserting data to the db: {e}")
            return False

    def get_data(self, year: int | str) -> pd.DataFrame:
        if year not in ["all", 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
            self.error("Option not available")
            return None

        self.info("Creating the query")
        query = "SELECT * FROM data_sus"

        if year != "all":
            self.info("Adding the condition")
            query += f" WHERE year = {year}"

        try:
            self.info("getting the data")
            df = pd.read_sql_query(query, self.conn)
            self.info(f"Retrieved the data: {df.shape[0]}")
            return df
        except Exception as e:
            self.error(f"Error retrieving the data: {e}")
            return None
