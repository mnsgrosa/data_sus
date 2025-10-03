import sqlite3
import pandas as pd
from typing import Dict, List, Any
from contextlib import contextmanager
from src.utils.logger import MainLogger

class SragDb(MainLogger):
    def __init__(self):
        super().__init__(__name__)
        self.db_name = "data_sus"
        self.schema_name = "srag"
        self.conn = sqlite3.connect("data_sus.db")
        self._check_schema()
            
    @contextmanager
    def get_cursor(self):
        if self.conn:
            self.conn = sqlite3.connect("srag.db")
        
        cursor = None
        try:
            self.info("Creating cursor")
            cursor = self.conn.cursor()
            yield cursor
            self.info("Commiting changes")
            self.conn.commit()
        except Exception as e:
            self.error("Error while using the cursor")
            if cursor:
                cursor.close()
            raise e
        finally:
            self.info("Closing the connection")
            if self.conn:
                self.conn.close()

    def _check_schema(self):
        with self.get_cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_sus (
                    year INTEGER,
                    SG_UF_NOT TEXT NOT NULL,
                    DT_NOTIFIC DATETIME NOT NULL,
                    UTI INTEGER,
                    VACINA_COV INTEGER,
                    HOSPITAL INTEGER
                )
            ''')
        return
    
    def insert(self, data:List[Dict[str, Any]]):
        if data is None:
            self.error("Empty data")
            return False
        
        insertion_query = """
            INSERT INTO data_sus (year, SG_UF_NOT, DT_NOTIFIC, UTI, VACINA_COV, HOSPITAL)
            VALUES (:year, :SG_UF_NOT, :DT_NOTIFIC, :UTI, :VACINA_COV, :HOSPITAL)
            """
        
        try:
            self.info("Starting insertion")
            with self.get_cursor() as cursor:
                for row_data in data :
                    cursor.execute(insertion_query, row_data)
            self.info("Insertion done")
            return True
        except Exception as e:
            self.error(f"Error while inserting data to the db: {e}")
            return False
        
    def get_data(self, year: str):
        if year not in ["all", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
            self.error("Option not available")
            return None
        
        self.info("Creating the query")
        query = "SELECT * FROM srag"

        if year != "all":
            self.info("Adding the condition")
            query += f"WHERE year = '{year}'"

        try:
            self.info("getting the data")
            with self.get_cursor() as cursor:
                df = pd.read_sql_query(query, cursor)
            self.info(f"Retrieved the data: {df.shape[0]}")
            return df
        except Exception as e:
            self.error("Error retrieving the data: {e}")
            return None