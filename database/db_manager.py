import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
from config.config import DATABASE_URL, DATABASE_DIR, DB_PATH
from config.logging_config import get_logger

logger = get_logger("DBManager")

class DatabaseManager:
    """Manages database connection, table initialization, and ETL loading for PostgreSQL and SQLite."""

    def __init__(self, db_url: str = None):
        self.db_url = db_url or str(DATABASE_URL)
        self.is_sqlite = self.db_url.startswith("sqlite")
        self.engine = create_engine(self.db_url, echo=False)
        logger.info(f"Initialized DatabaseManager connected to: {self.db_url}")

    def initialize_schema(self, schema_file: str = None):
        """Executes the DDL statements from schema.sql."""
        if schema_file is None:
            schema_file = DATABASE_DIR / "schema.sql"
        
        with open(schema_file, "r", encoding="utf-8") as f:
            sql_statements = f.read()

        logger.info(f"Applying schema definitions from {schema_file}...")
        
        # Execute individual CREATE TABLE statements
        statements = [stmt.strip() for stmt in sql_statements.split(";") if stmt.strip()]
        with self.engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
        logger.info("Database schema initialized successfully.")

    def load_table(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
        """Loads a Pandas DataFrame into the designated database table."""
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"Loaded {len(df)} rows into table '{table_name}'.")
        except Exception as e:
            logger.error(f"Error loading data into table '{table_name}': {e}")
            raise e

    def query(self, sql_query: str) -> pd.DataFrame:
        """Executes a SQL SELECT query and returns the results as a pandas DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(sql_query), conn)

    def get_table_counts(self) -> dict:
        """Returns row counts for all database tables."""
        tables = [
            "dim_student",
            "dim_course",
            "dim_demographics",
            "dim_macro_economy",
            "fact_student_performance",
            "student_analytical_mart"
        ]
        counts = {}
        with self.engine.connect() as conn:
            for t in tables:
                try:
                    res = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                    counts[t] = res
                except Exception:
                    counts[t] = 0
        return counts

if __name__ == "__main__":
    db = DatabaseManager()
    db.initialize_schema()
