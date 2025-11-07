import psycopg2
from psycopg2.extras import RealDictCursor
import os

def get_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            dbname=os.getenv("PGDATABASE", "biometria"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"),
            port=os.getenv("PGPORT", "5432"),
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"[DB ERROR] Connessione fallita: {e}")
        raise
