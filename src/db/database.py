import psycopg2
from psycopg2.extras import RealDictCursor
from config.config import DB_CONFIG


def get_connection():
    """Crea una connessione sicura al database PostgreSQL usando i parametri in config.DB_CONFIG."""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            dbname=DB_CONFIG["dbname"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            port=DB_CONFIG["port"],
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"[DB ERROR] Connessione fallita: {e}")
        raise
