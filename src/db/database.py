import psycopg2
from psycopg2.extras import RealDictCursor
from config.config_fingerprint import DB_CONFIG
import json
import logging
from typing import List, Dict

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

def clear_database():
    """
    Svuota tutte le tabelle principali del database:
    - minutiae
    - features_summary
    - images
    ATTENZIONE: operazione distruttiva irreversibile!
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Disabilita temporaneamente i vincoli di foreign key
        cur.execute("SET session_replication_role = 'replica';")

        # Svuota le tabelle in ordine corretto
        cur.execute("TRUNCATE TABLE minutiae RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE TABLE features_summary RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE TABLE images RESTART IDENTITY CASCADE;")

        # Riabilita vincoli
        cur.execute("SET session_replication_role = 'origin';")

        conn.commit()
        print("[DB INFO] Database svuotato correttamente.")
    except Exception as e:
        print(f"[DB ERROR] Errore durante il clear_database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            
def get_all_image_ids() -> list[int]:
    """Recupera tutti gli image_id dal DB come lista di interi."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id FROM images")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Ogni riga è un dizionario: {'id': 1}
    image_ids = [r['id'] for r in rows]
    return image_ids

def save_image_record(subject_id, filename, path_original, path_enhanced, path_skeleton, orientation_mean=None, preprocessing_time=None):
    """Registra i metadati nel database in modo sicuro."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Se subject_id è None, registriamo soggetto "anonimo"
        if subject_id is None:
            subject_id = None  # può restare None o creare entry anonima se necessario

        cur.execute("""
            INSERT INTO images (subject_id, filename, path_original, path_enhanced, path_skeleton, orientation_mean, preprocessing_time_sec, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (subject_id, filename, path_original, path_enhanced, path_skeleton, orientation_mean, preprocessing_time, "done"))
        image_id = cur.fetchone()["id"]
        conn.commit()
        return image_id
    except Exception as e:
        logging.error(f"Errore salvataggio immagine '{filename}' nel DB: {e}")
        return None
    finally:
        if conn:
            conn.close()

def ensure_subject(subject_code: str):
    """Controlla che il soggetto esista nel DB, altrimenti lo crea."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM subjects WHERE subject_code = %s", (subject_code,))
        row = cur.fetchone()
        if row:
            return row["id"]
        # Crea soggetto se non esiste
        cur.execute("INSERT INTO subjects (subject_code) VALUES (%s) RETURNING id", (subject_code,))
        subject_id = cur.fetchone()["id"]
        conn.commit()
        return subject_id
    except Exception as e:
        logging.error(f"Errore gestione soggetto '{subject_code}': {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_image_id_by_filename(filename: str):
    """Restituisce l'ID dell'immagine dato il filename."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM images WHERE filename = %s", (filename,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row["id"] if row else None


def save_minutiae(image_id: int, minutiae_list: list):
    """Inserisce in batch tutte le minutiae estratte per un'immagine."""
    if not minutiae_list:
        return

    conn = get_connection()
    cur = conn.cursor()

    insert_query = """
        INSERT INTO minutiae (image_id, x, y, type, orientation, quality, coherence, cn, deg)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    data = [
        (
            image_id,
            m.get("x"),
            m.get("y"),
            m.get("type"),
            m.get("orientation"),
            m.get("quality"),
            m.get("coherence"),
            m.get("cn"),
            m.get("deg")
        )
        for m in minutiae_list
    ]

    cur.executemany(insert_query, data)
    conn.commit()
    cur.close()
    conn.close()


def save_features_summary(image_id: int, raw_count: int, post_count: int,
                          avg_quality: float, avg_coherence: float,
                          processing_time_sec: float, params: dict):
    """Salva un riassunto dell’estrazione delle feature."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO features_summary (
            image_id, raw_count, post_count,
            avg_quality, avg_coherence, processing_time_sec, params_json
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        image_id, raw_count, post_count,
        avg_quality, avg_coherence, processing_time_sec, json.dumps(params)
    ))
    conn.commit()
    cur.close()
    conn.close()

def save_matching_result(image_a_id: int, image_b_id: int, score: float, method: str = "pair_matching"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO matching_results (image_a_id, image_b_id, score, method)
        VALUES (%s, %s, %s, %s)
    """, (image_a_id, image_b_id, score, method))
    conn.commit()
    cur.close()
    conn.close()

Minutia = Dict[str, float]

def load_minutiae_from_db(image_id: int, min_quality: float = 0.2) -> List[Minutia]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT x, y, type, orientation, quality, coherence, cn, deg
        FROM minutiae
        WHERE image_id = %s AND quality >= %s
    """, (image_id, min_quality))
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

def get_all_image_filenames() -> list[str]:
    """Restituisce tutti i filename presenti in tabella images."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT filename FROM images ORDER BY id ASC;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r["filename"] for r in rows]


def load_minutiae_by_filename(filename: str, min_quality: float = 0.2) -> List[Dict[str, float]]:
    """Carica le minutiae in base al filename (non all'ID numerico)."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT m.x, m.y, m.type, m.orientation, m.quality, m.coherence, m.cn, m.deg
        FROM minutiae m
        JOIN images i ON m.image_id = i.id
        WHERE i.filename = %s AND m.quality >= %s
    """, (filename, min_quality))
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data
