import pandas as pd
from pathlib import Path

CSV_PATH = Path("classifier/save_models/id_level_clusters.csv")

def check_id_consistency(csv_path):
    if not csv_path.exists():
        print(f"CSV non trovato: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # Assicurati che ci siano le colonne giuste
    required_cols = {"filename", "global_class"}
    if not required_cols.issubset(df.columns):
        print(f"Colonne mancanti nel CSV. Attese: {required_cols}")
        return

    # Estrai ID dalla filename (prima parte prima di "_")
    df['id'] = df['filename'].apply(lambda x: x.split('_')[0].lstrip("0") or "0")

    # Controlla la coerenza della global_class per ciascun ID
    inconsistent_ids = []
    for fid, group in df.groupby('id'):
        if group['global_class'].nunique() > 1:
            inconsistent_ids.append((fid, group['global_class'].unique()))

    if inconsistent_ids:
        print(f"Trovati ID con classi globali incoerenti:")
        for fid, classes in inconsistent_ids:
            print(f"  ID {fid}: classi {classes}")
    else:
        print("Tutti gli ID hanno una classe globale coerente.")


if __name__ == "__main__":
    check_id_consistency(CSV_PATH)
