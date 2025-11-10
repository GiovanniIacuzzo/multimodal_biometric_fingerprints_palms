import os
import shutil
import pandas as pd

csv_path = "classifier/save_models/id_level_clusters.csv"
base_output = "results/img/dataset_sorted"

# Carica il CSV
df = pd.read_csv(csv_path)

# Crea la cartella base se non esiste
os.makedirs(base_output, exist_ok=True)

# Itera su ogni riga
for _, row in df.iterrows():
    global_class = row["global_class"]
    cluster = f"cluster_{row['cluster_in_class']}"
    src = row["path"]
    
    # Crea percorso destinazione
    dest_dir = os.path.join(base_output, global_class, cluster)
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copia il file
    dest_path = os.path.join(dest_dir, row["filename"])
    shutil.copy(src, dest_path)

print("Dataset riorganizzato con successo!")
