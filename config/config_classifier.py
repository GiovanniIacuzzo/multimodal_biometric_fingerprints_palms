import os
import yaml
from types import SimpleNamespace

def load_config(yaml_path: str = "config/config_classifier.yml"):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Risolve percorsi assoluti ---
    root_dir = os.path.abspath(os.path.join(os.path.dirname(yaml_path), cfg["paths"]["root_dir"]))
    for k, v in cfg["paths"].items():
        if isinstance(v, str) and v.startswith("./"):
            cfg["paths"][k] = os.path.abspath(os.path.join(root_dir, v[2:]))

    # --- Crea automaticamente le directory principali ---
    for d in [cfg["paths"]["save_dir"], cfg["paths"]["figures_dir"], cfg["paths"]["sorted_dir"]]:
        os.makedirs(d, exist_ok=True)

    # --- Converti in SimpleNamespace per compatibilità ---
    def to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
        return d

    return to_namespace(cfg)

# ============================================================
# TEST DI CARICAMENTO
# ============================================================
if __name__ == "__main__":
    CONFIG = load_config()
    print("=== CONFIGURAZIONE SSL ===")
    print(CONFIG.ssl.model.backbone)
    print(CONFIG.paths.save_dir)
    print(CONFIG.ssl.logging.log_file)
