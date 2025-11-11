<h1 align="center">Multimodal Biometric Identification System</h1>

<div align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green" alt="OpenCV">
  <img src="https://img.shields.io/badge/Numpy-Sklearn-yellow" alt="Libraries">
  <img src="https://img.shields.io/badge/Status-Under%20Development-orange" alt="Status">
</div>

> _Pipeline completa per l’elaborazione e l’analisi di impronte digitali basata sul dataset **PolyU HRF DBII** (Hong Kong Polytechnic University High Resolution Fingerprint Database II)._

---

## 1. Introduzione

Questo progetto implementa una **pipeline biometrica modulare** per l’elaborazione e l’estrazione di feature da impronte digitali ad alta risoluzione.  
L’obiettivo principale è costruire un framework **robusto, riproducibile e sperimentalmente verificabile**, capace di affrontare le criticità più comuni nella biometria delle impronte:

- variabilità del contrasto e della luminosità,
- regioni di background e bordi non informativi,
- rumore strutturale e discontinuità nelle ridge.

La pipeline è interamente implementata in **Python**, con il supporto di librerie scientifiche quali `NumPy`, `SciPy`, `OpenCV` e `scikit-image`.  
Tutti i moduli sono stati sviluppati per mantenere **tracciabilità completa delle trasformazioni** e consentire analisi quantitative e qualitative sulle immagini biometriche.

---

## 2. Dataset: PolyU High Resolution Fingerprint Database II (PolyU HRF DBII)

La sperimentazione è basata sul dataset **PolyU HRF DBII**, uno dei benchmark più utilizzati per la ricerca sull’elaborazione di impronte digitali ad alta risoluzione.

### 2.1 Caratteristiche principali

| Proprietà | Valore |
|------------|--------|
| **Origine** | Department of Computing, The Hong Kong Polytechnic University |
| **Nome completo** | High Resolution Fingerprint Database II (DBII) |
| **Numero soggetti** | 148 |
| **Numero immagini per soggetto** | 10 |
| **Totale immagini** | 1480 |
| **Risoluzione** | 1200 dpi (≈ 21 µm/pixel) |
| **Formato** | jpg, 8-bit grayscale |
| **Dimensioni tipiche** | 240×320 o superiori |

Ogni soggetto è rappresentato da 10 campioni acquisiti in sessioni differenti, includendo variazioni di pressione, rotazione e parziale sovrapposizione.  
Questo rende il dataset adatto allo studio della **robustezza dei metodi di enhancement** e alla valutazione della **consistenza topologica** delle ridge.

---

## 3. Struttura generale della pipeline

La pipeline segue una sequenza di fasi organizzate e modulari.  
Ogni fase genera un output intermedio, utilizzato come input per la successiva.

```bash
input → Normalizzazione → Segmentazione → Binarizzazione → Thinning → Orientamento → Estrazione minutiae → Output finale
```

---

## Come eseguire la pipeline

Per eseguire correttamente la pipeline, segui i passaggi descritti di seguito:

### 1. Preparazione del dataset
- Scarica il dataset **PolyU HRF DBII**.  
- Posiziona i file nella cartella `dataset` all’interno del progetto.

### 2. Creazione e configurazione dell’ambiente
- Installa tutte le dipendenze necessarie.  
- Crea l’ambiente Conda eseguendo:
  - `prepare.sh` su macOS/Linux  
  - `prepare.bat` su Windows  
- Assicurati che tutte le librerie vengano installate correttamente.

### 3. Configurazione del database PostgreSQL
- Installa e configura PostgreSQL.  
- Crea il database utilizzando lo script `schema.sql`, che definisce tutte le tabelle necessarie.  
- Verifica che il database sia accessibile e funzionante.

### 4. Configurazione dell’ambiente di esecuzione
- Posiziona il file `.env` nella cartella `config`.  
- Assicurati che il file contenga tutte le variabili necessarie per:
  - Eseguire la pipeline  
  - Connettersi al database
- Es di come deve essere il `.env`:

```bash
# ===========================
# DATABASE
# ===========================
PGHOST=localhost
PGDATABASE=biometria
PGUSER=postgres
PGPASSWORD=super_secret_password
PGPORT=5432

# ===========================
# PRUNING E ORIENTAMENTO
# ===========================
PRUNE_ITERS=2
PRUNE_AREA=2
ORIENT_SIGMA=7.0

# ===========================
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ===========================
CLAHE_CLIP_LIMIT=2.0
CLAHE_TILE_SIZE=8

# ===========================
# FILTRI
# ===========================
BILATERAL_D=5
BILATERAL_SIGMA_COLOR=50.0
BILATERAL_SIGMA_SPACE=7.0
GAUSSIAN_SIGMA=0.7

# ===========================
# SEGMENTAZIONE E BINARIZZAZIONE
# ===========================
SAUVOLA_WIN=25
SAUVOLA_K=0.2
LOCAL_PATCH=64
MIN_OBJ_SIZE=30
MAX_HOLE_SIZE=100
MIN_SEGMENT_AREA=5000

# ===========================
# PARAMETRI GENERALI E VISUALIZZAZIONE
# ===========================
BLOCK_SIZE=16
ENERGY_THRESHOLD=0.01
REL_THRESHOLD=0.2
VIS_SCALE=8

```

### 5. Esecuzione della pipeline
- Attiva l’ambiente Conda corretto:

```bash
conda activate multimodal_biometric
```
- Dalla cartella principale del progetto, esegui lo script principale:

```bash
python -m scripts.run_pipeline
```
- I risultati verranno salvati nella cartella data.

> [!IMPORTANT] 
> **Note aggiuntive**
>
> Per eseguire correttamente la pipeline, assicurati di seguire i passaggi nell’ordine indicato:
>
> 1. Verifica che il database PostgreSQL e l’ambiente Conda siano attivi.
> 2. Esegui gli script di preparazione con i permessi corretti:
>
> ```bash
> ./prepare.sh      # macOS/Linux
> prepare.bat       # Windows
> ```
>
> 3. Attiva l’ambiente Conda corretto:
>
> ```bash
> conda activate multimodal_biometric
> ```
>
> 4. Esegui lo script principale della pipeline dalla cartella di progetto:
>
> ```bash
> python -m scripts.run_pipeline
> ```
>
> I risultati saranno salvati nella cartella `data`.


---

> [!CAUTION]
> Note:
Il progetto è ancora in fase di sviluppo.

--- 

<!--───────────────────────────────────────────────-->
<!--                   AUTORE                     -->
<!--───────────────────────────────────────────────-->

<h2 align="center">✨ Autore</h2>

<p align="center">
  <strong>Giovanni Giuseppe Iacuzzo</strong><br>
  <em>Studente di Ingegneria Dell'IA e della CyberSecurity · Università degli Studi Kore di Enna</em>
</p>

<p align="center">
  <a href="https://github.com/giovanniIacuzzo" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-%40giovanniIacuzzo-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  <a href="mailto:giovanni.iacuzzo@unikorestudent.com">
    <img src="https://img.shields.io/badge/Email-Contattami-blue?style=for-the-badge&logo=gmail" alt="Email"/>
  </a>
</p>