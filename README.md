# Multimodal Biometric Identification System

## 📖 Descrizione del Progetto

Questo progetto mira a sviluppare un **sistema di identificazione biometrica multimodale**, basato su:

- **Impronte digitali**  
- **Palmo della mano**

L'obiettivo è realizzare una pipeline robusta che sfrutti feature estratte tramite **HOG, Gabor e LBP** su più scale, combinate con analisi del **campo di orientamento locale** per classificare pattern globali (Arch, Loop, Whorl) e effettuare clustering interno per identificare soggetti.

Il sistema è pensato come base per applicazioni di **sicurezza biometrica**, autenticazione e ricerca forense.

---

## 🗂 Dataset Utilizzato

- **Nome:** PolyU High-Resolution Fingerprint Database II (PolyU HRF DBII)  
- **Tipo di dati:** immagini ad alta risoluzione di impronte digitali e palmo della mano  
- **Formato:** immagini `.jpg`  
- **Descrizione:**  
  Il dataset contiene campioni multipli per soggetto, con variazioni di pressione e posizione, utili per testare la robustezza delle feature e del clustering.

---

## ⚙️ Requisiti

- Python ≥ 3.10  
- OpenCV  
- scikit-image  
- scikit-learn  
- NumPy  
- Matplotlib

> Assicurarsi di avere un ambiente virtuale dedicato al progetto, ad esempio tramite `conda` o `venv`.

---

## 🏃‍♂️ Come Eseguire

### 1. Preparazione
1. Clonare il repository:
```bash
git clone https://github.com/GiovanniIacuzzo/multimodal_biometric_fingerprints_palms.git
cd multimodal_biometric_fingerprints_palms
```
Posizionare il dataset PolyU HRF DBII nella cartella dataset/ o configurare DATASET_DIR nel file config.py.

2. Esecuzione dello Script di Setup

- Linux / macOS:
```bash
bash prepare.sh
```

- Windows:
```bash
bat prepare.bat
```
Questo configurerà l'intero progetto installando le dipendenze, creando le cartelle per salvare log e immagini e creando l'ambiente conda necessario per eseguire il progetto, come verrà richiesto si dovrà attivare l'ambiente in questo modo:

```bash
conda activate multimodal_biometric
```

3. Esecuzione Manuale
Se si preferisce lanciare manualmente:
- Entrare nella cartella demo_classifier:
```bash
cd demo_classifier
```
- Eseguire il main script:
```bash
python demo_classifier.py
```

- I risultati saranno salvati in:
```bash
labels.csv
```
e eventuali grafici di distribuzione saranno generati automaticamente nella cartella results/ (configurabile).

>Note di Sviluppo:
Il progetto è ancora in fase di sviluppo.
Alcune funzioni, come consensus_kmeans e la standardizzazione globale, possono essere ottimizzate.
La pipeline attuale genera cluster_in_class basati su pattern globali e feature multimodali, ma non è ancora pronta per produzione.