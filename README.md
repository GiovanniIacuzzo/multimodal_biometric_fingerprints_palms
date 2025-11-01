<h1 align="center">Multimodal Biometric Identification System</h1>

<h2 align="left">🚀 Descrizione del Progetto</h2>

<div align="justify">
Benvenuti in <strong>Multimodal Biometric Identification System</strong>, un progetto accademico sviluppato per sostenere l'esame di <strong>Sistemi di Identificazione Biometrica</strong>.  
L’obiettivo è realizzare una pipeline <strong>robusta, modulare e scalabile</strong> per l’identificazione biometrica multimodale, combinando dati di <strong>impronte digitali</strong> e <strong>palmo della mano</strong>.
</div>

<h3 align="left">🔬 Approccio</h3>

<div align="justify">
Il sistema integra diverse tecniche avanzate di estrazione delle feature per catturare sia le <strong>micro-strutture locali</strong> che i <strong>pattern globali</strong>:
</div>

- **HOG (Histogram of Oriented Gradients):** cattura orientamenti e contorni locali.  
- **Gabor Filters:** evidenzia frequenze e orientamenti tipici delle creste epidermiche.  
- **LBP (Local Binary Patterns):** descrive micro-texture locali, utile per pattern minutiae.  
- **Campo di orientamento locale:** analizza la direzione predominante delle creste per classificare pattern globali: *Arch*, *Loop*, *Whorl*.  

<div align="justify">
Questa combinazione permette di ottenere una <strong>rappresentazione multimodale ricca e discriminante</strong>, pronta per il clustering interno e l’identificazione dei soggetti.
</div>

<h3 align="left">🛡 Applicazioni</h3>

<div align="justify">
Il progetto fornisce una base solida per applicazioni in:
</div>

- **Sistemi di sicurezza biometrica avanzati**  
- **Autenticazione e controllo accessi**  
- **Ricerca forense e studi scientifici su pattern biometrici**

<hr>

<h2 align="left">🗂 Dataset Utilizzato</h2>

<div align="justify">
Per testare l’affidabilità e la robustezza del sistema è stato scelto un dataset di riferimento:
</div>

- **Nome:** PolyU High-Resolution Fingerprint Database II (PolyU HRF DBII)  
- **Tipo di dati:** immagini ad alta risoluzione di impronte digitali e palmo della mano  
- **Formato:** `.jpg`  
- **Caratteristiche principali:**  
  - Campioni multipli per soggetto con variazioni di pressione e posizione  
  - Permette di testare <strong>robustezza, accuratezza e generalizzazione</strong> delle feature estratte  

> 💡 Il dataset consente di validare la pipeline in scenari realistici e complessi, simulando applicazioni di sicurezza reali.

<hr>

<h2 align="left">⚙️ Requisiti di Sistema</h2>

<div align="justify">
Il progetto è sviluppato in <strong>Python 3.10+</strong> e richiede le principali librerie scientifiche:
</div>

| Libreria          | Versione minima | Funzione principale                     |
|------------------|----------------|----------------------------------------|
| OpenCV            | ≥ 4.5          | Elaborazione immagini                  |
| scikit-image      | ≥ 0.19         | Feature extraction (HOG, LBP, ecc.)  |
| scikit-learn      | ≥ 1.3          | Clustering, PCA, t-SNE                |
| NumPy             | ≥ 1.24         | Computazione numerica                  |
| Matplotlib        | ≥ 3.7          | Visualizzazione opzionale             |

> 💡 Consiglio: creare un ambiente virtuale dedicato (`conda` o `venv`) per mantenere tutte le dipendenze isolate e garantire la riproducibilità.



## Struttura della repository:

```bash
├── 📁 config
│   └── ⚙️ environment.yml
│
├── 📁 demo classifier
│   ├── 🐍 classification.py
│   ├── 🐍 clustering.py
│   ├── 🐍 config.py
│   ├── 🐍 demo_classifier.py
│   ├── 🐍 feature_fun.py
│   └── 🐍 visualization.py
│
├── ⚙️ .gitignore
│
├── 📝 README.md
│
├── 📄 prepare.bat
└── 📄 prepare.sh
```

---

## 🏃‍♂️ Come Eseguire

### 1. Preparazione
1. Clonare il repository:
```bash
git clone https://github.com/GiovanniIacuzzo/multimodal_biometric_fingerprints_palms.git
cd multimodal_biometric_fingerprints_palms
```
Posizionare il dataset `PolyU HRF DBII` nella cartella `dataset/` o configurare `DATASET_DIR` nel file `config.py`.

2. Esecuzione dello Script di Setup

- Linux / macOS:
```bash
bash prepare.sh
```

- Windows:
```bash
prepare.bat
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