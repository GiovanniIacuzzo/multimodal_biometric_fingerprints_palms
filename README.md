<h1 align="center">Multimodal Biometric Identification System</h1>

<div align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge" alt="Python Version">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge" alt="OpenCV">
  <img src="https://img.shields.io/badge/Numpy-SciPy-yellow?style=for-the-badge" alt="Libraries">
  <img src="https://img.shields.io/badge/Status-Under%20Development-orange?style=for-the-badge" alt="Status">
</div>

---

# Introduzione

Questo progetto implementa una **pipeline modulare e completamente automatizzata per l’analisi, il preprocessing e il riconoscimento di impronte digitali ad alta risoluzione**.  
L’intero framework è stato progettato per supportare sperimentazioni riproducibili e scalabili nel campo della biometria, integrando:

- tecniche avanzate di **image enhancement**,  
- estrazione accurata delle **minuzie**,  
- matching basato su invarianti geometriche,  
- gestione multi-dataset con parsing intelligente dei filename,  
- strumenti di valutazione (FAR, FRR, ROC) a livello sperimentale.

### Obiettivi principali del framework

- **Robustezza**: resistenza a variazioni di pressione, rotazione, contrasto, rumore e parziale sovrapposizione delle ridge.  
- **Modularità**: ogni fase della pipeline (preprocessing → estrazione → matching → valutazione) può essere sostituita o estesa.  
- **Riproducibilità**: ogni trasformazione è tracciata e configurabile.  
- **Multi-dataset**: supporto integrato a dataset con formati eterogenei e convenzioni diverse.

### Tecnologie utilizzate

- **Python 3.x**  
- Librerie scientifiche: `NumPy`, `SciPy`, `OpenCV`, `scikit-image`  
- Machine learning e KD-Tree: `scikit-learn`  
- Analisi e catalogazione dataset: `pandas`, `tqdm`  
- Logging, benchmarking e strumenti diagnostici integrati.

---

# Dataset utilizzati

La pipeline supporta e normalizza **qualunque dataset di impronte digitali con formato leggibile**, tramite un sistema di riconoscimento dei filename basato su espressioni regolari.  
In questo progetto sono stati impiegati due dataset principali:

---

## 📌 PolyU High Resolution Fingerprint Database II (PolyU HRF DBII)

Questo dataset rappresenta un riferimento consolidato nella ricerca sulle impronte digitali ad alta risoluzione.

### Caratteristiche principali

| Proprietà | Valore |
|------------|--------|
| Origine | Hong Kong Polytechnic University |
| Nome | High Resolution Fingerprint Database II (DBII) |
| Soggetti | 148 |
| Immagini per soggetto | 10 |
| Totale immagini | 1480 |
| Risoluzione | 1200 dpi (≈ 21 µm/pixel) |
| Formato | JPG – 8-bit grayscale |
| Dimensioni | ~240×320 px |

> [!NOTE]  
> Ogni soggetto dispone di 10 acquisizioni indipendenti, con variazioni di rotazione, pressione, area acquisita e condizioni di contatto.  
> Questo lo rende ideale per valutare la stabilità delle minuzie e l’affidabilità del matching.

---

## 📌 NIST Fingerprint

Oltre al PolyU HRF, il progetto integra anche delle impronte **NIST**, caratterizzate da elevate difficoltà strutturali:

- impronte estremamente degradate,  
- artefatti e zone sature,  
- ridotto contrasto,  
- geometrie incomplete o danneggiate,
- acquisizione grossolana non ottima.

### Caratteristiche riconosciute

| Proprietà | Valore |
|-----------|--------|
| Nome pattern | `Fxxxx_nn.bmp` |
| Esempio | `F0001_01.bmp` |
| Parsing automatico | Sì (subject, finger, session=1) |
| Complessità | Molto alta |
| Formato | BMP, 8-bit grayscale |

> [!TIP]  
> Le impronte NIST sono utilizzate principalmente per **stress-test** della pipeline, poiché contengono casi estremi che mettono in difficoltà i metodi convenzionali.

---

# Sistema di Catalogazione Dataset

Per uniformare i dataset PolyU e NIST, la pipeline utilizza il modulo:
```bash
src/catalog/catalog.py
```
Questo componente:
1. **scansiona automaticamente tutti i cluster** (cartelle `cluster_*`);
2. **riconosce automaticamente il formato del filename** tramite tre regex:
   - `3_1_1.jpg` → formato standard  
   - `F0003_10.bmp` → formato NIST  
   - `S1387_02.bmp` → formato "S-pattern"  
3. **estrae i metadati**:  
   - `subject_id`  
   - `finger_id`  
   - `session_id`  
   - dimensioni dell'immagine  
   - cluster di appartenenza  
4. **genera un catalogo CSV** ordinato e pronto per tutte le successive fasi della pipeline.

>[!IMPORTANT]
>L’unificazione dei dataset tramite questa catalogazione è fondamentale per permettere un matching affidabile e un calcolo coerente delle metriche (FRR / FAR / ROC).
---

## Estrazione e Clustering delle Feature

Prima di passare all'elaboraizione delle immagini, ogni impronta viene rappresentata tramite un **embedding vettoriale** ottenuto con modelli di **Self-Supervised Learning (SSL)**.  
Questi embeddings catturano le caratteristiche distintive delle ridge e permettono un confronto affidabile tra campioni.

### Aggregazione degli embeddings

Per ridurre la variabilità intra-soggetto dovuta a pressione, orientamento o rumore, gli embeddings degli stessi soggetti vengono aggregati (ad esempio tramite media), generando una **rappresentazione unica per ciascun soggetto**.

### Clustering Globale

L’obiettivo del clustering è raggruppare immagini simili in base alle loro caratteristiche biometriche, senza fare uso degli ID reali dei soggetti. Questo approccio consente di:

- Valutare la capacità degli embeddings di distinguere soggetti diversi  
- Identificare pattern comuni tra impronte simili  
- Misurare la qualità della rappresentazione generata dal modello SSL

**Algoritmi applicati:**

1. **KMeans** – Raggruppa i dati in cluster ottimizzando la coesione interna, lavorando su embeddings normalizzati per misurare la **cosine similarity**.  
2. **Agglomerative Clustering** – Algoritmo gerarchico che unisce progressivamente i campioni più simili, utile per evidenziare eventuali sottogruppi.

### Riduzione dimensionale e visualizzazione

Poiché gli embeddings sono ad alta dimensione, viene applicata una **riduzione dimensionale** (PCA o UMAP) prima della visualizzazione. Questo consente di:

- Osservare la distribuzione dei campioni nello spazio 2D o 3D  
- Valutare visivamente la separazione dei cluster  
- Individuare eventuali outlier o campioni ambigui

### Valutazione dei cluster

La qualità dei cluster viene quantificata attraverso metriche consolidate:

- **Silhouette Score** – Misura coesione interna vs separazione  
- **Davies-Bouldin Index** – Valuta quanto i cluster sono distinti e compatte le loro strutture  
- **Calinski-Harabasz Index** – Analizza la dispersione tra e all’interno dei cluster

> [!NOTE]  
> Questa sezione fornisce la logica e il razionale del clustering, senza entrare nei dettagli implementativi. Prepara il lettore a comprendere la pipeline pratica di esecuzione.

---

## Pipeline di Elaborazione e Matching

La pipeline è articolata in fasi strutturate per garantire **precisione e robustezza** nel riconoscimento delle impronte digitali.

### Preprocessing

Le immagini vengono preparate tramite operazioni di preprocessing:

- Ridimensionamento e normalizzazione
- Rimozione del rumore tramite filtri e smoothing
- Eventuale binarizzazione preliminare per evidenziare dettagli delle ridge

### Segmentazione con Deep Learning

Il modello **UNet o SSL** segmenta le ridge principali, isolando le regioni di interesse:

- Estrazione delle strutture fondamentali  
- Miglioramento del rapporto segnale/rumore  
- Facilitazione dell’estrazione delle feature

### Estrazione delle feature e minutiae

Dalla segmentazione si estraggono:

- **Minutiae**: biforcazioni, terminazioni e punti caratteristici  
- **Embeddings vettoriali**: rappresentazioni numeriche dense dell’impronta, utili per il confronto

### Post-Processing

- Aggregazione intra-soggetto delle feature  
- Selezione delle minutiae migliori in base a criteri di qualità e distribuzione  
- Normalizzazione degli embeddings per uniformare la scala

### Matching e valutazione delle prestazioni

Il sistema utilizza un modello di matching basato su **RANSAC** e **trasformazioni rigide**, progettato per confrontare strutture di minutiae in modo robusto contro rotazioni, traslazioni e distorsioni locali.

#### Matching tra campioni

Il confronto tra due impronte avviene in tre fasi:

1. **Selezione preliminare delle corrispondenze**
   - ciascuna minutia viene confrontata con le vicine (KDTree)
   - vengono applicati vincoli su:
     - distanza locale
     - differenza di orientazione
     - tipo della minutia (ending/bifurcation)

2. **Stima della trasformazione (RANSAC)**
   - si cerca la rotazione + traslazione che massimizza gli *inliers*
   - le minutiae vengono allineate nel sistema di riferimento comune

3. **Valutazione delle corrispondenze**
   - ogni coppia minutia–minutia allineata riceve un **peso**
     basato su:
     - coerenza geometrica
     - differenza angolare
     - tipo della minutia
     - qualità locale
   - lo **score finale** è normalizzato in $([0, 1])$:
     - **1 → impronte altamente corrispondenti**
     - **0 → quasi certamente impostore**

#### Threshold e metriche

Il sistema calcola le metriche biometriche standard:

- **FRR(t)** – False Reject Rate: genuine con score < t  
- **FAR(t)** – False Accept Rate: impostor con score ≥ t  

Effetto del threshold:

- Threshold basso → FRR più alto (sistema più severo)  
- Threshold alto → FAR più alto (sistema più permissivo)  

> [!NOTE]  
> Questo approccio combina robustezza geometrica e pesatura delle minutiae,
> offrendo un matching stabile anche in presenza di rumore, rotazioni e pressioni non uniformi.

## Struttura PipeLine
  
Ogni fase della pipeline genera un output intermedio, utilizzato come input per la successiva.

```bash
input → Normalizzazione → Segmentazione → Binarizzazione → Thinning → Orientamento → Estrazione minutiae → Matching
```

## Struttura della repository

```bash
├── 📁 classifier
│   │
│   ├── 📁 dataset2
│   │   ├── 🐍 dataset.py
│   │   └── 🐍 preprocessing.py
│   │
│   ├── 📁 models
│   │   ├── 🐍 backbone.py
│   │   ├── 🐍 projection_head.py
│   │   └── 🐍 ssl_model.py
│   │
│   ├── 📁 utils
│   │   ├── 🐍 cluster_embeddings.py
│   │   ├── 🐍 extract_embeddings.py
│   │   ├── 🐍 loss.py
│   │   ├── 🐍 train_ssl.py
│   │   └── 🐍 utils.py
│   │
│   ├── 🐍 main_ssl_pipeline.py
│   ├── 🐍 sorted.py
│   └── 🐍 verify.py
│
├── 📁 config
│   ├── 🐍 config_classifier.py
│   ├── ⚙️ config_classifier.yml
│   ├── 🐍 config_fingerprint.py
│   ├── ⚙️ config_fingerprint.yml
│   ├── ⚙️ config_matching.yml
│   ├── ⚙️ config_path.yml
│   ├── ⚙️ config_segmentation.yml
│   └── ⚙️ environment.yml
│
├── 📁 src
│   ├── 📁 catalog
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 prepare_catalog.py
│   │
│   ├── 📁 features
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 extract_features.py
│   │   └── 🐍 post_processing.py
│   │
│   ├── 📁 matching
│   │   ├── 🐍 FAR.py
│   │   ├── 🐍 FRR.py
│   │   ├── 🐍 ROC.py
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 match.py
│   │   ├── 🐍 match_features.py
│   │   └── 🐍 utils.py
│   └── 📁 preprocessing
│       │
│       ├── 📁 segmentation
│       │   ├── 🐍 __init__.py
│       │   ├── 🐍 dataset.py
│       │   ├── 🐍 inference.py
│       │   ├── 🐍 model.py
│       │   └── 🐍 train.py
│       │
│       ├── 🐍 __init__.py
│       ├── 🐍 fingerprint_preprocess.py
│       ├── 🐍 orientation.py
│       └── 🐍 run_preprocessing.py
│
├── ⚙️ .gitignore
├── 📝 README.md
│
├── 📄 prepare.bat
└── 📄 prepare.sh
```

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