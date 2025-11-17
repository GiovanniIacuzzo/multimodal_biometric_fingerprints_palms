<h1 align="center">Multimodal Biometric Identification System</h1>

<div align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge" alt="Python Version">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge" alt="OpenCV">
  <img src="https://img.shields.io/badge/Numpy-SciPy-yellow?style=for-the-badge" alt="Libraries">
  <img src="https://img.shields.io/badge/Status-Under%20Development-orange?style=for-the-badge" alt="Status">
</div>

---

## Introduzione

Il presente progetto implementa una **pipeline modulare per l’analisi e il riconoscimento di impronte digitali ad alta risoluzione**, basata su approcci di machine learning avanzati e tecniche di elaborazione delle immagini.  
L’obiettivo è costruire un framework **robusto, riproducibile e sperimentalmente verificabile**, capace di gestire le criticità più comuni nella biometria delle impronte:

- Variabilità del contrasto e della luminosità  
- Presenza di regioni di background non informative  
- Rumore strutturale e discontinuità delle ridge

Tutte le elaborazioni sono realizzate in **Python**, utilizzando librerie scientifiche come `NumPy`, `SciPy`, `OpenCV` e `scikit-image`. La pipeline garantisce **tracciabilità completa delle trasformazioni**, permettendo un’analisi quantitativa e qualitativa approfondita.

---

## Dataset: PolyU High Resolution Fingerprint Database II (PolyU HRF DBII)

La sperimentazione si basa sul dataset **PolyU HRF DBII**, un riferimento consolidato nella ricerca sull’elaborazione di impronte digitali ad alta risoluzione.

### Caratteristiche principali

| Proprietà | Valore |
|------------|--------|
| Origine | Hong Kong Polytechnic University, Department of Computing |
| Nome completo | High Resolution Fingerprint Database II (DBII) |
| Numero soggetti | 148 |
| Immagini per soggetto | 10 |
| Totale immagini | 1480 |
| Risoluzione | 1200 dpi (≈ 21 µm/pixel) |
| Formato | jpg, 8-bit grayscale |
| Dimensioni tipiche | 240×320 o superiori |

> [!NOTE]  
> Ogni soggetto è rappresentato da 10 campioni acquisiti in sessioni differenti, includendo variazioni di pressione, orientamento e parziale sovrapposizione. Questo rende il dataset ideale per testare la robustezza dei metodi di enhancement e valutare la consistenza topologica delle ridge.

---

## Estrazione e Clustering delle Feature

Una volta pre-elaborate le immagini, ogni impronta viene rappresentata tramite un **embedding vettoriale** ottenuto con modelli di **Self-Supervised Learning (SSL)**.  
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

Il matching confronta le feature estratte tra coppie di campioni per determinare corrispondenze:

#### Matching tra campioni

- **Cosine Similarity**: valori vicini a 1 indicano alta somiglianza  
- **Euclidean Distance**: distanza bassa indica corrispondenza

#### Threshold e metriche

- Threshold basso → aumento dei falsi rifiuti (FRR)  
- Threshold alto → aumento dei falsi accettamenti (FAR)

#### Matching basato su cluster

- Campioni nello stesso cluster → maggiore affidabilità  
- Campioni in cluster diversi → bassa probabilità di match

> [!NOTE]  
> Il flusso garantisce un percorso chiaro dalla preparazione dell’immagine fino alla valutazione dei risultati, combinando rappresentazione dei dati e struttura globale dei cluster per ottimizzare precisione e robustezza.

## Struttura PipeLine
  
Ogni fase della pipeline genera un output intermedio, utilizzato come input per la successiva.

```bash
input → Normalizzazione → Segmentazione → Binarizzazione → Thinning → Orientamento → Estrazione minutiae → Matching
```

## Struttura della repository

```bash
├── 📁 classifier
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
├── 📁 scripts
│   └── 🐍 run_pipeline.py
│
├── 📁 src
│   ├── 📁 catalog
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 prepare_catalog.py
│   │
│   ├── 📁 db
│   │   ├── 🐍 database.py
│   │   └── 📄 schema.sql
│   │
│   ├── 📁 evaluation
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 evaluate_performance.py
│   │
│   ├── 📁 features
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 extract_features.py
│   │   └── 🐍 post_processing.py
│   │
│   ├── 📁 matching
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 match_features.py
│   │   └── 🐍 sweep.py
│   │
│   └── 📁 preprocessing
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