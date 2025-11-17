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

## 3. Estrazione e Clustering delle Feature

Dopo aver pre-processato le immagini e applicato il modello **Self-Supervised Learning (SSL)**, ogni impronta digitale viene rappresentata tramite un **embedding vettoriale**.  
Questi embeddings catturano le caratteristiche distintive delle ridge e delle strutture locali dell’impronta in uno spazio ad alta dimensione, permettendo di confrontare campioni tra loro in maniera robusta.

### 3.1 Aggregazione degli Embeddings per Soggetto

Per ridurre la variabilità intra-soggetto dovuta a pressione, orientamento o rumore, le embeddings degli stessi soggetti vengono aggregate (ad esempio tramite media).  
Questo processo genera una rappresentazione unica per ciascun soggetto, facilitando analisi successive come il clustering e la valutazione della similarità tra soggetti.

### 3.2 Clustering Globale

L’obiettivo del clustering è **raggruppare immagini simili tra loro** in base alle loro caratteristiche biometriche, senza utilizzare informazioni sugli ID reali.  
Questo approccio permette di:

- Valutare la capacità degli embeddings di distinguere soggetti diversi.
- Identificare pattern comuni tra impronte simili.
- Misurare la qualità della rappresentazione generata dal modello SSL.

Per ottenere cluster significativi, vengono applicati diversi algoritmi complementari:

1. **KMeans**  
   - Raggruppa i dati in un numero predefinito di cluster ottimizzando la coesione interna.
   - Lavora su embeddings normalizzati per misurare la **cosine similarity** tra vettori.

2. **Agglomerative Clustering**  
   - Algoritmo gerarchico che unisce progressivamente i campioni più simili in cluster.
   - Utile per evidenziare la struttura gerarchica dei dati e individuare eventuali sottogruppi all’interno di ciascun cluster.

### 3.3 Riduzione Dimensionale e Visualizzazione

Poiché gli embeddings sono ad alta dimensione, viene applicata una **riduzione dimensionale** (PCA o UMAP) prima della visualizzazione.  
Questo passaggio consente di:

- Osservare la distribuzione dei campioni nello spazio in 2D o 3D.
- Valutare visivamente la separazione dei cluster.
- Individuare outlier o campioni ambigui.

### 3.4 Valutazione dei Cluster

Per quantificare la qualità dei cluster, si calcolano metriche standard come:

- **Silhouette Score**: misura la coesione interna dei cluster rispetto alla separazione tra cluster diversi.
- **Davies-Bouldin Index**: valuta quanto i cluster sono distinti e compatte le loro strutture.
- **Calinski-Harabasz Index**: analizza la dispersione tra e all’interno dei cluster.

Queste metriche permettono di confrontare diversi algoritmi di clustering e configurazioni del modello SSL, garantendo una valutazione robusta e oggettiva della qualità delle rappresentazioni biometriche.

---
> [!NOTE] 
>
> Questa sezione descrive quindi **la logica e il razionale del clustering**, senza entrare nei dettagli di implementazione, preparando il lettore a comprendere successivamente come eseguire la pipeline.

---

## 4. Pipeline di Elaborazione e Matching

Il sistema biometrico segue un flusso strutturato per garantire precisione e robustezza nel riconoscimento delle impronte digitali. La pipeline si articola in diverse fasi, ognuna con un ruolo specifico.

### 4.1 Preprocessing

Prima di qualsiasi analisi, le immagini di impronte vengono preparate tramite operazioni di **preprocessing**:

- **Ridimensionamento e normalizzazione**: uniforma le immagini per il modello di deep learning.  
- **Rimozione del rumore**: filtri e tecniche di smoothing migliorano la qualità delle ridge.  
- **Binarizzazione preliminare**: eventualmente, per rendere più evidenti i dettagli delle linee.

Questa fase riduce la variabilità dovuta a pressione, orientamento e qualità della scansione.

### 4.2 Segmentazione tramite Deep Learning

Il modello di **Deep Learning (UNet o variante SSL)** viene utilizzato per segmentare le ridge principali e isolare le regioni di interesse:

- Estrae le strutture fondamentali dell’impronta.  
- Migliora il rapporto segnale/rumore e facilita l’estrazione delle feature.  
- Il modello viene allenato in modalità **self-supervised** o con dataset annotati di impronte binarizzate.

### 4.3 Estrazione delle Feature e Minutiae

Dalla segmentazione, si estraggono le **minutiae** e altre feature biometriche:

- **Punti caratteristici**: biforcazioni, terminazioni, ridge endings.  
- **Embeddings vettoriali**: rappresentazioni dense e numeriche dell’impronta, utili per il confronto tra campioni.  
- Queste feature catturano la struttura unica di ciascun soggetto.

### 4.4 Post-Processing

Dopo l’estrazione delle feature, vengono applicati passaggi di **post-processing** per rendere i dati più coerenti:

- **Aggregazione intra-soggetto**: media o pooling delle feature per ridurre variabilità interna.
- **Scelta delle migliori  Minutiae**: tra tutte le minutiae vengono scelte quelle che soddifsano tanti requisiti non devonno stare ai bordi dell'impornta non devono essere troppo addensati, la qualità dell'immagine nel punto di interesse deve essere sopra una soglia, in modo da filtrare le minutiae migliori rispettando il massimo di minutiae concesso.
- **Normalizzazione degli embeddings**: per uniformare la scala e facilitare il matching.  

### 4.5 Matching e Valutazione delle Prestazioni

Il **matching** confronta le feature estratte tra coppie di campioni per determinare corrispondenze:

#### 4.5.1 Matching Tra Campioni

- **Cosine Similarity**: misura l’angolo tra vettori; valori vicini a 1 indicano alta somiglianza.  
- **Euclidean Distance**: distanza “lineare” tra embeddings; valori piccoli indicano corrispondenza.

Il confronto può essere **intra-soggetto** o **inter-soggetto**.

#### 4.5.2 Definizione di Threshold

- Threshold basso → aumenta i falsi rifiuti (FRR)  
- Threshold alto → aumenta i falsi accettamenti (FAR)

#### 4.5.3 Metriche di Valutazione

- **FRR (False Rejection Rate)**  
- **FAR (False Acceptance Rate)**  

#### 4.5.4 Matching Basato su Cluster

- Campioni nello stesso cluster → maggiore affidabilità.  
- Campioni in cluster diversi → bassa probabilità di match.  

Questa fase combina la **rappresentazione dei dati** e la **struttura globale dei cluster** per ottimizzare precisione e robustezza.

---
> [!NOTE]  
> Il flusso descritto garantisce un percorso chiaro dalla preparazione dell’immagine fino alla valutazione dei risultati. Le implementazioni specifiche di segmentazione, estrazione feature e matching saranno illustrate nella sezione dedicata alla pipeline di esecuzione.

## Struttura PipeLine
  
Ogni fase della pipeline genera un output intermedio, utilizzato come input per la successiva.

```bash
input → Normalizzazione → Segmentazione → Binarizzazione → Thinning → Orientamento → Estrazione minutiae → Matching
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