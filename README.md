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

## 4. Descrizione delle fasi di elaborazione

### 4.1 Normalizzazione e Preprocessing iniziale

**Obiettivo:** ridurre le variazioni d’intensità e migliorare il contrasto tra ridge e valley.  
La normalizzazione assicura che ogni immagine abbia un range dinamico coerente e indipendente dalle condizioni di acquisizione.

**Implementazione:**

- **Normalizzazione lineare:**

  Ogni pixel \( I(x, y) \) viene rimappato in funzione della media e deviazione standard dell’immagine:

  ```math
  I_{norm}(x,y) = \frac{I(x,y) - \mu_I}{\sigma_I} \cdot \sigma_0 + \mu_0
  ```
dove:
- \mu_I, \sigma_I: media e deviazione standard dei valori di intensità dell’immagine originale;
- \( \mu_0 \), \( \sigma_0 \): parametri target di normalizzazione scelti empiricamente (es. 100 e 100);
- \( I_{norm}(x,y) \): valore normalizzato del pixel.

L’obiettivo di questa fase è rendere le impronte confrontabili tra loro, riducendo l’impatto di illuminazione, sensore e condizioni di contatto.

---

### 4.2 Segmentazione e mascheramento

**Scopo:** separare le regioni effettivamente biometriche (contenenti ridge) dal background o da aree prive di informazione.

**Procedura:**
1. Suddivisione dell’immagine in blocchi (es. 16×16 pixel);
2. Calcolo della varianza locale dell’intensità per ogni blocco;
3. Classificazione dei blocchi come “foreground” se la varianza supera una soglia empirica;
4. Applicazione di un filtro morfologico (closing + opening) per affinare la maschera;
5. Riduzione del rumore ai bordi tramite erosione controllata.

La **maschera binaria risultante** viene utilizzata per vincolare le elaborazioni successive esclusivamente alle regioni biometriche.

---

### 4.3 Binarizzazione

Una volta isolata la regione di interesse, l’immagine viene convertita in formato binario (ridge/valley).  
Sono stati sperimentati diversi approcci:

- **Binarizzazione globale (Otsu):**  
  efficace su immagini omogenee ma sensibile a variazioni locali;

- **Binarizzazione adattiva (Gaussian/Mean Adaptive Thresholding):**  
  migliora la separazione delle ridge in presenza di non uniformità di contrasto;

- **Metodi basati su Gabor filtering:**  
  sfruttano la direzionalità delle creste per preservare la continuità strutturale.

Il risultato è un’immagine binaria \( B(x,y) \in \{0,1\} \), dove le ridge sono rappresentate dai pixel bianchi.

---

### 4.4 Scheletrizzazione (Thinning)

**Obiettivo:** ridurre le ridge a spessori un pixel mantenendone la topologia.  
Viene impiegato l’algoritmo di **Zhang–Suen** o, in alternativa, la funzione `cv2.ximgproc.thinning()` di OpenCV, ottimizzata per immagini binarie.

Questa fase è cruciale poiché la qualità dello scheletro influenza direttamente la precisione dell’estrazione delle *minutiae*.

---

### 4.5 Calcolo dell’orientamento e mappa di frequenza

Per stimare la direzione delle creste e la loro periodicità si analizzano finestre locali (es. 16×16 pixel) calcolando:

- Gradiente \( G_x, G_y \) tramite Sobel;
- Angolo di orientamento medio:  
  ```math
  \theta = \frac{1}{2} \tan^{-1}\left(\frac{2 \sum G_x G_y}{\sum(G_x^2 - G_y^2)}\right)
  ```
dove:
- \( G_x, G_y \): componenti del gradiente calcolate rispettivamente lungo gli assi orizzontale e verticale;
- la somma è eseguita sull’intera finestra locale di analisi.

La mappa di orientamento risultante fornisce una rappresentazione direzionale delle creste, utile per:
- guidare i filtri di potenziamento orientati (ad es. Gabor o Butterworth direzionali);
- validare la coerenza angolare delle *minutiae* estratte;
- migliorare la robustezza dei matching basati su rotazioni.

Parallelamente, la **frequenza delle ridge** viene stimata tramite l’analisi del profilo proiettato ortogonalmente alla direzione dominante, calcolando la distanza media tra picchi successivi.  
La combinazione di orientamento e frequenza permette di generare una **mappa morfologica locale** utile per l’enhancement adattivo.

---

### 4.6 Estrazione delle Minutiae

**Scopo:** individuare e classificare i punti di terminazione e biforcazione delle ridge nello scheletro binarizzato.

L’estrazione si basa sull’analisi della **connettività 8-neighbors** di ciascun pixel bianco \( P(x, y) \) nello scheletro:

- Se il numero di pixel bianchi nel vicinato è **1 → terminazione**;
- Se il numero è **3 → biforcazione**;
- Gli altri casi vengono ignorati come punti intermedi.

Dopo l’identificazione preliminare, si applicano diversi **criteri di validazione**:

1. **Coerenza direzionale:**  
   la direzione della minutia deve essere compatibile con la mappa di orientamento locale;
2. **Distanza minima:**  
   due minutiae troppo vicine (< 6–8 px) vengono fuse o scartate;
3. **Maschera di validità:**  
   le minutiae esterne alla regione biometrica vengono eliminate;
4. **Rimozione falsi positivi:**  
   mediante analisi morfologica e controlli di simmetria locale.

Il risultato finale è un insieme di minutiae \( M = \{ m_i \}_{i=1}^{N} \) rappresentato da tuple:
```math
m_i = (x_i, y_i, \theta_i, t_i)
```
dove:
- \( (x_i, y_i) \) sono le coordinate spaziali della minutia all’interno dell’immagine scheletrizzata;
- \( \theta_i \) rappresenta l’orientamento locale della cresta in corrispondenza del punto;
- \( t_i \) indica la tipologia della minutia (`ending` o `bifurcation`).

---

### 4.7 Post–Processing e Filtraggio Avanzato delle Minutiae

Dopo l’estrazione grezza, le minutiae vengono sottoposte a una fase di **post–processing** finalizzata a ridurre errori dovuti a discontinuità topologiche, rumore morfologico e distorsioni locali.

**Principali operazioni:**

1. **Smoothing direzionale:**  
   il campo angolare viene regolarizzato tramite media vettoriale pesata, riducendo le discontinuità nelle aree di curvatura elevata.

2. **Eliminazione delle minutiae spurie ai bordi:**  
   vengono scartati i punti con distanza dal bordo inferiore a una soglia (tipicamente 10–15 px), per evitare falsi positivi causati da tagli o incompleta acquisizione.

3. **Filtraggio morfologico di coerenza:**  
   ogni minutia viene verificata rispetto alla sua **densità locale** di ridge e direzione dominante.  
   Se la deviazione angolare \(\Delta \theta > 45^\circ\) o la densità differisce troppo, la minutia è scartata.

4. **Raggruppamento (clustering spaziale):**  
   tramite algoritmo DBSCAN, minutiae troppo vicine e con direzioni compatibili vengono fuse in un singolo punto rappresentativo.

L’output di questa fase è un set \( M' \subseteq M \) di minutiae **validate e filtrate**, adatte alla successiva fase di codifica.

---

### 4.8 Generazione del Feature Vector

Le minutiae validate vengono convertite in un **feature vector compatto e normalizzato**, utile per analisi statistiche, classificazione o matching.

**Procedura di codifica:**

1. **Allineamento geometrico:**  
   - Calcolo del baricentro \((x_c, y_c)\) dell’impronta;  
   - Traslazione delle coordinate rispetto al baricentro:  
     ```math
     x'_i = x_i - x_c,\quad y'_i = y_i - y_c
     ```

2. **Normalizzazione della scala e dell’orientamento:**  
   Le coordinate vengono ridimensionate in base alla lunghezza media delle ridge e ruotate secondo la direzione principale \(\bar{\theta}\), ottenendo una rappresentazione **invariante a traslazione, rotazione e scala**.

3. **Codifica vettoriale finale:**
  ```math
   F = [x'_1, y'_1, \theta_1, t_1, x'_2, y'_2, \theta_2, t_2, \dots, x'_N, y'_N, \theta_N, t_N]
  ```
dove:
- \(x'_i, y'_i\) sono le coordinate centrate e normalizzate della minutia \(i\);
- \(\theta_i\) è l’orientamento locale;
- \(t_i\) indica il tipo (`ending` o `bifurcation`).

L’output \(F\) rappresenta la **firma biometrica digitale** dell’impronta, pronta per essere confrontata con altre impronte o utilizzata per classificazione/statistica.

---

### 4.9 Visualizzazione e Overlay delle Minutiae

Per facilitare la verifica qualitativa dei risultati, il sistema genera un overlay grafico delle minutiae sullo scheletro binarizzato:

- **Terminazioni (ending points):** cerchiate in verde;
- **Biforcazioni:** cerchiate in rosso;
- **Orientamento locale:** indicato da una linea breve centrata sulla minutia.

### 4.10 Logging, Tracciabilità e Output Strutturati

Per garantire **riproducibilità, auditabilità e facilità di debug**, ogni fase della pipeline è dotata di un sistema di **log e gestione dei file di output**. Questo consente di monitorare le trasformazioni applicate alle immagini biometriche e di conservare i risultati intermedi.

**Caratteristiche principali del sistema di logging:**

1. **Parametri di elaborazione:**  
   - Dimensione dei blocchi di segmentazione  
   - Soglie di binarizzazione  
   - Parametri dei filtri (Gabor, Gaussiani, ecc.)  
   - Metodi di thinning e validazione minutiae

2. **Metadati e statistiche:**  
   - Timestamp di inizio e fine fase  
   - Numero di minutiae rilevate, filtrate e rimosse  
   - Area biometrica rilevata (ROI)  
   - Eventuali errori o warning

3. **File di output organizzati per fase:**  

| Tipo di file        | Estensione   | Contenuto                                                   |
|--------------------|-------------|------------------------------------------------------------|
| Immagine normalizzata | `.png`      | Risultato del preprocessing con contrasto uniforme        |
| Maschera ROI         | `.png`      | Region of Interest segmentata (foreground)                |
| Immagine binaria     | `.png`      | Ridge binarizzate (0/1)                                   |
| Scheletro (thinning) | `.png`      | Ridge ridotte a spessore unitario                          |
| Mappa orientamenti   | `.png`      | Direzioni locali delle ridge                                |
| Overlay minutiae     | `.png`      | Visualizzazione delle terminazioni e biforcazioni          |
| Feature vector       | `.csv` / `.json` | Coordinate, orientamenti e tipo di ogni minutia        |
| Log di elaborazione  | `.txt`      | Parametri, metriche, note e timestamp                      |

**Vantaggi principali:**

- **Tracciabilità completa:** ogni immagine può essere ricostruita fino alla fase originale.  
- **Facilità di debug e tuning:** parametri modificabili senza perdere la cronologia dei risultati.  
- **Compatibilità con analisi statistiche:** feature numeriche e log strutturati permettono studi quantitativi su precisione, recall e robustezza del sistema.  

In sintesi, il modulo di logging e gestione output trasforma la pipeline in un framework **riproducibile e sperimentalmente verificabile**, fondamentale per la ricerca e lo sviluppo di sistemi biometrici ad alta affidabilità.

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