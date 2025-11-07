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
| **Formato** | TIFF, 8-bit grayscale |
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
- \( \mu_I \), \( \sigma_I \): media e deviazione standard dei valori di intensità dell’immagine originale;
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




>Note di Sviluppo:
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