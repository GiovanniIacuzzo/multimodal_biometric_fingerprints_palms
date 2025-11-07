<h1 align="center">Multimodal Biometric Identification System</h1>

> _Pipeline completa per l’elaborazione e l’analisi di impronte digitali basata sul dataset **PolyU HRF DBII** (Hong Kong Polytechnic University High Resolution Fingerprint Database II)._  

---

## 🔍 Introduzione

Questo progetto implementa una **pipeline biometrica** per l’elaborazione e l’estrazione di feature da impronte digitali ad alta risoluzione.  
L’obiettivo è fornire un framework sperimentale **robusto, riproducibile e scientificamente trasparente** per l’analisi delle impronte, dalla fase di acquisizione fino all’estrazione delle minutiae.

Ogni fase della pipeline è progettata per affrontare i problemi più comuni nelle immagini biometriche:
- **rumore e contrasto non uniforme**,
- **regioni di background e segmentazione imperfetta**,
- **distorsioni locali e discontinuità delle ridge**.

---

## 🧬 Dataset: PolyU High Resolution Fingerprint Database II (PolyU HFR DBII)

La pipeline è sviluppata e testata sul dataset **PolyU HRF DBII**, una delle più note basi di dati per l’analisi di impronte digitali ad alta risoluzione.

### 📁 Caratteristiche del dataset

- **Origine:** Department of Computing, The Hong Kong Polytechnic University  
- **Nome completo:** High Resolution Fingerprint Database II (DBII)  
- **Numero soggetti:** 148 individui  
- **Numero immagini totali:** 148 × 10 = **1480 impronte**  
- **Risoluzione:** 1200 dpi (pixel spacing ≈ 21 µm)  
- **Formato file:** TIFF a 8-bit grayscale  
- **Dimensione tipica:** 240×320 o superiore  

Ogni soggetto è rappresentato da **10 immagini acquisite in sessioni diverse**, con variazioni di pressione, rotazione, e parziale sovrapposizione.  
Questo rende il dataset ideale per testare algoritmi di **enhancement e robustezza strutturale** delle ridge.

---

## ⚙️ Funzionamento Generale della Pipeline

La pipeline segue una sequenza di fasi ordinate, ciascuna con scopi e trasformazioni specifiche.  
Ogni stadio produce **un output intermedio**, utilizzato come input per il successivo.

### 1️⃣ **Normalizzazione e Preprocessing Iniziale**

#### Obiettivo
Rimuovere variazioni d’intensità e migliorare il contrasto tra ridge e valley.  
Assicurare che ogni immagine presenti un range dinamico coerente prima della segmentazione.

#### Implementazione
- **Normalizzazione lineare:**  
  Ogni pixel `p` è rimappato come:  
  $$
  I_{norm}(x,y) = \frac{I(x,y) - \mu_I}{\sigma_I} \cdot \sigma_0 + \mu_0
  $$
  con valori target ($$\mu_0 = 128, \sigma_0 = 100$$).

- **CLAHE (Contrast Limited Adaptive Histogram Equalization):**  
  Migliora localmente il contrasto mantenendo la continuità tonale.  
  Parametri tipici: `clipLimit=2.0`, `tileGridSize=(8,8)`.

- **Denoising bilaterale e gaussiano:**  
  Combinazione di filtro bilaterale (`cv2.bilateralFilter`) e filtro gaussiano (`gaussian_filter` di SciPy) per preservare i bordi delle ridge.

📤 _Output: immagine normalizzata e denoised._

---

### 2️⃣ **Segmentazione**

#### Obiettivo
Separare la regione di impronta (foreground) dallo sfondo uniforme, riducendo il rumore periferico.

#### Implementazione
- Calcolo della **varianza locale** su blocchi 16×16.
- Thresholding di Otsu applicato alla mappa di varianza.
- Pulizia mediante **operazioni morfologiche** (`closing`, `opening`) e selezione del componente connesso più grande.
- Creazione di una **mask binaria** (foreground = 1).

📤 _Output: immagine segmentata + maschera binaria._

---

### 3️⃣ **Binarizzazione Adaptiva**

#### Obiettivo
Convertire l’immagine in una mappa binaria precisa dove le ridge siano chiaramente separabili.

#### Implementazione
- **Metodo Sauvola (adattivo):**  
  Calcolo del threshold locale $$(T(x,y) = m(x,y) [1 + k(\frac{s(x,y)}{R} - 1)])$$  
  con \(k = 0.3, R = 128\).
- **Metodo Otsu (globale):**  
  Applicato in combinazione per migliorare la robustezza in regioni di contrasto basso.
- Fusione dei due metodi con pesatura adattiva, regolata sulla varianza locale.

📤 _Output: immagine binaria robusta (ridges=1, valleys=0)._

---

### 4️⃣ **Skeletonization (Thinning)**

#### Obiettivo
Ridurre le ridge a una linea di spessore un pixel, mantenendo la topologia originale.

#### Implementazione
- Uso di `skimage.morphology.skeletonize` o metodo Zhang–Suen.  
- Pulizia di residui isolati con `remove_small_objects` e `binary_opening`.
- Verifica topologica per connettività e rimozione di pixel spurii.

📤 _Output: skeleton binario dell’impronta._

---

### 5️⃣ **Calcolo del Campo di Orientamento**

#### Obiettivo
Determinare la direzione dominante delle ridge in ogni regione locale.

#### Implementazione
- Derivate parziali con operatori **Sobel** \(G_x, G_y\).
- Calcolo tensoriale locale:
  $$
  \theta(x,y) = \frac{1}{2}\arctan\left(\frac{2G_xG_y}{G_x^2 - G_y^2}\right)
  $$
- Smoothing mediante filtro gaussiano 2D per garantire coerenza direzionale.
- Visualizzazione tramite mappa vettoriale o overlay colorato.

📤 _Output: mappa di orientamento + immagine visuale._

---

### 6️⃣ **Estrazione delle Minutiae**

#### Obiettivo
Identificare punti caratteristici dell’impronta:
- **Ending points** (terminazioni)
- **Bifurcations** (ramificazioni)

#### Implementazione
- Metodo **Crossing Number (CN)** su finestra 3×3:
  $$
  CN = \frac{1}{2}\sum_{i=1}^8 |P_i - P_{i+1}|
  $$
  - CN = 1 → _ending_
  - CN = 3 → _bifurcation_
- Rimozione duplicati tramite **KD-Tree** spaziale (distanza < 8 px).
- Calcolo dell’**orientamento locale** di ogni minutia con PCA su patch 11×11.
- Assegnazione attributi:
  ```python
  {
      "x": x_coord,
      "y": y_coord,
      "type": "ending" or "bifurcation",
      "theta": orientation_angle
  }
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