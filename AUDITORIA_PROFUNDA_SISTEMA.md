# 🔍 Auditoría Técnica, Matemática y Clínica Profunda del Sistema de Detección No Invasiva de Arritmias Cardíacas Fetales

**Proyecto:** `Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva`  
**Fecha de Auditoría:** Noviembre 2026  
**Alcance:** Calidad de datos, limpieza e imputación, análisis multicanal (`CinC 2013`, `NIFEA DB`, `FECGSYNDB`), preprocesamiento DSP, separación ciega de fuentes (`FastICA`), detección `fQRS`, ingeniería e importancia de características (`fHRV`), entrenamiento del modelo, validación cruzada por paciente y matrices de confusión exhaustivas.

---
## 1. Resumen Ejecutivo y Veredicto General

> [!CAUTION]
> **Veredicto de Producción:** Los datos actuales en [`data/dataset_features.csv`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/data/dataset_features.csv) y el flujo de entrenamiento en [`src/model_trainer.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/model_trainer.py) **NO son aptos para uso clínico ni producción en su estado actual**. Aunque el sistema reporta una exactitud (*Accuracy*) aparente de **98.33%** y un `ROC-AUC` de **99.79%**, nuestra auditoría demuestra matemáticamente que dicho resultado es producto de **fuga masiva de datos (*Data Leakage*)** y **evaluación sobre el propio conjunto de entrenamiento (*Train-on-Train*)**. Cuando el modelo se evalúa correctamente en pacientes no vistos mediante `StratifiedGroupKFold` o `Leave-One-Patient-Out` (sin fuga de ventanas ni clones sintéticos), la exactitud real cae a **48.74%** a nivel segmento (`ROC-AUC = 44.95%`, `MCC = -0.0537`) y a **41.67%** a nivel paciente (clasificando erróneamente como arrítmicos a **10 de los 12 fetos sanos**).

### Las 3 Causas Raíz del Problema
1. **Corrupción en la Extracción de Señal (`src/fqrs_detector.py`):**
   - La cancelación materna en [`cancelar_complejos_maternos`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/fqrs_detector.py#L12-L70) multiplica los 4 canales por una ventana de Hanning (`1.0 - 0.85 * hanning`) de $\pm 45\text{ ms}$. Esto **borra los latidos fetales (`fQRS`) que coinciden temporalmente con el QRS materno** y deja un 15% del pico materno más bordes abruptos de modulación.
   - El selector de componente en [`aislar_componente_fecg`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/fqrs_detector.py#L73-L128) elige la componente `FastICA` maximizando `kurtosis * (1 + 2.5 * potencia_1.8_3.5Hz)`. Como los picos maternos residuales son más escasos y afilados (mayor curtosis) y su 2.º armónico cae exactamente entre $2.0\text{ y }3.0\text{ Hz}$, **el algoritmo selecciona el ECG materno o ruido en más del 40% de los casos** (`BPM_mean < 105 bpm` en el **38.7%** de los segmentos sanos y **43.6%** de los segmentos con arritmia).
2. **Destrucción de la Separabilidad Clínica en `dataset_features.csv`:**
   - Debido a los errores del detector `fQRS`, las distribuciones de las 14 características en [`data/dataset_features.csv`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/data/dataset_features.csv) son **estadísticamente indistinguibles** entre fetos sanos (`Target=0`) y con arritmia (`Target=1`): p. ej., `BPM_mean` es **123.26 bpm** en sanos vs **121.98 bpm** en arritmia (`AUC = 0.5127`); `SDNN` es **73.07 ms** en sanos vs **77.93 ms** en arritmia (`AUC = 0.5089`); `RMSSD` es **87.19 ms** vs **86.96 ms** (`AUC = 0.5037`). Ninguna de las 14 variables supera un `AUC` univariado de **0.57**.
3. **Triple Fuga de Datos (*Data Leakage*) en el Entrenamiento (`src/model_trainer.py` y `src/generar_superdataset.py`):**
   - De las **1,850 filas** del CSV, **650 filas (`CinC_2013`) se descartan silenciosamente** en [`cargar_dataset`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/model_trainer.py#L22-L44).
   - De las **1,200 filas** restantes (`NIFEA_DB`), solo **357 son segmentos reales** (provenientes de 24 pacientes con ventanas solapadas al 50% más el registro completo) y **843 filas (70.25%) son clones sintéticos (`_aug`)** donde 5 de las 14 características ni siquiera fueron modificadas.
   - En [`entrenar_modelo_calibrado`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/model_trainer.py#L47-L115), el parámetro `groups` (ID de paciente) se recibe pero **jamás se utiliza**, y las métricas finales se calculan ejecutando `modelo_calibrado.predict_proba(X_scaled)` **sobre la misma matriz `X_scaled` con la que se entrenó el modelo**.

---

## 2. Auditoría de Calidad de Datos, Limpieza e Imputación

### 2.1 Radiografía Exacta de `data/dataset_features.csv`
El archivo [`data/dataset_features.csv`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/data/dataset_features.csv) contiene **1,850 filas** y **17 columnas** (14 características numéricas + `Registro`, `Dataset`, `Target`). Su composición real es la siguiente:

| Fuente (`Dataset`) | Tipo de Fila | Sanos (`Target=0`) | Arritmia (`Target=1`) | Total Filas | Pacientes Únicos | ¿Se usa en `model_trainer.py`? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`NIFEA_DB`** | Segmentos Reales | 155 | 202 | **357** | 24 (`12 NR`, `12 ARR`) | ✅ Sí |
| **`NIFEA_DB`** | Clones Sintéticos (`_aug`) | 347 | 496 | **843** | *(Clones de los 24)* | ✅ Sí (Fuga de datos) |
| **`CinC_2013`** | Segmentos Reales | 204 | 0 | **204** | 68 (de 75 en Set-A) | ❌ **No (Descartado)** |
| **`CinC_2013`** | Clones Sintéticos (`_aug`) | 446 | 0 | **446** | *(Clones de los 68)* | ❌ **No (Descartado)** |
| **TOTAL CSV** | **Reales (561) + Sintéticos (1,289)** | **1,152** | **698** | **1,850** | **92 pacientes** | Solo se usan 1,200 (`NIFEA_DB`) |

#### Hallazgos Críticos en la Construcción del Dataset:
1. **Descarte silencioso del 100% de `CinC_2013` durante el entrenamiento:**
   En [`src/model_trainer.py` (líneas 33–35)](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/model_trainer.py#L33-L35):
   ```python
   nifea_mask = df["Dataset"] == "NIFEA_DB"
   if nifea_mask.sum() >= 200:
       df = df[nifea_mask].copy()
   ```
   Como `NIFEA_DB` tiene 1,200 filas ($\ge 200$), las **650 filas de `CinC_2013` jamás entran al entrenamiento**.
2. **Pacientes faltantes en `NIFEA_DB` (`NR_13` y `NR_14`) y en `CinC_2013` (7 pacientes):**
   - Aunque [`src/build_dataset.py` (línea 19)](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/build_dataset.py#L19) itera sobre `NR_01` a `NR_14` (14 controles sanos), en `dataset_features.csv` **solo existen `NR_01` a `NR_12`**. Los registros `NR_13` y `NR_14` fallaron o no produjeron ventanas válidas y fueron omitidos silenciosamente. Asimismo, `data/nifea_raw` está vacío en disco (0 archivos) porque `build_dataset.py` descargó en *streaming* volátil sin persistir los `.dat`/`.hea` locales.
   - En `CinC_2013 Set-A` (75 pacientes reales en `data/cinc2013_real/set-a`), **7 pacientes no generaron ningún segmento válido** (solo hay 68 pacientes únicos en el CSV).
3. **Doble Conteo y Solapamiento del 50% por Paciente (`src/build_dataset.py`):**
   - En [`procesar_registro_multiventana` (líneas 37–60)](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/build_dataset.py#L37-L60), para cada paciente se inserta primero **una fila con el registro completo** (p. ej., 10 a 20 minutos enteros con hasta 3,065 latidos) y inmediatamente después se insertan **ventanas deslizantes de 90 segundos con paso de 45 segundos (50% de solapamiento)** extraídas exactamente de la misma señal.
   - Además, los pacientes tienen duraciones muy distintas, lo que genera un fuerte sesgo de representación por individuo: el paciente `ARR_06` aporta **42 segmentos reales** y `ARR_08` aporta **29**, mientras que `NR_12` aporta solo **9** y `ARR_05` aporta **10**.

---

### 2.2 Auditoría de Limpieza, Imputación de `NaN` y Valores Cero
Se auditaron tanto la imputación en tiempo de señal ([`imputar_nans_multicanal`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/preprocessing.py#L46-L75)) como la imputación tabular en [`model_trainer.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/model_trainer.py#L41):

1. **Imputación de Señal Cruda (`imputar_nans_multicanal` en `src/preprocessing.py`):**
   - **Cómo funciona:** Detecta `NaN` por canal y aplica interpolación lineal (`np.interp`) entre las muestras válidas adyacentes, rellenando los extremos con el primer/último valor válido (o ceros si todo el canal es `NaN`).
   - **Problema detectado:** En los 75 registros de `CinC 2013 Set-A` verificados en disco hay **0 `NaNs`**, pero en registros clínicos reales cuando ocurre una desconexión de electrodo de varios segundos (tramo largo de `NaN` o saturación plana), interpolar una rampa recta de varios segundos introduce una deriva artificial y silencia el canal sin advertir al algoritmo `FastICA`, contaminando la matriz de covarianza. Debe acotarse la interpolación a huecos cortos ($\le 200\text{ ms}$) y enmascarar tramos largos.
2. **Imputación Tabular (`X.fillna(X.median())` en `src/model_trainer.py` línea 41):**
   - En `dataset_features.csv` no hay celdas con `NaN` ni `Inf`, pero la mediana se calcula **antes de dividir entre entrenamiento y validación** (`X = X.fillna(X.median())`), lo cual constituye fuga estadística si hubiera valores faltantes.
3. **Inflación Masiva de Ceros en `VLF` (565 filas = 30.54% de todo el dataset con `VLF == 0.0`):**
   - Nuestra auditoría de valores exactos en `dataset_features.csv` reveló que la característica `VLF` (*Very Low Frequency power*, $0.0033\text{–}0.04\text{ Hz}$) vale exactamente **`0.0` en 565 de las 1,850 filas (30.54%)**, y `pNN50` vale `0.0` en 17 filas.
   - **Causa matemática exacta:** En [`calcular_frecuencia_welch` (`src/feature_extraction.py` línea 117)](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/feature_extraction.py#L117), el tacograma se interpola a $f_s = 4.0\text{ Hz}$ y se calcula Welch con `nperseg = min(len(rr_detrend), int(4.0 * 64))`. Para las ventanas de 35 s y 60 s (`CinC 2013`), la longitud es $N \approx 140\text{ a }240$ muestras, por lo que la resolución en frecuencia es $\Delta f = 4.0 / N \approx 0.0167\text{ a }0.0285\text{ Hz}$. Como la banda `VLF` termina en $0.04\text{ Hz}$, ¡solo existe **un único punto espectral** (p. ej. $f_1 = 0.0167\text{ Hz}$ o $0.028\text{ Hz}$) dentro del intervalo $[0.0033, 0.04)\text{ Hz}$! Al tener un solo punto, la integración trapezoidal `trapezoid(psd[idx_vlf], freqs[idx_vlf])` evalúa un intervalo de longitud cero y **devuelve exactamente `0.0`**. Fisiológicamente, además, la banda `VLF` requiere grabaciones de al menos 5 minutos ($300\text{ s}$) para ser estimada con validez clínica.

---

### 2.3 Corrupción Matemática en la Aumentación de Datos (`src/generar_superdataset.py`)
El script [`src/generar_superdataset.py` (líneas 74–113)](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/generar_superdataset.py#L74-L113) generó **1,289 filas sintéticas (`_aug`)** a partir de las **561 filas reales** (multiplicando el dataset por $3.3\times$). La función [`aplicar_aumento_fisiologico`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/generar_superdataset.py#L74-L113) presenta **tres fallos graves**:

1. **5 de las 14 características jamás fueron perturbadas (Clones Idénticos):**
   El código añade ruido gaussiano a 9 columnas (`BPM_mean`, `SDNN`, `RMSSD`, `SD1`, `SD2`, `SampEn`, `VLF`, `LF`, `HF`), pero **olvida modificar 5 columnas**:
   - `pNN50`
   - `LF_HF_ratio`
   - `SD1_SD2_ratio`
   - `DFA_alpha1`
   - `Num_Latidos_Validos`
   
   Como consecuencia, verificamos en `dataset_features.csv` que **1,783 de las 1,850 filas (96.4%) son duplicados exactos** en esas 5 columnas respecto a su segmento original. Un árbol de decisión (`GradientBoosting` o `RandomForest`) identifica instantáneamente a qué paciente y ventana pertenece cada fila `_aug` leyendo el par exacto `(pNN50, DFA_alpha1, Num_Latidos_Validos)`.
2. **Inconsistencia Algebraica Interna:**
   Al perturbar independientemente `SD1` y `SD2` sin recalcular `SD1_SD2_ratio`, en todas las filas `_aug` se rompe la identidad matemática $\text{SD1\_SD2\_ratio} = \text{SD1} / \text{SD2}$. Lo mismo ocurre con $\text{LF\_HF\_ratio} \ne \text{LF} / \text{HF}$ cuando `LF` o `HF` son recortados por `max(0.0, ...)` y con la relación física $\text{SD1} = \text{RMSSD} / \sqrt{2}$.
3. **Generación Sintética Antes del Split de Validación:**
   Cualquier aumentación de datos debe ocurrir **exclusivamente a nivel de señal o tacograma y únicamente dentro del fold de entrenamiento**, jamás en el CSV global previo a la validación.

---

## 3. Auditoría Profunda de los Canales por Base de Datos

El usuario solicitó revisar profundamente **qué valores se toman en cada canal de cada base de datos, por qué se escogen unos canales y otros no, y si las características de todos los canales sirven, son iguales o cambian**.

### 3.1 ¿Qué canales tiene cada base de datos y cuáles selecciona el código?

| Base de Datos | Canales Totales en el Archivo Original | Frecuencia ($f_s$) | Qué Canales Selecciona el Sistema y Por Qué | Archivo / Función Responsable |
| :--- | :--- | :---: | :--- | :--- |
| **`CinC Challenge 2013` (Set-A y Set-B)** | **4 canales** (`Abdominal 1`, `Abdominal 2`, `Abdominal 3`, `Abdominal 4`) | $1000\text{ Hz}$ | Toma los **4 canales nativos** porque el estándar del Challenge 2013 se diseñó con 4 derivaciones abdominales maternas. | [`cargar_cinc2013`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/data_loader.py#L12-L37) / [`build_dataset_full.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/build_dataset_full.py) |
| **`NIFEA DB` (Clínica Real de Arritmias)** | **5 o 6 canales** (`1 canal torácico materno` + `4 o 5 canales abdominales`) | $500\text{ Hz}$ o $1000\text{ Hz}$ | **Descarta el canal torácico** (`'tho'` o `'chest'`) y recorta a los **primeros 4 canales abdominales** (`indices_abd[:4]`) para forzar una matriz de 4 columnas compatible con CinC 2013. Si $f_s = 500\text{ Hz}$, remuestrea a $1000\text{ Hz}$ con `scipy.signal.resample`. | [`cargar_nifeadb`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/data_loader.py#L75-L115) / [`stream_nifeadb`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/data_streamer.py#L52-L86) |
| **`FECGSYNDB` (Simulador Multicanal)** | **34 canales** (`32 abdominales` en malla polar + `2 referencias maternas` en canales 33 y 34) | $250\text{ Hz}$ (o remuestreado) | Toma un subconjunto fijo de **4 canales**: `indices_cruz = [3, 12, 15, 27]` (canales 4, 13, 16 y 28) que forman una **cruz ortogonal abdominal** (superior, inferior, izquierdo, derecho alrededor del ombligo). | [`cargar_fecgsyndb`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/data_loader.py#L40-L72) / [`stream_fecgsyndb`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/data_streamer.py#L27-L46) |

> [!WARNING]
> **Error crítico de diseño al descartar el canal torácico en `NIFEA DB` y el 5.º canal abdominal:**  
> En `NIFEA DB`, el canal torácico contiene el **ECG materno puro (sin fECG)**, que es la referencia ideal para cancelar el QRS materno mediante filtrado adaptativo (LMS/RLS) o sustracción de plantilla sincronizada. En lugar de aprovecharlo para limpiar los canales abdominales, [`cargar_nifeadb`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/data_loader.py#L84-L93) **lo destruye antes del preprocesamiento** y además descarta ciegamente el 5.º canal abdominal (`indices_abd[:4]`) sin evaluar si ese 5.º canal tenía mejor relación señal-ruido (`SNR`) fetal que los primeros cuatro.

---

### 3.2 ¿Los valores y características de los 4 canales son iguales o cambian dentro del mismo paciente?
Ejecutamos un análisis cuantitativo canal por canal sobre los **75 pacientes reales de `CinC 2013 Set-A`**. A continuación se presentan las mediciones reales de desviación estándar de amplitud (`Std_Ch0..3` en $\mu\text{V}$), frecuencia cardíaca extraída (`BPM_Ch0..3`) y variabilidad (`SDNN_Ch0..3` en $\text{ms}$) en cada uno de los 4 canales abdominales para los primeros 10 registros:

| Registro | `Std_Ch0` | `Std_Ch1` | `Std_Ch2` | `Std_Ch3` | `BPM_Ch0` | `BPM_Ch1` | `BPM_Ch2` | `BPM_Ch3` | `SDNN_Ch0` | `SDNN_Ch1` | `SDNN_Ch2` | `SDNN_Ch3` | Diagnóstico Físico por Canal |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`a01`** | 10.2 | 18.6 | 7.8 | 10.8 | 80.5 | 85.3 | 80.5 | 98.2 | 41.8 | 93.7 | 42.0 | 157.8 | Los 4 canales individuales quedan dominados por residuo materno (~80–98 bpm); solo `FastICA Comp 0` aísla el feto (145.9 bpm). |
| **`a02`** | 48.0 | 20.6 | 41.5 | 41.6 | 116.5 | 127.0 | 119.3 | 124.2 | 153.2 | 120.7 | 132.7 | 128.7 | Mezcla severa materno-fetal en los 4 canales (`SDNN > 120 ms`). |
| **`a03`** | 24.2 | 20.0 | 8.6 | 16.6 | **127.2** | **128.0** | 122.4 | **127.6** | **31.7** | **22.6** | 94.3 | **35.1** | **Canales 0, 1 y 3 son limpios (`SDNN 22–35 ms`)**, mientras que el **Canal 2 tiene baja amplitud (`Std=8.6`) y puro ruido (`SDNN=94.3 ms`)**. |
| **`a04`** | 29.0 | 26.8 | 15.6 | 24.4 | 127.1 | **131.6** | **131.6** | **131.1** | 71.0 | **25.8** | **31.9** | **27.9** | **Canales 1, 2 y 3 extraen el feto perfecto (`131.6 bpm, SDNN ~26 ms`)**, pero el **Canal 0 introduce falsos picos (`SDNN=71.0 ms`)**. |
| **`a05`** | 22.9 | 20.2 | 9.9 | 20.4 | 127.1 | **129.0** | **128.9** | **129.0** | 72.5 | **18.0** | **11.1** | **8.2** | **Canal 3 (`SDNN=8.2 ms`) y Canal 2 (`11.1 ms`) son cristalinos**; el **Canal 0 está contaminado (`SDNN=72.5 ms`)**. |
| **`a06`** | 11.4 | 15.8 | 17.5 | 17.0 | 100.6 | 120.2 | 101.3 | 115.5 | 27.1 | 119.3 | 36.5 | 109.4 | **Canales 0 y 2 capturan exclusivamente a la madre (`100.6 bpm`)**; Canales 1 y 3 mezclan madre y feto (`SDNN > 109 ms`). |
| **`a07`** | 20.9 | 16.5 | 16.6 | 20.2 | 119.5 | 127.4 | 123.3 | 120.0 | 145.6 | 123.6 | 125.6 | 132.4 | Ruido de alta amplitud en los 4 canales tras la ventana de Hanning. |
| **`a08`** | 24.4 | 25.4 | 6.8 | 21.8 | **127.2** | **128.0** | 120.3 | **127.6** | **32.7** | **22.6** | 106.4 | **35.1** | **Canal 2 está casi muerto (`Std=6.8`, `SDNN=106.4 ms`)**, mientras que **Canal 1 es excelente (`SDNN=22.6 ms`)**. |
| **`a09`** | 12.1 | 16.4 | 9.5 | 14.3 | 114.2 | **129.2** | 95.3 | 93.8 | 126.9 | 69.2 | 29.5 | 46.8 | **Canales 2 y 3 capturan el pulso materno (`93–95 bpm`)**; solo el **Canal 1 ve al feto (`129.2 bpm`)**. |
| **`a10`** | 19.6 | 18.0 | 10.6 | 12.5 | **127.5** | 114.2 | **128.4** | 120.4 | **31.0** | 92.7 | 65.5 | 101.6 | **Solo el Canal 0 es limpio (`127.5 bpm, SDNN=31.0 ms`)**; Canales 1 y 3 están gravemente distorsionados (`SDNN > 92 ms`). |

#### Conclusiones Físicas y Biomédicas sobre los Canales:
1. **Las características de los 4 canales NO son iguales; cambian drásticamente dentro del mismo paciente:**
   El corazón fetal actúa como un dipolo eléctrico tridimensional cuya orientación cambia según la presentación fetal (cefálica, podálica, transversa) y el dorso fetal. Un electrodo colocado perpendicularmente al eje del vector QRS fetal recibe amplitud casi nula (p. ej., `Canal 2` en `a03` y `a08` con `Std = 6.8–8.6` y `SDNN > 94 ms`), mientras que un electrodo alineado con el eje del dipolo captura un complejo fQRS limpio (`Canal 1` en `a03` y `a08` con `SDNN = 22.6 ms`).
2. **¿Sirven todos los canales?**
   - **Para la separación ciega de fuentes (`FastICA` / `PCA` / `SVD`):** **SÍ sirven los múltiples canales** (siempre que no sean puro ruido de desconexión), porque `FastICA` necesita al menos $M \ge 3\text{–}4$ mezclas linealmente independientes para proyectar y separar el subespacio materno, el subespacio fetal y el ruido muscular.
   - **Para extraer características directamente de un canal fijo:** **NO**. Ningún canal fijo sirve para todos los pacientes (en `a06` y `a09` los canales 0 y 2 miden a la madre; en `a04` y `a05` el canal 0 es ruidoso y el canal 3 es perfecto; en `a10` el canal 0 es perfecto y el canal 3 es ruidoso).

---

### 3.3 Prueba Irrefutable contra Verdad Médica (`a01.fqrs`): Por qué falla el selector de `FastICA`
Evaluamos el registro [`data/cinc2013/set-a/a01`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/data/cinc2013/set-a/a01.fqrs), el cual cuenta con las **145 anotaciones médicas oficiales de referencia (`a01.fqrs`)** marcadas por cardiólogos expertos, usando la tolerancia estándar de competencia de $\pm 50\text{ ms}$:

| Fuente Evaluada en `a01` | Latidos Detectados | Verdaderos Positivos (`TP`) | Falsos Positivos (`FP`) | Falsos Negativos (`FN`) | Sensibilidad (`Se`) | Precisión (`PPV`) | **`F1-Score` vs `.fqrs`** | `BPM_mean` Extraído | `SDNN` (`ms`) | `RMSSD` (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Verdad Médica (`a01.fqrs`)** | **145** | 145 | 0 | 0 | 100.0% | 100.0% | **100.0%** | **146.95** | **43.87** | **16.22** |
| `Post-mQRS Canal 0` | 80 | 26 | 54 | 119 | 17.9% | 32.5% | **23.1%** | 80.5 | 41.8 | 59.1 |
| `Post-mQRS Canal 1` | 84 | 27 | 57 | 118 | 18.6% | 32.1% | **23.6%** | 85.3 | 93.7 | 104.0 |
| `Post-mQRS Canal 2` | 80 | 23 | 57 | 122 | 15.9% | 28.8% | **20.4%** | 80.5 | 42.0 | 59.5 |
| `Post-mQRS Canal 3` | 92 | 36 | 56 | 109 | 24.8% | 39.1% | **30.4%** | 98.2 | 157.8 | 164.8 |
| **`FastICA Componente 0` (Feto Real)** | **135** | **131** | **4** | **14** | **90.3%** | **97.0%** | **93.6% ✅** | **145.90** | **60.60** | **56.20** |
| **`FastICA Componente 1` (Elegido por error ❌)** | **80** | **26** | **54** | **119** | **17.9%** | **32.5%** | **23.1% ❌** | **80.46** | **41.79** | **59.09** |
| `FastICA Componente 2` (Ruido/Mezcla) | 93 | 12 | 81 | 133 | 8.3% | 12.9% | **10.1%** | 95.4 | 141.2 | 134.0 |
| `FastICA Componente 3` (Madre Residual) | 83 | 24 | 59 | 121 | 16.6% | 28.9% | **21.1%** | 82.4 | 68.4 | 91.7 |

> [!IMPORTANT]
> **Hallazgo clave en la tabla anterior:**  
> En `a01`, ningún canal individual podía detectar al feto directamente (`F1` entre `20.4%` y `30.4%`), pero **`FastICA` logró separar exitosamente el ECG fetal en la `Componente 0` con un `F1-Score` de `93.6%` (`131/145` latidos reales, `BPM = 145.9`)**.  
> Sin embargo, la función [`aislar_componente_fecg` (`src/fqrs_detector.py` líneas 96–119)](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/fqrs_detector.py#L96-L119) **descartó la `Componente 0` y seleccionó la `Componente 1` (`idx_fetal = 1`)**, que era el ECG materno de `80.46 bpm` (`F1 = 23.1%`)!  
> Esto provocó que en `a01` el sistema reportara un `BPM_mean` falso de **80.46 bpm** (en vez de **146.95 bpm**), inflara `pNN50` de **0.70%** a **43.59%** ($62\times$ mayor) y `SD1` de **11.51 ms** a **42.05 ms**.

---

## 4. Auditoría del Preprocesamiento y Detector `fQRS` (`preprocessing.py` y `fqrs_detector.py`)

### 4.1 Revisión de [`src/preprocessing.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/preprocessing.py)
1. **Eliminación de línea base (`remover_deriva_linea_base`, líneas 24–36):**
   - Aplica un doble filtro de mediana en cascada de $200\text{ ms}$ y $600\text{ ms}$ (`medfilt`). Es un método clásico para ECG adulto, pero computacionalmente muy lento en señales a $1000\text{ Hz}$ ($O(N \cdot K)$) y redundantemente solapado con el filtro pasa-banda posterior.
2. **Filtro Pasa-Banda (`filtro_butter_pasa_banda`, líneas 7–12):**
   - Usa una banda de **$1.0\text{ a }45.0\text{ Hz}$** (Butterworth orden 3).
   - **Problema biomédico:** Entre $1.0\text{ y }8.0\text{ Hz}$ reside toda la energía de la **onda P y onda T materna**, así como artefactos de movimiento fetal y contracción uterina (EHG). El complejo QRS fetal (`fQRS`) es mucho más estrecho ($20\text{–}40\text{ ms}$) que el QRS adulto, por lo que su energía principal se concentra entre **$10\text{ y }40\text{ Hz}$** (o mínimo $3\text{–}45\text{ Hz}$ para separación de plantilla y $10\text{–}40\text{ Hz}$ para detección de picos). Dejar pasar de $1.0\text{ a }8.0\text{ Hz}$ contamina `FastICA` con ondas T maternas gigantescas.
3. **Filtro Notch (`filtro_notch`, líneas 15–21):**
   - Solo filtra **$50.0\text{ Hz}$**, ignorando redes eléctricas de **$60.0\text{ Hz}$** (América/Colombia/EE. UU., aunque el pasa-bajo a $45\text{ Hz}$ atenúa parte de $60\text{ Hz}$, no rechaza su alias o subarmónicos si no se aplica un notch adaptativo 50/60 Hz).

### 4.2 Revisión de [`src/fqrs_detector.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/fqrs_detector.py)
1. **Fallo en la detección de picos maternos (líneas 24–39):**
   - Utiliza un umbral estático en el percentil 88 (`umbral_m = np.percentile(sig_m_integ, 88)`) con `distance = 0.42 * fs` ($420\text{ ms} \equiv 142\text{ bpm}$). En el caso de `a01` o cuando se pasa `return_intermediates=True`, además hay un bug en la línea 208 (`return picos_fetales, senal_fetal, fuentes_ica, idx_fetal, senal_sin_madre` devuelve la matriz `senal_sin_madre` de longitud 60,000 en el quinto elemento en lugar de `picos_maternos`).
2. **Fallo destructivo en la cancelación materna por ventana de Hanning (líneas 44–68):**
   - En lugar de restar la **plantilla morfológica del complejo P-QRS-T materno** (mediante SVD o *Template Subtraction* escalado por mínimos cuadrados en cada canal), el código multiplica todos los canales por `1.0 - 0.85 * np.hanning(len_mask)` durante $\pm 45\text{ ms}$ alrededor del pico materno.
   - **Consecuencia 1:** Cada vez que un latido fetal ocurre dentro de los $90\text{ ms}$ del QRS materno (lo cual sucede fisiológicamente cada 3 a 5 segundos por batimiento de frecuencias entre ~80 bpm maternos y ~140 bpm fetales), **el latido fetal es destruido (atenuado al 15%)**, generando un Falso Negativo (`FN`) sistemático.
   - **Consecuencia 2:** La onda T materna (ubicada $150\text{–}300\text{ ms}$ después del pico R materno) queda **100% intacta**, y el 15% restante del pico R materno sigue teniendo amplitud comparable al QRS fetal.
3. **Fallo en la selección de componente `FastICA` (líneas 96–119):**
   - La puntuación `score = max(0.1, kurt) * (1.0 + 2.5 * potencia_fetal)` en la banda $1.8\text{–}3.5\text{ Hz}$ falla porque:
     1. El 2.º armónico del ritmo cardíaco materno ($80\text{ bpm} = 1.33\text{ Hz} \implies 2 f_m = 2.67\text{ Hz}$) cae exactamente en el centro de la banda $1.8\text{–}3.5\text{ Hz}$.
     2. La curtosis (`scipy.stats.kurtosis`) es más alta para trenes de impulsos de menor frecuencia (80 latidos maternos/minuto o artefactos esporádicos) que para trenes de 145 latidos fetales/minuto.
   - **Cómo debe hacerse correctamente:** Evaluar las 4 componentes de `FastICA` + los 4 canales post-sustracción ejecutando el detector de picos candidatos en cada uno y eligiendo el candidato que maximice un **Índice de Calidad de Señal de Latidos (`bSQI` / regularidad fisiológica del tacograma)**: frecuencia media dentro del rango fetal ($110\text{–}180\text{ bpm}$), mínima proporción de intervalos RR sucesivos con saltos $>20\%$, y alta autocorrelación periódica.

---

## 5. Auditoría de Características (`feature_extraction.py`) e Importancia de Variables

### 5.1 Las 14 Características Creadas y su Auditoría Estadística Real
Evaluamos las 14 características extraídas en [`src/feature_extraction.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/feature_extraction.py) sobre los **357 segmentos reales de `NIFEA_DB`** (155 Sanos vs 202 Arritmia) y medimos:
1. **Importancia Gini (`Gini_Importance_%`)** asignada por el `GradientBoostingClassifier` entrenado actualmente.
2. **Poder discriminativo real univariado (`Univariate_AUC_Real`)** (donde $0.50 =$ azar puro y $1.00 =$ separación perfecta).
3. **Correlación lineal de Pearson con `Target` (`Pearson_Corr_Real`)**.
4. **Media real en Fetos Sanos (`Target=0`) vs Fetos con Arritmia (`Target=1`)**.

| Rango | Característica | Dominio | `Gini_Importance_%` (Modelo Actual) | `Univariate_AUC_Real` (0.50 = Azar) | `Pearson_Corr_Real` | Media en Sanos (`Target=0`, $n=155$) | Media en Arritmia (`Target=1`, $n=202$) | Diagnóstico Técnico de la Característica |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`SampEn`** | No Lineal | **18.25%** | 0.5699 | -0.1046 | 0.3712 | 0.3358 | Mayor importancia en el árbol, pero invertida (menor entropía en arritmia por artefactos en sanos) y débil (`AUC = 0.57`). |
| **2** | **`BPM_mean`** | Temporal | **10.96%** | 0.5127 | -0.0197 | 123.26 bpm | 121.98 bpm | **Contaminada por pulso materno** (<105 bpm en ~41% de las filas). Separabilidad real nula (`AUC = 0.51`). |
| **3** | **`Num_Latidos_Validos`** | Metadato | **10.47%** | 0.5473 | +0.0684 | 316.05 latidos | 396.32 latidos | 🚨 **FUGA DE METADATO:** Depende de la duración del archivo (ventana completa de 20 min vs 90 s), NO es una variable fisiológica. |
| **4** | **`SD1_SD2_ratio`** | No Lineal (Poincaré) | **8.17%** | 0.5313 | -0.0972 | 0.8758 | 0.8364 | No modificada en `_aug` (clonada idéntica). |
| **5** | **`pNN50`** | Temporal | **7.76%** | 0.5303 | -0.0258 | 32.47% | 31.39% | Inflada artificialmente (>30% en sanos cuando en fetos sanos reales es <5%). No modificada en `_aug`. |
| **6** | **`SDNN`** | Temporal | **7.59%** | 0.5089 | +0.0531 | 73.07 ms | 77.93 ms | Inflada por pérdida de latidos durante mQRS (`AUC = 0.51`). |
| **7** | **`DFA_alpha1`** | No Lineal (Fractal) | **6.99%** | 0.5327 | +0.0913 | 0.4648 | 0.5086 | No modificada en `_aug` (clonada idéntica). |
| **8** | **`VLF`** | Frecuencial | **6.10%** | 0.5389 | +0.1066 | 177.26 $\text{ms}^2$ | 258.59 $\text{ms}^2$ | 🚨 **30.5% de las filas tienen `VLF = 0.0`** por resolución insuficiente de Welch en ventanas $\le 60\text{ s}$. |
| **9** | **`LF_HF_ratio`** | Frecuencial | **5.98%** | 0.5396 | +0.0561 | 0.8886 | 0.9963 | No modificada en `_aug` (clonada idéntica). |
| **10** | **`SD2`** | No Lineal (Poincaré) | **5.71%** | 0.5174 | +0.0725 | 78.66 ms | 85.94 ms | `AUC = 0.52` (indistinguible entre clases). |
| **11** | **`HF`** | Frecuencial | **5.35%** | 0.5180 | -0.0241 | 1906.84 $\text{ms}^2$ | 1824.21 $\text{ms}^2$ | Dominada por saltos abruptos de falsos positivos/negativos de `fQRS`. |
| **12** | **`LF`** | Frecuencial | **3.02%** | 0.5364 | +0.0170 | 1277.67 $\text{ms}^2$ | 1317.85 $\text{ms}^2$ | `AUC = 0.54`. |
| **13** | **`RMSSD`** | Temporal | **1.91%** | 0.5037 | -0.0021 | 87.19 ms | 86.96 ms | **Idéntica en sanos (`87.19 ms`) y arritmia (`86.96 ms`)**, `AUC = 0.5037`. |
| **14** | **`SD1`** | No Lineal (Poincaré) | **1.75%** | 0.5035 | -0.0014 | 61.84 ms | 61.70 ms | **Redundancia matemática exacta:** $\text{SD1} \equiv \text{RMSSD} / \sqrt{2}$ (correlación $r = 1.000$ en datos reales). |

### 5.2 ¿Por qué el tacograma en `feature_extraction.py` no filtra los errores del detector?
En [`filtrar_tacograma_fisiologico` (`src/feature_extraction.py` líneas 7–40)](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/feature_extraction.py#L7-L40):
- La línea 21 acepta cualquier intervalo RR entre **`230.0 ms` y `860.0 ms`** ($69.8\text{ a }260.9\text{ bpm}$).
- **Dos errores críticos:**
  1. $69.8\text{ a }100\text{ bpm}$ ($600\text{ a }860\text{ ms}$) es el rango normal del **corazón materno adulto en reposo**, lo que permite que un tacograma 100% materno pase como "fisiológicamente válido".
  2. Cuando el feto late a $143\text{ bpm}$ ($RR = 420\text{ ms}$) y la ventana de Hanning materna borra **un latido fetal**, el intervalo entre el latido anterior y el siguiente es $2 \times 420\text{ ms} = 840\text{ ms}$. Como $840\text{ ms} \le 860\text{ ms}$, ¡el filtro **lo acepta como un intervalo RR válido**! Y cuando se pierden varios latidos seguidos por coincidencia materna, la mediana local de 7 latidos (`med_local`) se desplaza, dejando saltos de $420 \to 840 \to 420\text{ ms}$ que inflan `RMSSD`, `SDNN`, `SD1` y `pNN50` a valores patológicos en fetos completamente sanos.
- Por otro lado, si en lugar de un latido perdido hay una **extrasístole real (PAC/PVC)** con pausa compensatoria y el filtro de mediana (`abs(r - med_local) / med_local > 0.50`) la reemplaza por `med_local`, **borra la arritmia real** si no se contabilizan antes las métricas de ectopia (`% de latidos ectópicos`, `acoplamiento prematuro`, `pausa compensatoria`).

---

## 6. Auditoría del Entrenamiento del Modelo, Métricas Exhaustivas y Matrices de Confusión

### 6.1 Cómo se está entrenando actualmente el modelo (`src/model_trainer.py`)
En [`src/model_trainer.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/model_trainer.py):
1. Carga `dataset_features.csv`, filtra solo `NIFEA_DB` ($N = 1,200$ filas: 357 reales + 843 `_aug`).
2. Escala las 14 características con `StandardScaler().fit_transform(X)` sobre las 1,200 filas completas.
3. Instancia `GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, min_samples_leaf=3, subsample=0.85, random_state=42)`.
4. Lo envuelve en `CalibratedClassifierCV(estimator=base_estimator, method="isotonic", cv=5)` y llama a `modelo_calibrado.fit(X_scaled, y)`.
   - **Nota crítica:** Aunque la línea 7 importa `StratifiedGroupKFold` y la línea 53 recibe `groups = df_train["Registro"].values`, **`groups` NO se pasa en ningún lugar**. El `cv=5` interno de `CalibratedClassifierCV` usa un `StratifiedKFold` estándar sin agrupar por paciente, mezclando las ventanas solapadas y los clones `_aug` del mismo paciente en entrenamiento y validación.
5. Finalmente, en las líneas 82–95 calcula todas las métricas ejecutando:
   ```python
   y_prob = modelo_calibrado.predict_proba(X_scaled)[:, 1]
   y_pred = (y_prob >= 0.50).astype(int)
   ```
   Es decir, **evalúa el modelo sobre los mismos datos `X_scaled` con los que acaba de entrenarlo**.

---

### 6.2 Tabla Maestra de Todas las Métricas y Matrices de Confusión en los 6 Escenarios de Evaluación

Para conocer con absoluta transparencia el rendimiento del sistema, calculamos **todas las métricas posibles** y las **matrices de confusión exactas (`TN`, `FP`, `FN`, `TP`)** en 6 escenarios progresivos:
- **Escenario A:** Evaluación actual de `model_trainer.py` (*Train-on-Train* sobre las 1,200 filas de `NIFEA_DB` con `_aug`, umbral $0.50$ y umbral UI $0.46$).
- **Escenario B:** Modelo actual evaluado *in-sample* únicamente sobre los **357 segmentos reales** de `NIFEA_DB` (sigue siendo *Train-on-Train*).
- **Escenario C:** Validación cruzada 5-Fold (`StratifiedKFold`) *Out-Of-Fold* sobre las 1,200 filas **sin agrupar por paciente** (demuestra cómo los clones `_aug` y ventanas solapadas inflan artificialmente el CV a **92.08%**).
- **Escenario D (VERDADERA VALIDACIÓN POR SEGMENTO):** Validación cruzada 5-Fold agrupada por paciente (`StratifiedGroupKFold`) sobre los **357 segmentos reales** (ningún paciente de entrenamiento aparece en validación; cero clones `_aug`).
- **Escenario E (VERDADERA VALIDACIÓN CLÍNICA `Leave-One-Patient-Out`):** Entrenar con 23 pacientes reales y evaluar en el paciente 24 dejado fuera (repetido para los 24 pacientes de `NIFEA_DB`), reportado tanto **a nivel segmento ($N=357$)** como **a nivel paciente ($N=24$: 12 Sanos vs 12 Arritmias)**.
- **Escenario F (Generalización Externa en `CinC 2013 Set-A`):** Evaluar el modelo guardado en los **204 segmentos reales de `CinC_2013`** (todos etiquetados como `Target=0`).

#### Tabla 1: Matrices de Confusión Exactas (`TN`, `FP`, `FN`, `TP`)

| Escenario de Evaluación | Total ($N$) | Verdaderos Negativos (`TN` - Sanos bien) | Falsos Positivos (`FP` - Falsa Alarma) | Falsos Negativos (`FN` - Arritmia no vista) | Verdaderos Positivos (`TP` - Arritmia detectada) | Tasa de Falsos Positivos (`FPR`) | Tasa de Falsos Negativos (`FNR`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1. Reportado Actual (`Train-on-Train`, $N=1200$, $th=0.50$)** | 1,200 | **492** | **10** | **10** | **688** | 1.99% | 1.43% |
| **A2. Reportado Actual (`Train-on-Train`, $N=1200$, $th=0.46$ UI)** | 1,200 | **492** | **10** | **9** | **689** | 1.99% | 1.29% |
| **B. Segmentos Reales `In-Sample` ($N=357$, $th=0.50$)** | 357 | **149** | **6** | **8** | **194** | 3.87% | 3.96% |
| **C. 5-Fold CV Sin Agrupar Pacientes ($N=1200$, Con Fuga)** | 1,200 | **434** | **68** | **27** | **671** | 13.55% | 3.87% |
| **D. 🚨 5-Fold `StratifiedGroupKFold` Real ($N=357$, Cero Fuga)** | **357** | **57** | **98** | **85** | **117** | **63.23%** | **42.08%** |
| **E1. 🚨 `Leave-One-Patient-Out` Segmentos ($N=357$, Cero Fuga)** | **357** | **53** | **102** | **99** | **103** | **65.81%** | **49.01%** |
| **E2. 🚨 `Leave-One-Patient-Out` Pacientes ($N=24$, Cero Fuga)** | **24** | **2** | **10** | **4** | **8** | **83.33%** | **33.33%** |
| **F. 🚨 Evaluación Externa `CinC 2013 Set-A` ($N=204$ Sanos)** | **204** | **65** | **139** | *0 (N/A)* | *0 (N/A)* | **68.14%** | *N/A* |

#### Tabla 2: Todas las Métricas de Rendimiento Clínico y Estadístico

| Escenario de Evaluación | `Accuracy` (Exactitud) | `Sensitivity` / `Recall` (`TPR`) | `Specificity` (`TNR`) | `Precision` (`PPV`) | `NPV` (Valor Pred. Neg.) | `F1-Score` (Binario) | `F1-Score` (Macro) | `ROC-AUC` | `PR-AUC` | `MCC` (Matthews) | `Brier Score` (Error Calib.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1. `Train-on-Train` ($N=1200, th=0.50$)** | **98.33%** | **98.57%** | **98.01%** | **98.57%** | **98.01%** | **98.57%** | **98.29%** | **99.79%** | **99.85%** | **+0.9658** | **0.0152** |
| **A2. `Train-on-Train` ($N=1200, th=0.46$)** | **98.42%** | **98.71%** | **98.01%** | **98.57%** | **98.20%** | **98.64%** | **98.37%** | **99.79%** | **99.85%** | **+0.9675** | **0.0152** |
| **B. Reales `In-Sample` ($N=357, th=0.50$)** | **96.08%** | **96.04%** | **96.13%** | **97.00%** | **94.90%** | **96.52%** | **96.02%** | **99.28%** | **99.45%** | **+0.9203** | **0.0283** |
| **C. 5-Fold CV Sin Agrupar ($N=1200$)** | **92.08%** | **96.13%** | **86.45%** | **90.80%** | **94.14%** | **93.39%** | **91.76%** | **97.61%** | **98.18%** | **+0.8375** | **0.0813** |
| **D. 🚨 5-Fold `GroupKFold` Real ($N=357$)** | **48.74%** | **57.92%** | **36.77%** | **54.42%** | **40.14%** | **56.12%** | **47.25%** | **44.95%** | **52.59%** | **-0.0537** | **0.3343** |
| **E1. 🚨 `LOPO` Segmentos Reales ($N=357$)** | **43.70%** | **50.99%** | **34.19%** | **50.24%** | **34.87%** | **50.61%** | **42.57%** | **37.52%** | **48.43%** | **-0.1485** | **0.3679** |
| **E2. 🚨 `LOPO` Nivel Paciente ($N=24$)** | **41.67%** | **66.67%** | **16.67%** | **44.44%** | **33.33%** | **53.33%** | **37.78%** | **24.31%** | **40.92%** | **-0.1768** | **0.3137** |
| **F. 🚨 `CinC 2013 Set-A` ($N=204$ Sanos)** | **31.86%** | *N/A* | **31.86%** | *0.00%* | **100.0%** | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | **0.5142** |

### 6.3 Interpretación de la Matriz de Confusión Real
1. **Por qué cae de 98.33% (Escenario A) a 48.74% (Escenario D) y 41.67% (Escenario E2):**
   - Como demostramos en la Sección 5.1, las 14 características tienen un `AUC` real de ~0.51–0.57 porque el detector `fQRS` captura el pulso materno o ruido en ambos grupos (`NR` y `ARR`).
   - Al no existir diferencia real en las características entre sanos y enfermos en el CSV actual, el `GradientBoostingClassifier` de profundidad 4 simplemente **memorizó la firma individual de cada uno de los 24 pacientes** (su `Num_Latidos_Validos`, su `pNN50` congelado y su `SampEn` basal).
   - Cuando se evalúa sobre el mismo paciente (Escenarios A, B y C), acierta el **92%–98%**. Pero cuando se evalúa sobre un **paciente nuevo que el modelo nunca ha visto** (Escenarios D, E y F), el modelo tiene una tasa de **Falsos Positivos del 63.2% al 83.3%**:
     - En `Leave-One-Patient-Out` a nivel paciente (Escenario E2), de los **12 bebés sanos (`NR_01` a `NR_12`)**, el modelo etiqueta a **10 como Arritmia (`FP = 10`, `TN = 2`, Especificidad = 16.67%)**.
     - En `CinC 2013 Set-A` (Escenario F, 204 segmentos de control), el modelo etiqueta a **139 segmentos como Arritmia (`FP = 139`, `TN = 65`, Especificidad = 31.86%)**.

---

### 6.4 Auditoría de la Validación Circular en `SET_B_ETIQUETAS_CLINICAS.md` (`src/procesar_set_b.py`)
El archivo [`SET_B_ETIQUETAS_CLINICAS.md`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/SET_B_ETIQUETAS_CLINICAS.md), generado por [`src/procesar_set_b.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/procesar_set_b.py), afirma haber validado los 95 registros de `CinC 2013 Set-B` obteniendo 92 fetos con arritmia y 3 sanos. Nuestra auditoría identificó una **circularidad metodológica grave**:
1. **`CinC 2013 Set-B` NO tiene etiquetas clínicas de arritmia en PhysioNet:** Es el conjunto de prueba ciego del Challenge 2013 para detección de complejos `fQRS`.
2. **Etiquetas autogeneradas sobre ruido del detector:** La función [`diagnosticar_reglas_cardiologicas` (`src/procesar_set_b.py` líneas 26–59)](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/procesar_set_b.py#L26-L59) aplicó reglas fijas (`BPM < 110` $\to$ Bradicardia; `SD1 > 50` o `pNN50 > 45` $\to$ Arritmia / Ectopia) sobre los picos ruidosos producidos por [`extraer_fqrs_optimo`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/fqrs_detector.py#L168-L208).
3. Como el detector captura el pulso materno (`< 110 bpm`) o genera falsos picos (`SD1 > 50 ms`) en casi todos los registros de 60 s, el script inventó que **92 de los 95 fetos (96.8%) tenían arritmia severa** y luego comparó las predicciones del modelo (que también da ~70–85% de falsos positivos en registros de 60 s) contra esas mismas reglas autogeneradas.

---

## 7. Lista Consolidada de Todo lo que Hay que Cambiar

| # | Módulo / Archivo | Hallazgo Exacto | Cambio Obligatorio | Prioridad |
| :---: | :--- | :--- | :--- | :---: |
| **1** | [`src/data_loader.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/data_loader.py) y [`src/build_dataset.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/build_dataset.py) | `NIFEA DB` descarta el canal torácico materno, trunca a `indices_abd[:4]`, remuestrea con FFT (`signal.resample`), no guarda los archivos crudos en `data/nifea_raw`, y pierde `NR_13` y `NR_14`. | Persistir en disco los **26 registros completos de `NIFEA DB` (`14 NR` + `12 ARR`)**, conservar el canal torácico como referencia materna explícita para cancelación adaptativa/SVD, evaluar calidad en **todos** los canales abdominales disponibles (4 o 5), y usar `scipy.signal.resample_poly` para evitar *ringing* de Gibbs. | 🔴 Crítica |
| **2** | [`src/preprocessing.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/preprocessing.py) | El pasa-banda de `1–45 Hz` deja pasar la onda P y T materna (`1–8 Hz`); el Notch solo filtra `50 Hz`; `imputar_nans_multicanal` interpola tramos largos sin límite. | Separar el preprocesamiento en dos ramas: `1–45 Hz` para estimación de plantilla materna y **`8–40 Hz` (o `10–45 Hz`)** para realce de `fQRS` antes de `FastICA`/detección de picos; detectar automáticamente red de `50 Hz` vs `60 Hz`; limitar interpolación de `NaN` a huecos $\le 200\text{ ms}$. | 🔴 Crítica |
| **3** | [`src/fqrs_detector.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/fqrs_detector.py) (`cancelar_complejos_maternos`) | La ventana multiplicativa `1.0 - 0.85 * hanning` ($\pm 45\text{ ms}$) borra los latidos fetales coincidentes con el QRS materno y deja intacta la onda T materna y el 15% del pico R materno. | Reemplazar la ventana de Hanning por **Sustracción de Plantilla Materna Adaptativa (`TS-PCA` / `SVD`)** sobre la ventana completa del complejo materno ($[-200\text{ ms}, +400\text{ ms}]$ alrededor de cada mQRS), preservando intacto el complejo `fQRS` superpuesto. Corregir además el bug de retorno en la línea 208. | 🔴 Crítica |
| **4** | [`src/fqrs_detector.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/fqrs_detector.py) (`aislar_componente_fecg`) | El selector `kurt * (1 + 2.5 * potencia_1.8_3.5Hz)` elige el residuo materno o ruido en `>40%` de los casos (como se demostró en `a01`, donde eligió `IC 1` con F1=23.1% en lugar de `IC 0` con F1=93.6%). | Implementar un **Selector Multicanal + Multi-ICA basado en Calidad de Latidos (`bSQI` y Regularidad R-R)**: extraer trenes de picos candidatos de las 4 componentes `FastICA` y de los canales abdominales post-TS-PCA, descartar candidatos con frecuencia materna ($<105\text{ bpm}$ si hay otro candidato en $115\text{–}175\text{ bpm}$) y elegir el canal/componente con mayor `bSQI` y periodicidad fisiológica. | 🔴 Crítica |
| **5** | [`src/feature_extraction.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/feature_extraction.py) | El filtro RR acepta intervalos de hasta `860 ms` (incluyendo latidos fetales perdidos $2 \times 420\text{ ms} = 840\text{ ms}$ y ritmo materno); `VLF` vale `0.0` en el 30.5% del dataset; `SD1` duplica exactamente a `RMSSD`; `Num_Latidos_Validos` es un metadato de duración. | 1) Corregir interpolación de latidos perdidos ($RR \approx 2 \times \text{mediana local}$) dividiendo el intervalo en 2 latidos antes de calcular `SDNN`/`RMSSD`, mientras se registran por separado las métricas específicas de arritmia (**`Ectopic_Beat_Ratio`**, **`Prematurity_Index`**, **`Compensatory_Pause_Ratio`**, **`HRV_Triangular_Index`**, **`Tachogram_Entropy`**). 2) Eliminar `Num_Latidos_Validos`, `VLF` (en ventanas $<300\text{ s}$) y la colinealidad `SD1`/`RMSSD`. | 🔴 Crítica |
| **6** | [`src/build_dataset.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/build_dataset.py) y [`src/generar_superdataset.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/generar_superdataset.py) | Mezcla el registro completo con ventanas de 90 s del mismo paciente; `generar_superdataset.py` crea 1,289 filas `_aug` antes del split dejando 5 columnas congeladas y rompiendo identidades algebraicas. | **Eliminar por completo `generar_superdataset.py` y las filas `_aug` del CSV maestro**. Usar ventanas uniformes (p. ej. 60 s sin mezclar con el registro completo), exigir un umbral mínimo de calidad de señal (`SQI >= 0.75`) para admitir una ventana, y equilibrar el peso por paciente (`sample_weight` inversamente proporcional al número de ventanas del paciente). | 🔴 Crítica |
| **7** | [`src/model_trainer.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/model_trainer.py) | No usa `groups` en `StratifiedGroupKFold`, escala antes del split, y evalúa métricas sobre `X_scaled` (*Train-on-Train*). | Implementar validación cruzada estricta por paciente (**`StratifiedGroupKFold(n_splits=5)`** y **`LeaveOneGroupOut()`** pasando `groups=df["Registro"]`), encapsular `StandardScaler` dentro de un `Pipeline` para que se ajuste solo en el fold de entrenamiento, y reportar métricas **exclusivamente Out-Of-Fold (`OOF`)** tanto a nivel segmento como a nivel paciente. | 🔴 Crítica |
| **8** | [`src/procesar_set_b.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/src/procesar_set_b.py) y [`app.py`](file:///c:/Users/santi/Desktop/deteccion%20de%20aritmias/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva/app.py) | `procesar_set_b.py` inventa etiquetas clínicas sobre señales ruidosas sin verdad médica; `app.py` no advierte cuando el `SQI` es bajo o cuando se detectó el pulso materno. | Eliminar la validación circular de `Set-B`; validar el detector `fQRS` contra las **75 anotaciones médicas `.fqrs` de `CinC 2013 Set-A`** (meta: `F1 >= 90%`) y validar el clasificador de arritmias en **`NIFEA DB` (26 pacientes)** + **`FECGSYNDB` (casos de arritmia simulada `c3`/`Ectopic`)**; añadir un semáforo de Calidad de Señal (`SQI`) en la interfaz `app.py`. | 🟠 Alta |

