import numpy as np
from scipy import signal  # type: ignore
from scipy import stats  # type: ignore
from scipy.integrate import trapezoid  # type: ignore


def _mediana_movil(arr, window=9):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr
    half = window // 2
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.median(arr[max(0, i - half) : min(n, i + half + 1)])
    return out


def calcular_tacograma_rr(picos_indices, fs=1000.0):
    """
    Convierte picos fQRS en un tacograma RR clínico reconstruyendo artefactos instrumentales
    sin destruir eventos arrítmicos genuinos (extrasístoles PAC/PVC, pausas compensatorias,
    bradicardias y taquicardias):
    1. Fusión de falsos positivos aislados (dos sub-intervalos cortos cuya suma equivale a 1 ciclo local).
    2. División de falsos negativos aislados (un intervalo de ~2x la mediana local NO precedido por
       acoplamiento prematuro se divide en dos latidos sinusales).
    3. Preservación de pares ectópicos (latido prematuro seguido de pausa compensatoria) y filtrado
       exclusivo de espigas técnicas aisladas no acopladas.
    """
    if picos_indices is None or len(picos_indices) < 5:
        return np.array([], dtype=np.float64)

    rr_crudo = (np.diff(picos_indices) / float(fs)) * 1000.0
    rr_fis = [float(r) for r in rr_crudo if 210.0 <= r <= 1050.0]
    if len(rr_fis) < 5:
        return np.array([], dtype=np.float64)

    # Paso 1: Fusionar falsos positivos de ruido inter-latido (r_i + r_{i+1} ~= med_local)
    med_init = _mediana_movil(rr_fis, window=9)
    rr_step1 = []
    i = 0
    n_fis = len(rr_fis)
    while i < n_fis:
        m_loc = med_init[i]
        if (
            i + 1 < n_fis
            and rr_fis[i] < 0.68 * m_loc
            and rr_fis[i + 1] < 0.68 * m_loc
            and 0.85 * m_loc <= (rr_fis[i] + rr_fis[i + 1]) <= 1.16 * m_loc
        ):
            rr_step1.append(rr_fis[i] + rr_fis[i + 1])
            i += 2
        else:
            rr_step1.append(rr_fis[i])
            i += 1

    if len(rr_step1) < 5:
        return np.array([], dtype=np.float64)

    # Paso 2: Dividir latidos perdidos aislados (~2x mediana local sin prematuridad previa)
    med_s1 = _mediana_movil(rr_step1, window=9)
    rr_step2 = []
    for idx, r_val in enumerate(rr_step1):
        m_loc = med_s1[idx]
        prev_normal = True
        if idx > 0:
            prev_normal = rr_step1[idx - 1] >= 0.87 * med_s1[idx - 1]
        if prev_normal and (1.76 * m_loc <= r_val <= 2.26 * m_loc):
            rr_step2.append(0.5 * r_val)
            rr_step2.append(0.5 * r_val)
        else:
            rr_step2.append(r_val)

    rr_arr = np.asarray(rr_step2, dtype=np.float64)
    if len(rr_arr) < 5:
        return np.array([], dtype=np.float64)

    # Paso 3: Suprimir únicamente artefactos extremos (>42% de desviación) que NO formen
    # parte de una extrasístole acoplada (prematuro + pausa compensatoria)
    med_s2 = _mediana_movil(rr_arr, window=9)
    rr_limpio = np.copy(rr_arr)
    n_arr = len(rr_arr)

    for idx in range(n_arr):
        m_loc = med_s2[idx]
        r_val = rr_arr[idx]
        desv_rel = abs(r_val - m_loc) / (m_loc + 1e-8)
        if desv_rel <= 0.42:
            continue

        # Verificar si es el latido prematuro de una extrasístole (seguido de pausa >= 0.98 * m_loc)
        es_prematuro_acoplado = (
            idx + 1 < n_arr
            and 0.52 * m_loc <= r_val <= 0.85 * m_loc
            and 0.98 * m_loc <= rr_arr[idx + 1] <= 1.70 * m_loc
        )
        # Verificar si es la pausa compensatoria posterior a un latido prematuro
        es_pausa_compensatoria = (
            idx > 0
            and 0.52 * med_s2[idx - 1] <= rr_arr[idx - 1] <= 0.86 * med_s2[idx - 1]
            and 1.05 * m_loc <= r_val <= 1.70 * m_loc
        )

        if es_prematuro_acoplado or es_pausa_compensatoria:
            rr_limpio[idx] = np.clip(r_val, 0.55 * m_loc, 1.60 * m_loc)
        else:
            rr_limpio[idx] = m_loc

    return rr_limpio


def extraer_features_temporales(rr_ms):
    """
    Métricas clásicas y fetales en el dominio del tiempo:
    - Media (BPM_mean)
    - Desviación estándar (SDNN)
    - Varianza (RR_variance = SDNN^2)
    - Máximo (RR_max) y Mínimo (RR_min)
    - Rango (RR_range = RR_max - RR_min)
    - RMSSD, pNN50 y pNN20
    """
    if len(rr_ms) < 2:
        return {
            "BPM_mean": 0.0,
            "SDNN": 0.0,
            "RR_variance": 0.0,
            "RR_min": 0.0,
            "RR_max": 0.0,
            "RR_range": 0.0,
            "RMSSD": 0.0,
            "pNN50": 0.0,
            "pNN20": 0.0,
        }

    bpm_inst = 60000.0 / rr_ms
    diff_rr = np.diff(rr_ms)
    abs_diff = np.abs(diff_rr)
    sdnn = float(np.std(rr_ms, ddof=1))
    rr_min = float(np.min(rr_ms))
    rr_max = float(np.max(rr_ms))
    rr_range = float(rr_max - rr_min)
    rr_var = float(sdnn ** 2)

    return {
        "BPM_mean": float(np.mean(bpm_inst)),
        "SDNN": sdnn,
        "RR_variance": rr_var,
        "RR_min": rr_min,
        "RR_max": rr_max,
        "RR_range": rr_range,
        "RMSSD": float(np.sqrt(np.mean(diff_rr ** 2))),
        "pNN50": float((np.sum(abs_diff > 50.0) / len(abs_diff)) * 100.0),
        "pNN20": float((np.sum(abs_diff > 20.0) / len(abs_diff)) * 100.0),
    }


def extraer_features_forma_distribucion(rr_ms):
    """
    Medidas de forma de la distribución del tacograma RR:
    - Asimetría (Skewness): sesgo de la distribución de intervalos RR
    - Curtosis (Kurtosis): apuntamiento o colas pesadas de la distribución
    """
    if len(rr_ms) < 3:
        return {
            "RR_skewness": 0.0,
            "RR_kurtosis": 0.0,
        }

    sk = float(stats.skew(rr_ms, bias=False)) if len(rr_ms) >= 3 else 0.0
    kt = float(stats.kurtosis(rr_ms, bias=False)) if len(rr_ms) >= 4 else 0.0

    return {
        "RR_skewness": float(sk) if np.isfinite(sk) else 0.0,
        "RR_kurtosis": float(kt) if np.isfinite(kt) else 0.0,
    }


def extraer_features_energeticas_senal(rr_ms, senal=None):
    """
    Medidas energéticas de la señal fisiológica (variaciones dinámicas latido a latido):
    - Energía total de la señal: sum(|dRR[n]|^2)
    - Potencia promedio: (1/N) * sum(|dRR[n]|^2)
    """
    if rr_ms is not None and len(rr_ms) >= 2:
        diff_rr = np.diff(np.asarray(rr_ms, dtype=np.float64))
        energia = float(np.sum(diff_rr ** 2))
        potencia = float(np.mean(diff_rr ** 2))
        return {
            "Signal_Energy": energia,
            "Signal_Power": potencia,
        }

    if senal is None or len(senal) == 0:
        return {
            "Signal_Energy": 0.0,
            "Signal_Power": 0.0,
        }

    sig = np.asarray(senal, dtype=np.float64)
    sig_ac = sig - np.mean(sig)
    return {
        "Signal_Energy": float(np.sum(sig_ac ** 2)),
        "Signal_Power": float(np.mean(sig_ac ** 2)),
    }


def extraer_features_arritmia_clinica(rr_ms):
    """
    Biomarcadores específicos para detección clínica de arritmias fetales
    (extrasístoles supraventriculares PAC, ventriculares PVC, ritmos irregulares,
    bradicardia y taquicardia fetal):
    - Ectopic_Beat_Ratio: % de ciclos con acoplamiento ectópico o salto arrítmico (>10% respecto a mediana local).
    - Prematurity_Index: Profundidad de acoplamiento prematuro (1 - P5(RR / RR_med)).
    - Compensatory_Pause_Ratio: Magnitud de pausa post-ectópica (P95(RR / RR_med) - 1).
    - CV_RR_Robust: Coeficiente de variación intercuartílico robusto (IQR / Mediana * 100).
    - nRMSSD: RMSSD normalizado por la duración del ciclo cardíaco (RMSSD / Mediana * 100).
    - Bradytachy_Score: Desviación basal fuera del rango normocárdico fetal (115 - 160 bpm).
    """
    if len(rr_ms) < 5:
        return {
            "Ectopic_Beat_Ratio": 0.0,
            "Prematurity_Index": 0.0,
            "Compensatory_Pause_Ratio": 0.0,
            "CV_RR_Robust": 0.0,
            "nRMSSD": 0.0,
            "Bradytachy_Score": 0.0,
        }

    med_local = _mediana_movil(rr_ms, window=9)
    ratios = rr_ms / (med_local + 1e-8)
    med_global = float(np.median(rr_ms))
    bpm_med = 60000.0 / (med_global + 1e-8)

    # Latidos con desviación rítmica significativa (>10% del ciclo basal local)
    ectopic_mask = np.abs(ratios - 1.0) > 0.10
    ectopic_ratio = float(np.mean(ectopic_mask) * 100.0)

    p5_ratio = float(np.percentile(ratios, 5))
    p95_ratio = float(np.percentile(ratios, 95))
    prematurity_idx = float(max(0.0, 1.0 - p5_ratio))
    comp_pause_ratio = float(max(0.0, p95_ratio - 1.0))

    iqr_rr = float(np.percentile(rr_ms, 75) - np.percentile(rr_ms, 25))
    cv_robust = float((iqr_rr / (med_global + 1e-8)) * 100.0)

    diff_rr = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
    n_rmssd = float((rmssd / (med_global + 1e-8)) * 100.0)

    # Desviación basal fuera del rango normocárdico fetal basal a término (130 - 158 bpm)
    bradytachy = float(max(0.0, 130.0 - bpm_med) + max(0.0, bpm_med - 158.0))

    return {
        "Ectopic_Beat_Ratio": ectopic_ratio,
        "Prematurity_Index": prematurity_idx,
        "Compensatory_Pause_Ratio": comp_pause_ratio,
        "CV_RR_Robust": cv_robust,
        "nRMSSD": n_rmssd,
        "Bradytachy_Score": bradytachy,
    }


def extraer_features_frecuenciales(rr_ms, fs_interp=4.0):
    """
    Métricas en el dominio espectral adaptadas a la fisiología fetal mediante densidad
    espectral de Welch con zero-padding (`nfft=1024`) para garantizar resolución fina
    ($\Delta f = 0.0039\text{ Hz}$) en las tres bandas fetales:
    - VLF: 0.0033 - 0.04 Hz
    - LF : 0.04 - 0.20 Hz (actividad simpática y barorrefleja fetal)
    - HF : 0.20 - 1.00 Hz (actividad vagal y movimientos respiratorios fetales)
    """
    if len(rr_ms) < 10:
        return {"VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}

    tiempo_acum = np.cumsum(rr_ms) / 1000.0
    tiempo_rel = tiempo_acum - tiempo_acum[0]
    if tiempo_rel[-1] < 4.0:
        return {"VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}

    t_interp = np.arange(0.0, tiempo_rel[-1], 1.0 / fs_interp)
    if len(t_interp) < 16:
        return {"VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}

    rr_interp = np.asarray(np.interp(t_interp, tiempo_rel, rr_ms), dtype=np.float64)
    rr_detrend = signal.detrend(rr_interp, type="linear")

    nperseg = min(len(rr_detrend), int(fs_interp * 64))
    nfft = max(1024, nperseg)

    freqs, psd = signal.welch(rr_detrend, fs=fs_interp, nperseg=nperseg, nfft=nfft)

    vlf_band = (freqs >= 0.0033) & (freqs < 0.04)
    lf_band = (freqs >= 0.04) & (freqs < 0.20)
    hf_band = (freqs >= 0.20) & (freqs <= 1.00)

    def _integrar(y_vals, x_vals):
        if len(y_vals) < 2 or not np.any(y_vals):
            return 0.0
        return float(trapezoid(y_vals, x=x_vals))

    vlf_pow = _integrar(psd[vlf_band], freqs[vlf_band])
    lf_pow = _integrar(psd[lf_band], freqs[lf_band])
    hf_pow = _integrar(psd[hf_band], freqs[hf_band])
    lf_hf = float(lf_pow / (hf_pow + 1e-8)) if hf_pow > 0 else 0.0

    return {
        "VLF": vlf_pow,
        "LF": lf_pow,
        "HF": hf_pow,
        "LF_HF_ratio": lf_hf,
    }


def extraer_features_stft(rr_ms, fs_interp=4.0):
    """
    Análisis en el Dominio Tiempo-Frecuencia mediante Transformada de Fourier
    en Ventanas (Short-Time Fourier Transform - STFT):
    Evalúa la dinámica y variabilidad espectral de la señal a lo largo del tiempo:
    - STFT_Power_Mean: Potencia espectral media de la matriz tiempo-frecuencia.
    - STFT_Power_Var: Variabilidad temporal de la energía espectral (indicador de cambios transitorios de ritmo).
    - STFT_LF_HF_Mean: Promedio temporal del balance simpático/vagal calculado ventana a ventana.
    - STFT_Spectral_Entropy: Entropía espectral promediada en el tiempo (grado de desorganización o complejidad).
    """
    if len(rr_ms) < 15:
        return {
            "STFT_Power_Mean": 0.0,
            "STFT_Power_Var": 0.0,
            "STFT_LF_HF_Mean": 0.0,
            "STFT_Spectral_Entropy": 0.0,
        }

    tiempo_acum = np.cumsum(rr_ms) / 1000.0
    tiempo_rel = tiempo_acum - tiempo_acum[0]
    if tiempo_rel[-1] < 8.0:
        return {
            "STFT_Power_Mean": 0.0,
            "STFT_Power_Var": 0.0,
            "STFT_LF_HF_Mean": 0.0,
            "STFT_Spectral_Entropy": 0.0,
        }

    t_interp = np.arange(0.0, tiempo_rel[-1], 1.0 / fs_interp)
    if len(t_interp) < 24:
        return {
            "STFT_Power_Mean": 0.0,
            "STFT_Power_Var": 0.0,
            "STFT_LF_HF_Mean": 0.0,
            "STFT_Spectral_Entropy": 0.0,
        }

    rr_interp = np.asarray(np.interp(t_interp, tiempo_rel, rr_ms), dtype=np.float64)
    rr_detrend = signal.detrend(rr_interp, type="linear")

    # Ventana de 16 segundos (~64 muestras a 4 Hz) con solapamiento del 75%
    nperseg = min(len(rr_detrend), int(fs_interp * 16))
    if nperseg < 8:
        nperseg = 8
    noverlap = int(nperseg * 0.75)
    nfft = max(256, nperseg)

    try:
        f_stft, t_stft, zxx = signal.stft(
            rr_detrend,
            fs=fs_interp,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary=None,
        )
        potencia_tf = np.abs(zxx) ** 2  # (n_freqs, n_times)

        # Potencia promedio y varianza temporal de potencia
        p_mean = float(np.mean(potencia_tf))
        p_var = float(np.var(np.sum(potencia_tf, axis=0)))

        # Balance LF/HF dinámico en el tiempo
        lf_mask = (f_stft >= 0.04) & (f_stft < 0.20)
        hf_mask = (f_stft >= 0.20) & (f_stft <= 1.00)

        lf_stft = np.sum(potencia_tf[lf_mask, :], axis=0) if np.any(lf_mask) else np.zeros(potencia_tf.shape[1])
        hf_stft = np.sum(potencia_tf[hf_mask, :], axis=0) if np.any(hf_mask) else np.zeros(potencia_tf.shape[1])
        ratios_t = lf_stft / (hf_stft + 1e-8)
        lf_hf_mean = float(np.mean(ratios_t)) if len(ratios_t) > 0 else 0.0

        # Entropía espectral tiempo-frecuencia normalizada
        psd_norm = potencia_tf / (np.sum(potencia_tf, axis=0, keepdims=True) + 1e-12)
        entropias = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=0)
        max_ent = np.log(potencia_tf.shape[0]) if potencia_tf.shape[0] > 1 else 1.0
        spectral_ent = float(np.mean(entropias / (max_ent + 1e-8)))

        return {
            "STFT_Power_Mean": p_mean,
            "STFT_Power_Var": p_var,
            "STFT_LF_HF_Mean": lf_hf_mean,
            "STFT_Spectral_Entropy": spectral_ent,
        }
    except Exception:
        return {
            "STFT_Power_Mean": 0.0,
            "STFT_Power_Var": 0.0,
            "STFT_LF_HF_Mean": 0.0,
            "STFT_Spectral_Entropy": 0.0,
        }


def _calcular_sampen_numpy(serie, m=2, r=0.2):
    """
    Cálculo de Sample Entropy (SampEn) con NumPy optimizado en memoria por bloques.
    """
    serie = np.asarray(serie, dtype=np.float64)
    N = len(serie)
    if N <= m + 1:
        return 0.0

    s_std = np.std(serie)
    if s_std < 1e-8:
        return 0.0
    r_val = r * s_std

    def _phi(dim):
        x = np.array([serie[i : i + dim] for i in range(N - dim + 1)])
        K = len(x)
        if K <= 1:
            return 0.0

        total_coincidencias = 0
        chunk_size = 256
        for start in range(0, K, chunk_size):
            end = min(K, start + chunk_size)
            chunk = x[start:end]
            diff = np.max(np.abs(chunk[:, None, :] - x[None, :, :]), axis=2)
            total_coincidencias += int(np.sum(diff <= r_val))

        total_coincidencias -= K
        return total_coincidencias / (K * (K - 1))

    try:
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m > 0 and phi_m1 > 0:
            return float(-np.log(phi_m1 / phi_m))
        return 0.0
    except Exception:
        return 0.0


def _calcular_dfa_numpy(serie):
    """
    Cálculo de DFA (Detrended Fluctuation Analysis, exponente alpha1 de corto plazo) con NumPy.
    """
    N = len(serie)
    if N < 16:
        return 0.0

    y = np.cumsum(serie - np.mean(serie))
    escalas = np.unique(np.logspace(np.log10(4), np.log10(max(5, N // 4)), num=8).astype(int))

    fluctuaciones = []
    escalas_validas = []

    for s in escalas:
        if s < 4 or s >= N:
            continue
        n_segmentos = N // s
        if n_segmentos < 2:
            continue

        f_s_total = 0.0
        for seg in range(n_segmentos):
            idx = slice(seg * s, (seg + 1) * s)
            t = np.arange(s)
            val = y[idx]
            p = np.polyfit(t, val, 1)
            tendencia = np.polyval(p, t)
            f_s_total += np.sum((val - tendencia) ** 2)

        f_s = np.sqrt(f_s_total / (n_segmentos * s))
        if f_s > 0:
            fluctuaciones.append(f_s)
            escalas_validas.append(s)

    if len(escalas_validas) < 3:
        return 0.0

    poly = np.polyfit(np.log(escalas_validas), np.log(fluctuaciones), 1)
    return float(poly[0])


def extraer_features_no_lineales(rr_ms):
    """
    Métricas no lineales: Poincaré (SD1, SD2, SD1_SD2_ratio), SampEn y DFA (alpha1).
    """
    if len(rr_ms) < 15:
        return {
            "SD1": 0.0,
            "SD2": 0.0,
            "SD1_SD2_ratio": 0.0,
            "SampEn": 0.0,
            "DFA_alpha1": 0.0,
        }

    diff_rr = np.diff(rr_ms)
    var_rr = np.var(rr_ms, ddof=1)
    var_diff = np.var(diff_rr, ddof=1)

    sd1 = np.sqrt(0.5 * var_diff)
    sd2 = np.sqrt(max(0.0, 2 * var_rr - 0.5 * var_diff))
    sd1_sd2 = float(sd1 / sd2) if sd2 > 0 else 0.0

    sampen = _calcular_sampen_numpy(rr_ms)
    dfa_alpha = _calcular_dfa_numpy(rr_ms)

    return {
        "SD1": float(sd1),
        "SD2": float(sd2),
        "SD1_SD2_ratio": sd1_sd2,
        "SampEn": sampen,
        "DFA_alpha1": dfa_alpha,
    }


def extraer_vector_caracteristicas_completo(picos_indices, fs=1000.0, senal_cruda_o_fecg=None):
    """
    Calcula el vector consolidado de características clínicas fHRV:
    - Dominio del tiempo clásico y extendido (Media, Desviación estándar, Varianza, Min, Max, Rango, RMSSD, pNN50, pNN20).
    - Medidas de forma de la distribución (Asimetría / Skewness, Curtosis / Kurtosis).
    - Medidas energéticas de la señal (Energía total y Potencia promedio).
    - Biomarcadores de arritmia / ectopia clínica fetal.
    - Dominio de la frecuencia adaptado (Welch VLF, LF, HF, LF/HF).
    - Dominio tiempo-frecuencia (STFT potencia media, varianza temporal, balance LF/HF y entropía espectral).
    - No lineales (Poincaré SD1, SD2, ratio, SampEn, DFA alpha1).
    """
    rr_ms = calcular_tacograma_rr(picos_indices, fs)
    feats = {}
    feats.update(extraer_features_temporales(rr_ms))
    feats.update(extraer_features_forma_distribucion(rr_ms))
    feats.update(extraer_features_energeticas_senal(rr_ms, senal=senal_cruda_o_fecg))
    feats.update(extraer_features_arritmia_clinica(rr_ms))
    feats.update(extraer_features_frecuenciales(rr_ms))
    feats.update(extraer_features_stft(rr_ms))
    feats.update(extraer_features_no_lineales(rr_ms))
    feats["Num_Latidos_Validos"] = len(rr_ms)
    return feats
