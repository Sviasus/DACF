import warnings
import numpy as np
import scipy.signal as signal  # type: ignore
from scipy.signal import butter, filtfilt, find_peaks  # type: ignore
from sklearn.decomposition import FastICA

try:
    from preprocessing import preprocesar_senal_multicanal
except ImportError:
    from src.preprocessing import preprocesar_senal_multicanal


def _corregir_polaridad_qrs(sig_1d, fs=1000.0):
    """
    Orienta los complejos QRS hacia arriba (polaridad positiva) comparando los
    percentiles extremos (99.2 vs 0.8) sobre la banda QRS (10-35 Hz).
    """
    sig_1d = np.asarray(sig_1d, dtype=np.float64).ravel()
    if len(sig_1d) < 32 or np.std(sig_1d) < 1e-8:
        return sig_1d
    nyq = 0.5 * fs
    b, a = butter(2, [10.0 / nyq, min(35.0 / nyq, 0.99)], btype="bandpass")
    s_f = filtfilt(b, a, sig_1d)
    p_pos = np.percentile(s_f, 99.2)
    p_neg = np.abs(np.percentile(s_f, 0.8))
    if p_neg > 1.08 * p_pos:
        return -sig_1d
    return sig_1d


def detectar_picos_maternos(signals, fs=1000.0, thoracic_ref=None):
    """
    Detecta con alta precisión los complejos QRS maternos (mQRS):
    - Si se dispone de canal torácico materno (`thoracic_ref`), lo emplea directamente.
    - De lo contrario, obtiene la primera componente principal (SVD) de los canales
      abdominales filtrados en la banda QRS materna (6-25 Hz).
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]
    n_samples, n_channels = signals.shape

    nyq = 0.5 * fs
    b_m, a_m = butter(2, [6.0 / nyq, min(25.0 / nyq, 0.99)], btype="bandpass")

    if thoracic_ref is not None and len(thoracic_ref) == n_samples:
        sig_ref = filtfilt(b_m, a_m, np.asarray(thoracic_ref, dtype=np.float64).ravel())
    else:
        sig_m_band = filtfilt(b_m, a_m, signals, axis=0)
        stds = np.std(sig_m_band, axis=0)
        valid_ch = np.where(stds > 1e-6)[0]
        if len(valid_ch) == 0:
            return np.array([], dtype=int), np.zeros(n_samples)
        sig_norm = sig_m_band[:, valid_ch] / (stds[valid_ch] + 1e-8)
        try:
            u, s_vals, _ = np.linalg.svd(sig_norm, full_matrices=False)
            sig_ref = u[:, 0] * s_vals[0]
        except Exception:
            canal_dom = int(np.argmax(stds))
            sig_ref = sig_m_band[:, canal_dom]

    sig_ref = _corregir_polaridad_qrs(sig_ref, fs=fs)

    sig_diff = np.gradient(sig_ref) ** 2
    w_integ = max(1, int(0.075 * fs))
    sig_integ = np.convolve(sig_diff, np.ones(w_integ) / w_integ, mode="same")

    dist_min_m = int(0.42 * fs)  # Máx ~142 bpm materno
    win_len = int(4.0 * fs)
    picos_cand = []

    for start in range(0, n_samples, win_len):
        end = min(start + win_len, n_samples)
        seg = sig_integ[start:end]
        if len(seg) < dist_min_m:
            continue
        med_s = np.median(seg)
        p96 = np.percentile(seg, 96)
        thr = med_s + 0.30 * (p96 - med_s)
        pk, _ = find_peaks(seg, height=thr, distance=dist_min_m)
        picos_cand.extend(start + pk)

    w_ref = int(0.040 * fs)
    picos_refinados = []
    for pm in sorted(list(set(picos_cand))):
        ini = max(0, pm - w_ref)
        fin = min(n_samples, pm + w_ref)
        if fin > ini:
            p_max = ini + int(np.argmax(sig_ref[ini:fin]))
            picos_refinados.append(p_max)

    picos_maternos = []
    for p in sorted(list(set(picos_refinados))):
        if not picos_maternos or (p - picos_maternos[-1]) >= dist_min_m:
            picos_maternos.append(p)

    return np.array(picos_maternos, dtype=int), sig_ref


def cancelar_complejos_maternos(signals, fs=1000.0, thoracic_ref=None):
    """
    Cancela el ECG materno (ondas P, complejo QRS y onda T) en cada canal abdominal
    mediante Sustracción de Plantilla en Subespacio SVD (TS-SVD / TS-PCA):
    - Preserva íntegramente los complejos fQRS fetales que coinciden en el tiempo con el mQRS.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]
    n_samples, n_channels = signals.shape
    signals_limpias = np.copy(signals)

    picos_maternos, _ = detectar_picos_maternos(signals, fs=fs, thoracic_ref=thoracic_ref)
    n_picos = len(picos_maternos)
    if n_picos < 4:
        return signals_limpias, list(picos_maternos)

    rr_m = np.diff(picos_maternos)
    med_rr_m = float(np.median(rr_m)) if len(rr_m) > 0 else 0.75 * fs

    w_pre = min(int(0.20 * fs), int(0.35 * med_rr_m))
    w_post = min(int(0.40 * fs), int(0.55 * med_rr_m))
    win_len = w_pre + w_post + 1
    taper = signal.windows.tukey(win_len, alpha=0.12)

    valid_beat_idx = [
        idx for idx, pm in enumerate(picos_maternos)
        if (pm - w_pre >= 0) and (pm + w_post < n_samples)
    ]

    if len(valid_beat_idx) >= 4:
        n_svd = min(4, max(2, len(valid_beat_idx) // 10))
        for ch in range(n_channels):
            col = signals_limpias[:, ch]
            mat_beats = np.zeros((len(valid_beat_idx), win_len), dtype=np.float64)
            for k, b_idx in enumerate(valid_beat_idx):
                pm = picos_maternos[b_idx]
                mat_beats[k, :] = col[pm - w_pre : pm + w_post + 1]

            try:
                u, s, vt = np.linalg.svd(mat_beats, full_matrices=False)
                reconstruccion = (u[:, :n_svd] * s[:n_svd]) @ vt[:n_svd, :]
            except Exception:
                media_beat = np.median(mat_beats, axis=0, keepdims=True)
                reconstruccion = np.repeat(media_beat, len(valid_beat_idx), axis=0)

            for k, b_idx in enumerate(valid_beat_idx):
                pm = picos_maternos[b_idx]
                col[pm - w_pre : pm + w_post + 1] -= reconstruccion[k, :] * taper

            plantilla_media = np.median(mat_beats, axis=0)
            for idx, pm in enumerate(picos_maternos):
                if idx in valid_beat_idx:
                    continue
                ini = max(0, pm - w_pre)
                fin = min(n_samples, pm + w_post + 1)
                p_ini = ini - (pm - w_pre)
                p_fin = p_ini + (fin - ini)
                seg_p = plantilla_media[p_ini:p_fin]
                seg_c = col[ini:fin]
                escala = np.dot(seg_c, seg_p) / (np.dot(seg_p, seg_p) + 1e-8)
                escala = np.clip(escala, 0.0, 2.0)
                col[ini:fin] -= escala * seg_p * taper[p_ini:p_fin]

            signals_limpias[:, ch] = col

    return signals_limpias, list(picos_maternos)


def detector_pan_tompkins_fetal(fecg_signal, fs=1000.0):
    """
    Detector Pan-Tompkins adaptativo optimizado para complejos fQRS fetales:
    1. Pasa-banda específico fQRS (10 - 38 Hz).
    2. Derivada de energía y media móvil corta (50 ms, acorde al ancho del fQRS de 20-40 ms).
    3. Umbralización adaptativa en ventanas de 2.5 s con período refractario de 235 ms
       y rescate de latidos con caída súbita de amplitud (Search-Back).
    """
    fecg_signal = np.asarray(fecg_signal, dtype=np.float64).ravel()
    n_total = len(fecg_signal)
    if n_total < int(0.5 * fs) or np.std(fecg_signal) < 1e-8:
        return np.array([], dtype=int)

    nyq = 0.5 * fs
    b, a = butter(2, [10.0 / nyq, min(38.0 / nyq, 0.99)], btype="bandpass")
    sig_band = filtfilt(b, a, fecg_signal)

    sig_diff = np.gradient(sig_band)
    sig_sq = sig_diff ** 2

    w_len = max(1, int(0.050 * fs))
    sig_integ = np.convolve(sig_sq, np.ones(w_len) / w_len, mode="same")

    dist_minima = int(0.235 * fs)
    win_size = int(2.5 * fs)
    todos_picos = []

    for start in range(0, n_total, win_size):
        end = min(start + win_size, n_total)
        segmento = sig_integ[start:end]
        if len(segmento) < dist_minima:
            continue
        med_local = np.median(segmento)
        p94 = np.percentile(segmento, 94)
        umbral_local = med_local + 0.24 * (p94 - med_local)
        picos_seg, _ = find_peaks(segmento, height=umbral_local, distance=dist_minima)
        todos_picos.extend(start + picos_seg)

    w_search = int(0.030 * fs)
    picos_ref = []
    for p in sorted(list(set(todos_picos))):
        ini = max(0, p - w_search)
        fin = min(n_total, p + w_search)
        if fin > ini:
            p_max = ini + int(np.argmax(sig_band[ini:fin]))
            picos_ref.append(p_max)

    picos_filtrados = []
    for p in sorted(list(set(picos_ref))):
        if not picos_filtrados or (p - picos_filtrados[-1]) >= dist_minima:
            picos_filtrados.append(p)

    # Search-Back cuando falta un latido aislado (RR > 1.72 * mediana)
    if len(picos_filtrados) >= 8:
        rr_init = np.diff(picos_filtrados)
        med_rr = float(np.median(rr_init))
        if 0.25 * fs <= med_rr <= 0.65 * fs:
            nuevos_picos = [picos_filtrados[0]]
            for k in range(1, len(picos_filtrados)):
                prev_p = nuevos_picos[-1]
                curr_p = picos_filtrados[k]
                gap = curr_p - prev_p
                if 1.72 * med_rr <= gap <= 2.35 * med_rr:
                    s_ini = prev_p + int(0.36 * gap)
                    s_fin = prev_p + int(0.64 * gap)
                    if s_fin > s_ini + 10:
                        sub_env = sig_integ[s_ini:s_fin]
                        med_bg = np.median(sig_integ[max(0, prev_p - int(fs)) : min(n_total, curr_p + int(fs))])
                        if np.max(sub_env) > 1.35 * med_bg:
                            cand_mid = s_ini + int(np.argmax(sub_env))
                            r_ini = max(0, cand_mid - w_search)
                            r_fin = min(n_total, cand_mid + w_search)
                            p_mid = r_ini + int(np.argmax(sig_band[r_ini:r_fin]))
                            if (p_mid - prev_p) >= dist_minima and (curr_p - p_mid) >= dist_minima:
                                nuevos_picos.append(p_mid)
                nuevos_picos.append(curr_p)
            picos_filtrados = nuevos_picos

    return np.array(picos_filtrados, dtype=int)


def calcular_sqi_senal_fetal(sig_1d, picos, fs=1000.0, bpm_materno=None, picos_maternos=None):
    """
    Calcula un Índice de Calidad de Señal de Latidos Fetales (bSQI / Score Fisiológico)
    en [0, 1] evaluando:
    1. Frecuencia cardíaca mediana en el rango fetal fisiológico (110 - 190 bpm) y
       desambiguación frente a la frecuencia cardíaca materna (`bpm_materno`).
    2. Rechazo de coincidencia excesiva de fase con los picos maternos (`picos_maternos`).
    3. Estabilidad del tacograma y SNR de los complejos QRS detectados.
    """
    if len(picos) < 8:
        return 0.0, 0.0

    duracion_s = len(sig_1d) / fs
    rr_ms = (np.diff(picos) / fs) * 1000.0
    valid_rr = rr_ms[(rr_ms >= 240.0) & (rr_ms <= 850.0)]
    if len(valid_rr) < max(6, int(0.5 * len(rr_ms))):
        return 0.0, 0.0

    med_rr = float(np.median(valid_rr))
    bpm_med = 60000.0 / (med_rr + 1e-8)

    rel_diffs = np.abs(np.diff(valid_rr)) / (med_rr + 1e-8)
    prop_estable = float(np.mean(rel_diffs < 0.18))
    iqr_rr = float(np.percentile(valid_rr, 75) - np.percentile(valid_rr, 25))
    rcv = iqr_rr / (med_rr + 1e-8)
    regularidad = prop_estable * np.exp(-2.2 * max(0.0, rcv - 0.08))

    factor_frec = 1.0
    if bpm_med < 105.0:
        factor_frec *= 0.12
    elif bpm_med < 114.0:
        factor_frec *= 0.55
    elif bpm_med > 205.0:
        factor_frec *= 0.25

    if bpm_materno is not None and bpm_materno > 45.0:
        if abs(bpm_med - bpm_materno) < 7.5:
            factor_frec *= 0.08
        elif abs(bpm_med - 2.0 * bpm_materno) < 6.0 and prop_estable < 0.65:
            factor_frec *= 0.35

    if picos_maternos is not None and len(picos_maternos) >= 6:
        pm_arr = np.asarray(picos_maternos, dtype=int)
        tol_m = int(0.020 * fs)
        idx_c = np.searchsorted(picos, pm_arr)
        coinc_m = 0
        for k, pm_val in enumerate(pm_arr):
            pos = idx_c[k]
            if (pos < len(picos) and abs(picos[pos] - pm_val) <= tol_m) or (
                pos > 0 and abs(picos[pos - 1] - pm_val) <= tol_m
            ):
                coinc_m += 1
        ratio_coinc_materna = coinc_m / float(len(pm_arr))
        if ratio_coinc_materna > 0.35:
            factor_frec *= float(np.exp(-3.5 * (ratio_coinc_materna - 0.30)))

    latidos_esperados = duracion_s * (bpm_med / 60.0)
    cobertura = min(1.0, len(valid_rr) / (0.85 * latidos_esperados + 1e-8))

    nyq = 0.5 * fs
    b, a = butter(2, [10.0 / nyq, min(38.0 / nyq, 0.99)], btype="bandpass")
    s_f = np.abs(filtfilt(b, a, sig_1d))
    amp_picos = np.median(s_f[picos])
    piso_ruido = np.percentile(s_f, 65) + 1e-8
    snr_norm = float(np.clip((amp_picos / piso_ruido - 1.5) / 6.0, 0.05, 1.0))

    sqi = float(np.clip(regularidad * cobertura * (0.65 + 0.35 * snr_norm), 0.0, 1.0))
    score_total = sqi * factor_frec
    return score_total, sqi


def _calcular_bsqi_consenso(picos_a, picos_b, fs=1000.0, tol_ms=35.0):
    """
    Calcula el acuerdo F1 (bSQI) entre dos trenes de picos dentro de +/- tol_ms.
    """
    if len(picos_a) < 5 or len(picos_b) < 5:
        return 0.0
    tol = int((tol_ms / 1000.0) * fs)
    idx_b = np.searchsorted(picos_b, picos_a)
    coincidencias = 0
    for i, pa in enumerate(picos_a):
        pos = idx_b[i]
        ok = False
        if pos < len(picos_b) and abs(picos_b[pos] - pa) <= tol:
            ok = True
        elif pos > 0 and abs(picos_b[pos - 1] - pa) <= tol:
            ok = True
        if ok:
            coincidencias += 1
    return float(2.0 * coincidencias / (len(picos_a) + len(picos_b)))


def aislar_componente_fecg(signals, fs=1000.0, n_components=None, picos_maternos=None):
    """
    Separa las fuentes bioeléctricas mediante FastICA y evalúa un banco de candidatos
    (componentes FastICA + canales abdominales post-sustracción SVD) seleccionando
    aquel que maximiza el índice fisiológico de calidad fetal (bSQI + consenso inter-canal
    + rechazo explícito de la frecuencia cardíaca materna).
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]

    n_samples, n_channels = signals.shape
    if n_components is None:
        n_components = min(n_channels, 4)

    bpm_materno = None
    if picos_maternos is not None and len(picos_maternos) >= 4:
        rr_m = (np.diff(picos_maternos) / fs) * 1000.0
        rr_m_val = rr_m[(rr_m >= 380.0) & (rr_m <= 1300.0)]
        if len(rr_m_val) >= 3:
            bpm_materno = float(60000.0 / np.median(rr_m_val))

    nyq = 0.5 * fs
    b_f, a_f = butter(3, [8.0 / nyq, min(40.0 / nyq, 0.99)], btype="bandpass")
    signals_qrs = filtfilt(b_f, a_f, signals, axis=0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = FastICA(
                n_components=n_components,
                random_state=42,
                max_iter=1500,
                tol=1e-3,
                whiten="unit-variance",
            )
            fuentes_ica = ica.fit_transform(signals_qrs)
    except Exception:
        fuentes_ica = np.copy(signals_qrs[:, :n_components])

    candidatos_sigs = []
    candidatos_picos = []
    candidatos_scores = []
    candidatos_sqi = []

    for i in range(fuentes_ica.shape[1]):
        s_corr = _corregir_polaridad_qrs(fuentes_ica[:, i], fs=fs)
        fuentes_ica[:, i] = s_corr
        pks = detector_pan_tompkins_fetal(s_corr, fs=fs)
        sc, sqi = calcular_sqi_senal_fetal(
            s_corr, pks, fs=fs, bpm_materno=bpm_materno, picos_maternos=picos_maternos
        )
        candidatos_sigs.append(s_corr)
        candidatos_picos.append(pks)
        candidatos_scores.append(sc)
        candidatos_sqi.append(sqi)

    for c in range(n_channels):
        s_corr = _corregir_polaridad_qrs(signals_qrs[:, c], fs=fs)
        pks = detector_pan_tompkins_fetal(s_corr, fs=fs)
        sc, sqi = calcular_sqi_senal_fetal(
            s_corr, pks, fs=fs, bpm_materno=bpm_materno, picos_maternos=picos_maternos
        )
        candidatos_sigs.append(s_corr)
        candidatos_picos.append(pks)
        candidatos_scores.append(sc)
        candidatos_sqi.append(sqi)

    n_cand = len(candidatos_sigs)
    scores_finales = np.array(candidatos_scores, dtype=np.float64)
    for i in range(n_cand):
        if scores_finales[i] < 0.10:
            continue
        acuerdos = []
        for j in range(n_cand):
            if i == j or scores_finales[j] < 0.15:
                continue
            acuerdos.append(_calcular_bsqi_consenso(candidatos_picos[i], candidatos_picos[j], fs=fs))
        if acuerdos:
            mejor_acuerdo = max(acuerdos)
            scores_finales[i] *= (1.0 + 0.35 * mejor_acuerdo)

    mejor_global_idx = int(np.argmax(scores_finales))
    fecg_aislado = candidatos_sigs[mejor_global_idx]

    if mejor_global_idx < fuentes_ica.shape[1]:
        mejor_idx_ica = mejor_global_idx
    else:
        mejor_idx_ica = mejor_global_idx - fuentes_ica.shape[1]
        if mejor_idx_ica < fuentes_ica.shape[1]:
            fuentes_ica[:, mejor_idx_ica] = fecg_aislado
        else:
            mejor_idx_ica = 0
            fuentes_ica[:, 0] = fecg_aislado

    return fecg_aislado, fuentes_ica, mejor_idx_ica


def extraer_fqrs_optimo(signals, fs=1000.0, return_intermediates=False, thoracic_ref=None):
    """
    Pipeline clínico de extracción de complejos fQRS fetales:
    1. Preprocesamiento multicanal (3-45 Hz + Notch 50/60 Hz + eliminación rápida de línea base).
    2. Sustracción de plantilla materna por subespacio SVD (TS-SVD) preservando fQRS superpuestos.
    3. Separación ciega de fuentes (FastICA) + banco de canales con selector inteligente bSQI.
    4. Detección Pan-Tompkins fetal con rescate Search-Back.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if len(signals) == 0:
        picos_vacio = np.array([], dtype=int)
        if return_intermediates:
            return picos_vacio, np.array([]), np.array([]), -1, np.array([])
        return picos_vacio

    sig_filtrada = preprocesar_senal_multicanal(signals, fs=fs, aplicar_notch=True)
    sig_sin_materno, picos_maternos = cancelar_complejos_maternos(
        sig_filtrada, fs=fs, thoracic_ref=thoracic_ref
    )
    fecg_aislado, fuentes, idx_fetal = aislar_componente_fecg(
        sig_sin_materno, fs=fs, picos_maternos=picos_maternos
    )
    picos_fqrs = detector_pan_tompkins_fetal(fecg_aislado, fs=fs)

    if return_intermediates:
        return picos_fqrs, fecg_aislado, fuentes, idx_fetal, sig_filtrada
    return picos_fqrs
