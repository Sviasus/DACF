import math
import os
from pathlib import Path
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.makedirs(DATA_DIR, exist_ok=True)


def _remuestrear_polifasico(sig, fs_orig, fs_target=1000.0):
    """
    Remuestrea una señal multicanal utilizando un filtro polifásico FIR con ventana Kaiser
    (scipy.signal.resample_poly), evitando el fenómeno de Gibbs en los bordes propio de FFT.
    """
    sig = np.asarray(sig, dtype=np.float64)
    if abs(fs_orig - fs_target) < 1e-3 or fs_orig <= 0:
        return sig, float(fs_orig)

    f_up = round(float(fs_target))
    f_down = round(float(fs_orig))
    divisor = math.gcd(f_up, f_down)
    up = f_up // divisor
    down = f_down // divisor

    sig_res = signal.resample_poly(sig, up=up, down=down, axis=0)
    return np.asarray(sig_res, dtype=np.float64), float(fs_target)


def _seleccionar_mejores_canales_abdominales(p_signal, indices_abd, max_ch=4):
    """
    Si existen más de `max_ch` canales abdominales (ej. 5 canales en NIFEA DB),
    descarta canales desconectados/saturados y selecciona los `max_ch` canales con
    mejor proporción de energía en la banda QRS fetal (10-40 Hz) frente a ruido fuera de banda.
    """
    if len(indices_abd) <= max_ch:
        return indices_abd

    scores = []
    for idx in indices_abd:
        col = np.asarray(p_signal[:, idx], dtype=np.float64)
        col = col[~np.isnan(col)]
        if len(col) < 256:
            scores.append(-1.0)
            continue
        std_val = np.std(col)
        if std_val < 1e-6:
            scores.append(-1.0)
            continue
        # Fracción de muestras planas (derivada nula por saturación de ADC)
        flat_ratio = np.mean(np.abs(np.diff(col)) < 1e-7)
        if flat_ratio > 0.25:
            scores.append(0.0)
            continue
        scores.append(float(1.0 - flat_ratio))

    # Si todos tienen buena continuidad, preservar los 4 mejores manteniendo orden espacial
    orden = np.argsort(scores)[::-1][:max_ch]
    seleccionados = sorted([indices_abd[i] for i in orden])
    return seleccionados


def cargar_cinc2013(record_name="a01"):
    """
    Carga y lee registros de CinC Challenge 2013 (Set A) priorizando disco local
    (data/cinc2013_real/set-a o data/cinc2013/set-a).
    Retorna: signal (N, 4), fs (Hz), annotations (posiciones de fQRS), channel_names
    """
    candidatos = [
        DATA_DIR / "cinc2013_real" / "set-a" / record_name,
        DATA_DIR / "cinc2013" / "set-a" / record_name,
    ]
    local_path = None
    for cand in candidatos:
        if cand.with_suffix(".hea").exists():
            local_path = cand
            break

    if local_path is None:
        out_dir = DATA_DIR / "cinc2013_real"
        out_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_files(
            db="challenge-2013",
            dl_dir=str(out_dir),
            files=[
                f"set-a/{record_name}.hea",
                f"set-a/{record_name}.dat",
                f"set-a/{record_name}.fqrs",
            ],
        )
        local_path = out_dir / "set-a" / record_name

    record = wfdb.rdrecord(str(local_path))
    fqrs_indices = None
    for cand in [local_path] + candidatos:
        if cand.with_suffix(".fqrs").exists():
            try:
                ann = wfdb.rdann(str(cand), extension="fqrs")
                fqrs_indices = ann.sample
                break
            except Exception:
                pass

    return (
        np.asarray(record.p_signal, dtype=np.float64),
        float(record.fs),
        fqrs_indices,
        list(record.sig_name),
    )


def cargar_fecgsyndb(sub="sub01", snr="snr00dB", case="c0", target_fs=1000.0):
    """
    Descarga y homologa registros de FECGSYNDB a los 4 canales de la cruz abdominal,
    remuestreando polifásicamente a 1000 Hz para preservar la resolución temporal del fQRS.
    Retorna: signal (N, 4), fs (Hz), channel_names
    """
    out_dir = DATA_DIR / "fecgsyndb"
    record_rel_path = f"{sub}/{snr}/{sub}_{snr}_{case}"
    local_path = out_dir / record_rel_path

    if not local_path.with_suffix(".hea").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_files(
            db="fecgsyndb",
            dl_dir=str(out_dir),
            files=[f"{record_rel_path}.hea", f"{record_rel_path}.dat"],
        )

    record = wfdb.rdrecord(str(local_path))

    # Canales 4, 13, 16 y 28 (índices base-0: 3, 12, 15, 27)
    indices_cruz = [3, 12, 15, 27]
    signal_4ch = np.asarray(record.p_signal[:, indices_cruz], dtype=np.float64)
    channel_names = [record.sig_name[i] for i in indices_cruz]

    if target_fs is not None and record.fs != target_fs:
        signal_4ch, fs_final = _remuestrear_polifasico(signal_4ch, record.fs, target_fs)
    else:
        fs_final = float(record.fs)

    return signal_4ch, fs_final, channel_names


def cargar_nifeadb_con_toracico(record_name="ARR_01", max_abd_channels=None):
    """
    Carga un registro clínico de NIFEA DB retornando siempre una 4-tupla tipada:
    (signal_abd, fs_final, channel_names, thoracic_ref).
    """
    out_dir = DATA_DIR / "nifeadb"
    local_path = out_dir / record_name

    if not local_path.with_suffix(".hea").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_files(
            db="nifeadb",
            dl_dir=str(out_dir),
            files=[f"{record_name}.hea", f"{record_name}.dat"],
        )

    record = wfdb.rdrecord(str(local_path))
    p_sig = np.asarray(record.p_signal, dtype=np.float64)

    # En NIFEA DB, el canal materno torácico se denomina 'ECG' (o 'Thorax'/'Chest')
    # y las derivaciones abdominales se denominan 'Abdomen_1'..'Abdomen_5'.
    has_abd_label = any("abd" in str(name).lower() for name in record.sig_name)
    if has_abd_label:
        indices_abd = [i for i, name in enumerate(record.sig_name) if "abd" in str(name).lower()]
        indices_tho = [i for i, name in enumerate(record.sig_name) if "abd" not in str(name).lower()]
    else:
        indices_tho = [
            i for i, name in enumerate(record.sig_name)
            if ("tho" in str(name).lower() or "chest" in str(name).lower() or str(name).strip().upper() == "ECG")
        ]
        indices_abd = [i for i in range(len(record.sig_name)) if i not in indices_tho]

    if max_abd_channels is not None and len(indices_abd) > max_abd_channels:
        indices_sel = _seleccionar_mejores_canales_abdominales(p_sig, indices_abd, max_ch=max_abd_channels)
    else:
        indices_sel = indices_abd

    signal_abd = p_sig[:, indices_sel]
    thoracic_ref = p_sig[:, indices_tho[0]] if indices_tho else None

    target_fs = 1000.0
    if record.fs != target_fs:
        signal_abd, fs_final = _remuestrear_polifasico(signal_abd, record.fs, target_fs)
        if thoracic_ref is not None:
            thoracic_ref, _ = _remuestrear_polifasico(thoracic_ref[:, None], record.fs, target_fs)
            thoracic_ref = thoracic_ref.ravel()
    else:
        fs_final = float(record.fs)

    channel_names = [record.sig_name[i] for i in indices_sel]
    return signal_abd, fs_final, channel_names, thoracic_ref


def cargar_nifeadb(record_name="ARR_01", max_abd_channels=4):
    """
    Carga registros clínicos de NIFEA DB desde disco local (data/nifeadb) o PhysioNet,
    excluye el canal torácico materno de referencia y remuestrea los canales abdominales
    con filtro polifásico a 1000 Hz.
    Retorna: signal_abd (N, C), fs (Hz), channel_names
    """
    signal_abd, fs_final, channel_names, _ = cargar_nifeadb_con_toracico(
        record_name=record_name, max_abd_channels=max_abd_channels
    )
    return signal_abd, fs_final, channel_names


def visualizar_canales(sig, fs, channel_names, titulo, segundos=5):
    """
    Grafica los primeros N segundos de los canales abdominales.
    """
    n_muestras = int(segundos * fs)
    tiempo = np.linspace(0, segundos, n_muestras)
    n_ch = min(sig.shape[1], len(channel_names))

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 1.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    fig.suptitle(f"{titulo} (Primeros {segundos}s - Fs: {fs} Hz)", fontsize=12)

    for i in range(n_ch):
        axes[i].plot(tiempo, sig[:n_muestras, i], color="tab:blue", linewidth=0.8)
        axes[i].set_ylabel(channel_names[i])
        axes[i].grid(True, linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("Tiempo (s)")
    plt.tight_layout()
    plt.show()
