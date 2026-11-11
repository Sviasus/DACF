import os
import sys
from pathlib import Path
import wfdb
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from data_loader import (
    DATA_DIR,
    _remuestrear_polifasico,
    _seleccionar_mejores_canales_abdominales,
    cargar_cinc2013,
    cargar_nifeadb,
    visualizar_canales,
)


def stream_cinc2013(record_name="a01"):
    """
    Lee un registro de CinC 2013 (Set A) priorizando caché local en disco
    (data/cinc2013_real/set-a) con respaldo en streaming desde PhysioNet.
    """
    try:
        return cargar_cinc2013(record_name)
    except Exception:
        record = wfdb.rdrecord(record_name, pn_dir="challenge-2013/1.0.0/set-a")
        try:
            ann = wfdb.rdann(record_name, extension="fqrs", pn_dir="challenge-2013/1.0.0/set-a")
            fqrs_indices = ann.sample
        except Exception:
            fqrs_indices = None
        return (
            np.asarray(record.p_signal, dtype=np.float64),
            float(record.fs),
            fqrs_indices,
            list(record.sig_name),
        )


def stream_fecgsyndb(index=0, target_fs=1000.0):
    """
    Obtiene la ruta oficial de la lista de PhysioNet y extrae los 4 canales de la cruz abdominal,
    remuestreando con filtro polifásico a 1000 Hz.
    """
    record_list = wfdb.get_record_list("fecgsyndb")
    full_path = record_list[index]
    sub_folder, rec_name = os.path.split(full_path)
    pn_path = f"fecgsyndb/1.0.0/{sub_folder}"

    record = wfdb.rdrecord(rec_name, pn_dir=pn_path)
    indices_cruz = [3, 12, 15, 27]
    signal_4ch = np.asarray(record.p_signal[:, indices_cruz], dtype=np.float64)
    channel_names = [record.sig_name[i] for i in indices_cruz]

    if target_fs is not None and record.fs != target_fs:
        signal_4ch, fs_final = _remuestrear_polifasico(signal_4ch, record.fs, target_fs)
    else:
        fs_final = float(record.fs)

    return signal_4ch, fs_final, channel_names, full_path


def stream_nifeadb(record_name="ARR_01", max_abd_channels=4):
    """
    Lee registros clínicos reales ('ARR_01' a 'ARR_12', 'NR_01' a 'NR_14')
    priorizando disco local (data/nifeadb) con respaldo en PhysioNet Cloud,
    utilizando remuestreo polifásico a 1000 Hz.
    Retorna: signal_abd (N, C), fs (Hz), channel_names
    """
    try:
        return cargar_nifeadb(
            record_name=record_name,
            max_abd_channels=max_abd_channels,
        )
    except Exception:
        record = wfdb.rdrecord(record_name, pn_dir="nifeadb/1.0.0")
        p_sig = np.asarray(record.p_signal, dtype=np.float64)

        has_abd_label = any("abd" in str(name).lower() for name in record.sig_name)
        if has_abd_label:
            indices_abd = [i for i, name in enumerate(record.sig_name) if "abd" in str(name).lower()]
        else:
            indices_tho = [
                i for i, name in enumerate(record.sig_name)
                if ("tho" in str(name).lower() or "chest" in str(name).lower() or str(name).strip().upper() == "ECG")
            ]
            indices_abd = [i for i in range(len(record.sig_name)) if i not in indices_tho]

        if max_abd_channels is not None and len(indices_abd) > max_abd_channels:
            indices_sel = _seleccionar_mejores_canales_abdominales(
                p_sig, indices_abd, max_ch=max_abd_channels
            )
        else:
            indices_sel = indices_abd

        signal_abd = p_sig[:, indices_sel]

        target_fs = 1000.0
        if record.fs != target_fs:
            signal_abd, fs_final = _remuestrear_polifasico(signal_abd, record.fs, target_fs)
        else:
            fs_final = float(record.fs)

        channel_names = [record.sig_name[i] for i in indices_sel]
        return signal_abd, fs_final, channel_names
