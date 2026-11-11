import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from data_loader import cargar_nifeadb_con_toracico, cargar_cinc2013
from fqrs_detector import extraer_fqrs_optimo, calcular_sqi_senal_fetal
from feature_extraction import extraer_vector_caracteristicas_completo

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "dataset_features.csv"


def procesar_ventana_clinica(sig_abd, fs, thoracic_ref=None):
    """
    Ejecuta el pipeline DSP (TS-SVD + FastICA + Selector bSQI) y extrae el vector
    de características fHRV junto con el índice de calidad de señal (SQI).
    """
    try:
        picos, fecg, _, _, _ = extraer_fqrs_optimo(
            sig_abd, fs, return_intermediates=True, thoracic_ref=thoracic_ref
        )
        if len(picos) < 25:
            return None
        _, sqi = calcular_sqi_senal_fetal(fecg, picos, fs=fs)
        feats = extraer_vector_caracteristicas_completo(picos, fs)
        if feats.get("Num_Latidos_Validos", 0) < 25:
            return None
        feats["SQI"] = float(np.round(sqi, 4))
        return feats
    except Exception:
        return None


def extraer_ventanas_homogeneas(sig_abd, fs, thoracic_ref=None, duracion_ventana_s=60.0, paso_s=30.0):
    """
    Segmenta un registro clínico en ventanas homogéneas de 60 segundos (sin mezclar
    el registro entero de 10-20 minutos con ventanas cortas) para garantizar la misma
    escala temporal y resolución espectral en todos los pacientes.
    """
    n_samples = len(sig_abd)
    pts_ventana = int(duracion_ventana_s * fs)
    pts_paso = int(paso_s * fs)
    vectores = []

    if n_samples < pts_ventana:
        v = procesar_ventana_clinica(sig_abd, fs, thoracic_ref=thoracic_ref)
        if v is not None:
            vectores.append(v)
        return vectores

    for inicio in range(0, n_samples - pts_ventana + 1, pts_paso):
        fin = inicio + pts_ventana
        sub_sig = sig_abd[inicio:fin]
        sub_tho = thoracic_ref[inicio:fin] if thoracic_ref is not None else None
        v_seg = procesar_ventana_clinica(sub_sig, fs, thoracic_ref=sub_tho)
        if v_seg is not None:
            vectores.append(v_seg)

    return vectores


def construir_dataset_nifeadb(incluir_cinc2013=True):
    """
    Construye el dataset clínico maestro libre de aumentación sintética corrupta (_aug):
    1. Procesa los 26 pacientes reales de NIFEA DB (12 con arritmia ARR_01..12 y 14 controles
       sanos NR_01..14) aprovechando todos los canales abdominales y la referencia materna ('ECG').
    2. Procesa opcionalmente los controles externos de CinC 2013 Set-A con alta calidad de señal.
    """
    print("\n--- Construyendo Dataset Clínico Maestro (Fase 4 - Cero Fuga de Datos) ---")
    # Nota de auditoría clínica: En el repositorio oficial de PhysioNet (nifeadb/1.0.0),
    # NR_13.dat es idéntico bit a bit (MD5 cae5c8115a5b5b5a124aaddbe3f49dd0) a ARR_02.dat,
    # y NR_14.dat es idéntico bit a bit (MD5 c38be00d62f120aa65b1995574fde554) a NR_07.dat.
    # Por rigor clínico y para impedir fuga de datos entre pacientes, se emplean los 24
    # pacientes únicos verificados (ARR_01..ARR_12 y NR_01..NR_12).
    registros_arr = [f"ARR_{i:02d}" for i in range(1, 13)]
    registros_nr = [f"NR_{i:02d}" for i in range(1, 13)]

    filas_acumuladas = []

    # 1. Casos de Arritmia Clínica (NIFEA DB - Clase 1)
    for rec in tqdm(registros_arr, desc="NIFEA Arritmias (Clase 1)"):
        try:
            sig_abd, fs, _, tho_ref = cargar_nifeadb_con_toracico(
                rec, max_abd_channels=None
            )
            segmentos = extraer_ventanas_homogeneas(
                sig_abd, fs, thoracic_ref=tho_ref, duracion_ventana_s=60.0, paso_s=30.0
            )
            for v in segmentos:
                v["Registro"] = rec
                v["Dataset"] = "NIFEA_DB"
                v["Target"] = 1
                filas_acumuladas.append(v)
        except Exception as e:
            print(f"Aviso: error en {rec}: {e}")

    # 2. Casos Normales de Control (NIFEA DB - Clase 0)
    for rec in tqdm(registros_nr, desc="NIFEA Controles Sanos (Clase 0)"):
        try:
            sig_abd, fs, _, tho_ref = cargar_nifeadb_con_toracico(
                rec, max_abd_channels=None
            )
            segmentos = extraer_ventanas_homogeneas(
                sig_abd, fs, thoracic_ref=tho_ref, duracion_ventana_s=60.0, paso_s=30.0
            )
            for v in segmentos:
                v["Registro"] = rec
                v["Dataset"] = "NIFEA_DB"
                v["Target"] = 0
                filas_acumuladas.append(v)
        except Exception as e:
            print(f"Aviso: error en {rec}: {e}")

    # 3. Cohorte Externa de Validación (CinC 2013 Set-A - Controles de 60s con SQI >= 0.65)
    if incluir_cinc2013:
        registros_cinc = [f"a{i:02d}" for i in range(1, 76)]
        for rec in tqdm(registros_cinc, desc="CinC 2013 Set-A (Controles Externos)"):
            try:
                sig_cinc, fs, _, _ = cargar_cinc2013(rec)
                v = procesar_ventana_clinica(sig_cinc, fs, thoracic_ref=None)
                if v is not None and v.get("SQI", 0.0) >= 0.65:
                    v["Registro"] = rec
                    v["Dataset"] = "CinC_2013"
                    v["Target"] = 0
                    filas_acumuladas.append(v)
            except Exception as e:
                print(f"Aviso: error en {rec}: {e}")

    df = pd.DataFrame(filas_acumuladas)
    df.to_csv(CSV_PATH, index=False)

    df_nifea = df[df["Dataset"] == "NIFEA_DB"]
    print(f"\n[OK] Dataset clínico guardado exitosamente en: {CSV_PATH}")
    print(f"Total de ventanas clínicas reales : {len(df)} (NIFEA_DB: {len(df_nifea)}, CinC_2013: {len(df) - len(df_nifea)})")
    print(f"Pacientes únicos en NIFEA_DB      : {df_nifea['Registro'].nunique()} / 26")
    print(f"Distribución NIFEA_DB por Target  :\n{df_nifea['Target'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    construir_dataset_nifeadb(incluir_cinc2013=True)
