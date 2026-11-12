import os
import sys
import shap
import matplotlib.pyplot as plt
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wfdb

# Configuración de rutas internas
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from src.data_loader import _remuestrear_polifasico  # type: ignore
    from src.data_streamer import stream_nifeadb  # type: ignore
    from src.preprocessing import preprocesar_senal_multicanal  # type: ignore
    from src.fqrs_detector import (  # type: ignore
        extraer_fqrs_optimo,
        calcular_sqi_senal_fetal,
        detectar_picos_maternos,
    )
    from src.feature_extraction import (  # type: ignore
        calcular_tacograma_rr,
        extraer_vector_caracteristicas_completo,
    )
except ImportError:
    from data_loader import _remuestrear_polifasico  # type: ignore
    from data_streamer import stream_nifeadb  # type: ignore
    from preprocessing import preprocesar_senal_multicanal  # type: ignore
    from fqrs_detector import (  # type: ignore
        extraer_fqrs_optimo,
        calcular_sqi_senal_fetal,
        detectar_picos_maternos,
    )
    from feature_extraction import (  # type: ignore
        calcular_tacograma_rr,
        extraer_vector_caracteristicas_completo,
    )


st.set_page_config(
    page_title="CardioFetal AI | Detección No Invasiva de Arritmias",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS Estables
st.markdown(
    """
    <style>
        html, body, [class*="css"] { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; 
        }
        .metric-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            margin-bottom: 10px;
        }
        .metric-title {
            color: #94a3b8;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 1.6rem;
            font-weight: 700;
        }
        .status-box-normal {
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid #10b981;
            border-radius: 10px;
            padding: 16px;
            color: #34d399;
            font-size: 1.25rem;
            font-weight: 700;
            text-align: center;
        }
        .status-box-warning {
            background-color: rgba(245, 158, 11, 0.15);
            border: 1px solid #f59e0b;
            border-radius: 10px;
            padding: 16px;
            color: #fbbf24;
            font-size: 1.25rem;
            font-weight: 700;
            text-align: center;
        }
        .status-box-alert {
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            border-radius: 10px;
            padding: 16px;
            color: #f87171;
            font-size: 1.25rem;
            font-weight: 700;
            text-align: center;
        }
    </style>
""",
    unsafe_allow_html=True,
)


MODELS_DIR = os.path.join(CURRENT_DIR, "models")
CINC_LOCAL_DIR = os.path.join(CURRENT_DIR, "data", "cinc2013_real", "set-a")


@st.cache_resource
def cargar_artefactos():
    try:
        import sklearn._loss as _sk_loss
        if "_loss" not in sys.modules:
            sys.modules["_loss"] = _sk_loss
    except Exception:
        pass

    try:
        mod = joblib.load(os.path.join(MODELS_DIR, "detector_arritmias_fetal.pkl"))
        scl = joblib.load(os.path.join(MODELS_DIR, "scaler_fhrv.pkl"))
        fts = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        thr = joblib.load(os.path.join(MODELS_DIR, "decision_threshold.pkl"))
        return mod, scl, fts, float(thr)
    except Exception as err_load:
        try:
            try:
                from src.model_trainer import cargar_dataset, entrenar_modelo_calibrado  # type: ignore
            except ImportError:
                from model_trainer import cargar_dataset, entrenar_modelo_calibrado  # type: ignore

            csv_path = os.path.join(CURRENT_DIR, "data", "dataset_features.csv")
            if os.path.exists(csv_path):
                X, y, groups, features, df_tr = cargar_dataset(csv_path)
                mod, scl, fts, thr = entrenar_modelo_calibrado(X, y, groups, features, df_train=df_tr)
                return mod, scl, fts, float(thr)
            st.error(f"Error cargando modelos: {err_load}")
            return None, None, None, 0.48
        except Exception as err_train:
            st.error(f"Error cargando o sincronizando modelos: {err_load} | {err_train}")
            return None, None, None, 0.48


modelo, scaler, feature_names, default_threshold = cargar_artefactos()


# ----------------- FUNCIONES DE CARGA Y CACHÉ DE DATOS -----------------
@st.cache_data(show_spinner="Cargando registro biomédico CinC 2013...")
def cargar_registro_cinc(subconjunto: str, registro_id: str):
    subfolder = "set-a" if "Set-A" in subconjunto else "set-b"
    pn_dir = "challenge-2013/1.0.0/set-a" if "Set-A" in subconjunto else "challenge-2013/1.0.0/set-b"

    rutas_candidatas = [
        os.path.join(CURRENT_DIR, "data", "cinc2013_real", subfolder, registro_id),
        os.path.join(CURRENT_DIR, "data", "cinc2013", subfolder, registro_id),
    ]
    for ruta in rutas_candidatas:
        if os.path.exists(f"{ruta}.hea"):
            rec = wfdb.rdrecord(ruta)
            p_sig = getattr(rec, "p_signal", None)
            rec_fs = float(getattr(rec, "fs", 1000.0) or 1000.0)
            if p_sig is not None:
                return np.array(p_sig, dtype=np.float64, copy=True), rec_fs, "Disco Local", None

    rec = wfdb.rdrecord(registro_id, pn_dir=pn_dir)
    p_sig = getattr(rec, "p_signal", None)
    rec_fs = float(getattr(rec, "fs", 1000.0) or 1000.0)
    if p_sig is None:
        raise ValueError(f"No se pudo extraer señal del registro {registro_id}")
    return np.array(p_sig, dtype=np.float64, copy=True), rec_fs, "PhysioNet Cloud", None


@st.cache_data(show_spinner="Cargando registro clínico NIFEA DB...")
def cargar_registro_nifea(registro_id: str, duracion_max_seg: float = 60.0):
    """
    Carga registro de NIFEA DB priorizando caché en disco local (data/nifeadb)
    con respaldo en PhysioNet Cloud, separando el canal materno torácico ('ECG')
    como referencia de sustracción SVD y preservando los canales abdominales ('Abdomen_1..5').
    """
    local_path = os.path.join(CURRENT_DIR, "data", "nifeadb", registro_id)
    if os.path.exists(f"{local_path}.hea"):
        hdr = wfdb.rdheader(local_path)
        fs_hdr = float(getattr(hdr, "fs", 1000.0) or 1000.0)
        sampto = int(duracion_max_seg * fs_hdr)
        rec = wfdb.rdrecord(local_path, sampto=sampto)
        origen = "Disco Local"
    else:
        hdr = wfdb.rdheader(registro_id, pn_dir="nifeadb/1.0.0")
        fs_hdr = float(getattr(hdr, "fs", 1000.0) or 1000.0)
        sampto = int(duracion_max_seg * fs_hdr)
        rec = wfdb.rdrecord(registro_id, pn_dir="nifeadb/1.0.0", sampto=sampto)
        origen = "PhysioNet Cloud"

    p_sig = getattr(rec, "p_signal", None)
    rec_fs = float(getattr(rec, "fs", 1000.0) or 1000.0)
    sig_names = list(getattr(rec, "sig_name", []) or [])
    if p_sig is None:
        raise ValueError(f"No se pudo leer señal de {registro_id}")

    p_sig_mat = np.asarray(p_sig, dtype=np.float64)
    has_abd_label = any("abd" in str(name).lower() for name in sig_names)
    if has_abd_label:
        indices_abd = [i for i, name in enumerate(sig_names) if "abd" in str(name).lower()]
        indices_tho = [i for i, name in enumerate(sig_names) if "abd" not in str(name).lower()]
    else:
        indices_tho = [
            i for i, name in enumerate(sig_names)
            if ("tho" in str(name).lower() or "chest" in str(name).lower() or str(name).strip().upper() == "ECG")
        ]
        indices_abd = [i for i in range(len(sig_names)) if i not in indices_tho]

    signal_abd = p_sig_mat[:, indices_abd]
    thoracic_ref = p_sig_mat[:, indices_tho[0]] if indices_tho else None

    target_fs = 1000.0
    if rec_fs != target_fs and rec_fs > 0:
        signal_abd, fs_final = _remuestrear_polifasico(signal_abd, rec_fs, target_fs)
        if thoracic_ref is not None:
            thoracic_ref, _ = _remuestrear_polifasico(thoracic_ref[:, None], rec_fs, target_fs)
            thoracic_ref = thoracic_ref.ravel()
    else:
        fs_final = rec_fs

    return np.array(signal_abd, dtype=np.float64, copy=True), fs_final, origen, thoracic_ref


# ----------------- BARRA LATERAL: ENTRADA DE DATOS -----------------
st.sidebar.title("🩺 Control de Análisis")
st.sidebar.markdown("---")

modo_entrada = st.sidebar.radio(
    "Fuente de datos:",
    [
        "Base de datos CinC 2013 (Set-A y Set-B)",
        "Base de datos NIFEA (Clínica Real)",
        "Subir archivo propio (WFDB / CSV)",
    ],
    index=0,
)

sig_raw = None
thoracic_ref = None
fs = 1000.0
nombre_muestra = ""

if modo_entrada == "Base de datos CinC 2013 (Set-A y Set-B)":
    cinc_set_tipo = st.sidebar.radio("Subconjunto CinC:", ["Set-A (a01 - a75)", "Set-B (b01 - b99)"])

    if "Set-A" in cinc_set_tipo:
        registros_cinc = [f"a{i:02d}" for i in range(1, 76)]
        registro_cinc_sel = st.sidebar.selectbox("Registro CinC 2013 (Set-A):", registros_cinc, index=0)
    else:
        registros_cinc = [f"b{i:02d}" for i in range(1, 100)]
        registro_cinc_sel = st.sidebar.selectbox("Registro CinC 2013 (Set-B):", registros_cinc, index=0)

    nombre_muestra = f"CinC 2013 - {registro_cinc_sel}"
    try:
        sig_raw, fs, origen_dato, thoracic_ref = cargar_registro_cinc(cinc_set_tipo, registro_cinc_sel)
        st.sidebar.success(f"Registro `{registro_cinc_sel}` listo ({len(sig_raw)/fs:.0f}s, {fs:.0f} Hz) [{origen_dato}].")
    except Exception as e:
        st.sidebar.error(f"Error cargando registro CinC: {e}")

elif modo_entrada == "Base de datos NIFEA (Clínica Real)":
    # 24 pacientes clínicos únicos verificados (excluyendo NR_13 y NR_14 que son duplicados MD5 en PhysioNet)
    registros_nifea = (
        [f"NR_{i:02d}" for i in range(1, 13)]
        + [f"ARR_{i:02d}" for i in range(1, 13)]
    )
    registro_sel = st.sidebar.selectbox("Registro Fetal (NIFEA DB):", registros_nifea, index=0)
    nombre_muestra = f"NIFEA DB - {registro_sel}"
    try:
        sig_raw, fs, origen_dato, thoracic_ref = cargar_registro_nifea(registro_sel)
        st.sidebar.success(f"Registro `{registro_sel}` listo ({len(sig_raw)/fs:.0f}s, {fs:.0f} Hz) [{origen_dato}].")
    except Exception as e:
        st.sidebar.error(f"Error descargando registro NIFEA: {e}")

else:
    st.sidebar.markdown("**Carga de Archivos Externos**")
    archivos_subidos = st.sidebar.file_uploader(
        "Selecciona los archivos (.dat y .hea juntos, o un archivo .csv):",
        accept_multiple_files=True,
        type=["dat", "hea", "csv"],
    )

    if archivos_subidos:
        csv_file = next((f for f in archivos_subidos if f.name.lower().endswith(".csv")), None)
        dat_file = next((f for f in archivos_subidos if f.name.lower().endswith(".dat")), None)
        hea_file = next((f for f in archivos_subidos if f.name.lower().endswith(".hea")), None)

        if csv_file:
            try:
                df_subido = pd.read_csv(csv_file)
                sig_raw = df_subido.select_dtypes(include=[np.number]).values
                nombre_muestra = csv_file.name
                fs = st.sidebar.number_input("Frecuencia de muestreo (Hz):", min_value=100.0, max_value=2000.0, value=1000.0)
            except Exception as e:
                st.sidebar.error(f"Error leyendo CSV: {e}")
        elif dat_file and hea_file:
            st.sidebar.info(f"Archivos listos: `{dat_file.name}` y `{hea_file.name}`")
            btn_procesar = st.sidebar.button("▶ Procesar Registro WFDB", type="primary")
            if btn_procesar or "cached_sig_raw" in st.session_state:
                if btn_procesar:
                    upload_tmp = os.path.join(tempfile.gettempdir(), "cardiofetal_wfdb")
                    os.makedirs(upload_tmp, exist_ok=True)
                    base_name = os.path.splitext(dat_file.name)[0]
                    p_dat = os.path.join(upload_tmp, dat_file.name)
                    p_hea = os.path.join(upload_tmp, hea_file.name)
                    with open(p_dat, "wb") as f_out:
                        f_out.write(dat_file.getbuffer())
                    with open(p_hea, "wb") as f_out:
                        f_out.write(hea_file.getbuffer())
                    try:
                        record = wfdb.rdrecord(os.path.join(upload_tmp, base_name))
                        p_sig_up = getattr(record, "p_signal", None)
                        fs_up = float(getattr(record, "fs", 1000.0) or 1000.0)
                        if p_sig_up is not None:
                            st.session_state["cached_sig_raw"] = np.array(p_sig_up, dtype=np.float64, copy=True)
                            st.session_state["cached_fs"] = fs_up
                            st.session_state["cached_name"] = base_name
                    except Exception as e:
                        st.sidebar.error(f"Error interpretando WFDB: {e}")

                if "cached_sig_raw" in st.session_state:
                    sig_raw = st.session_state["cached_sig_raw"]
                    fs = st.session_state["cached_fs"]
                    nombre_muestra = st.session_state["cached_name"]
        else:
            st.sidebar.warning("Para registros WFDB, sube tanto el archivo .dat como el .hea simultáneamente.")

tiempo_visualizar = st.sidebar.slider("Ventana de Señal a Visualizar (s):", min_value=3.0, max_value=20.0, value=8.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Calibración Diagnóstica")
umbral_clinico = st.sidebar.slider(
    "Umbral de Decisión Clínico",
    min_value=0.20,
    max_value=0.80,
    value=float(default_threshold) if default_threshold else 0.48,
    step=0.02,
    help="Define el punto de corte de probabilidad para alertar arritmias fetales.",
)

# ----------------- PANEL PRINCIPAL -----------------
st.title("🩺 Sistema de Monitoreo y Detección de Arritmias Fetales")
st.markdown(
    "Plataforma clínica de procesamiento digital de señales bioeléctricas abdominales (**ni-fECG**), "
    "sustracción materna por subespacio (**TS-SVD**), aislamiento por **FastICA + Selector bSQI** y análisis multiparamétrico de **fHRV**."
)

if sig_raw is not None:
    try:
        duracion_sec = len(sig_raw) / fs

        with st.spinner("Procesando señal biomédica y separando componente fetal..."):
            picos_mat, _ = detectar_picos_maternos(sig_raw, fs=fs, thoracic_ref=thoracic_ref)
            bpm_materno = 0.0
            if len(picos_mat) >= 4:
                rr_m = (np.diff(picos_mat) / fs) * 1000.0
                rr_m_val = rr_m[(rr_m >= 380.0) & (rr_m <= 1300.0)]
                if len(rr_m_val) >= 3:
                    bpm_materno = float(60000.0 / np.median(rr_m_val))

            fqrs_peaks, fecg_ica, fuentes_todas, idx_fetal, sig_filt = extraer_fqrs_optimo(
                sig_raw, fs, return_intermediates=True, thoracic_ref=thoracic_ref
            )
            _, sqi_fetal = calcular_sqi_senal_fetal(
                fecg_ica, fqrs_peaks, fs=fs, bpm_materno=bpm_materno, picos_maternos=picos_mat
            )
            features_dict = extraer_vector_caracteristicas_completo(fqrs_peaks, fs)
            features_dict["SQI"] = float(sqi_fetal)
            features_dict["BPM_Materno"] = float(bpm_materno)

        st.info(
            f"📋 **Muestra Activa:** `{nombre_muestra}` | **Duración:** `{duracion_sec:.1f} s` | "
            f"**Fs:** `{fs:.0f} Hz` | **Latidos Fetales (fQRS):** `{len(fqrs_peaks)}` | "
            f"**FC Materna Ref:** `{bpm_materno:.1f} bpm` | **Calidad Señal (bSQI):** `{sqi_fetal * 100:.1f}%`"
        )

        if sqi_fetal < 0.35:
            st.warning(
                "⚠️ **Advertencia de Calidad de Señal (bSQI < 35%):** El contacto de los electrodos abdominales "
                "presenta ruido electromiográfico elevado. Verifique la colocación de derivaciones antes de emitir un juicio clínico."
            )

        if modelo is not None and scaler is not None and feature_names is not None and features_dict is not None:
            df_feat = pd.DataFrame([features_dict])
            for col_req in feature_names:
                if col_req not in df_feat.columns:
                    df_feat[col_req] = 0.0
            X_mat = np.asarray(df_feat[feature_names].values, dtype=np.float64)

            X_scaled = scaler.transform(X_mat)
            prob_patologia = float(modelo.predict_proba(X_scaled)[0, 1])

            # --- RENDERIZADO DE LAS TARJETAS (Mantiene tu código original) ---
            col_diag, col_bpm, col_sdnn, col_ect = st.columns([1.4, 1, 1, 1])
            with col_diag:
                if prob_patologia < (umbral_clinico - 0.10):
                    st.markdown('<div class="status-box-normal">✅ Ritmo Fetal Normal (Control)</div>', unsafe_allow_html=True)
                    st.caption(f"Probabilidad de arritmia: **{prob_patologia * 100:.1f}%** | Certeza Normal: **{(1 - prob_patologia) * 100:.1f}%**")
                elif prob_patologia < umbral_clinico:
                    st.markdown('<div class="status-box-warning">🟡 Observación Clínica (Límite)</div>', unsafe_allow_html=True)
                    st.caption(f"Probabilidad de arritmia: **{prob_patologia * 100:.1f}%** | Cercano al umbral ({umbral_clinico * 100:.0f}%).")
                else:
                    st.markdown('<div class="status-box-alert">⚠️ ALERTA: Patología / Arritmia Fetal</div>', unsafe_allow_html=True)
                    st.caption(f"Probabilidad de arritmia: **{prob_patologia * 100:.1f}%** | Supera umbral clínico ({umbral_clinico * 100:.0f}%).")

            with col_bpm:
                st.markdown(f'<div class="metric-card"><div class="metric-title">FC Fetal</div><div class="metric-value">{features_dict.get("BPM_mean", 0):.1f} <span style="font-size: 0.9rem;">bpm</span></div></div>', unsafe_allow_html=True)
            with col_sdnn:
                st.markdown(f'<div class="metric-card"><div class="metric-title">SDNN</div><div class="metric-value">{features_dict.get("SDNN", 0):.1f} <span style="font-size: 0.9rem;">ms</span></div></div>', unsafe_allow_html=True)
            with col_ect:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Índice Ectopia</div><div class="metric-value">{features_dict.get("Ectopic_Beat_Ratio", 0):.1f} <span style="font-size: 0.9rem;">%</span></div></div>', unsafe_allow_html=True)

            st.markdown("---")

            # --- NUEVA SECCIÓN: EXPLICABILIDAD CLÍNICA CON SHAP ---
            with st.expander("🧠 ¿Por qué el modelo tomó esta decisión? (Análisis SHAP)"):
                st.markdown("Este gráfico muestra cómo cada característica del corazón fetal sumó (rojo) o restó (azul) a la probabilidad de predecir una arritmia.")
                try:
                    # 1. Extraer el modelo base del calibrador
                    if hasattr(modelo, "calibrated_classifiers_"):
                        if hasattr(modelo.calibrated_classifiers_[0], "estimator"):
                            modelo_base = modelo.calibrated_classifiers_[0].estimator
                        else:
                            modelo_base = modelo.calibrated_classifiers_[0].base_estimator
                    else:
                        modelo_base = modelo
                        
                    # 2. Desempaquetar si hay un VotingClassifier
                    if type(modelo_base).__name__ == 'VotingClassifier' and hasattr(modelo_base, "estimators_"):
                        modelo_base = modelo_base.estimators_[0]
                        
                    # 3. Inicializar el explicador SHAP con el modelo de árbol puro
                    explainer = shap.TreeExplainer(modelo_base)
                    shap_values = explainer.shap_values(X_scaled)
                    
                    # 4. Formatear la salida asegurando que sea un vector 1D (solo la clase Patológica)
                    if isinstance(shap_values, list):
                        # Si devuelve una lista por clase, tomamos la clase 1 (Arritmia) y la muestra 0
                        valores_shap_1d = shap_values[1][0]
                        valor_base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                    else:
                        # Si devuelve un array numpy
                        if len(shap_values.shape) == 3:
                            # Forma (muestras, caracteristicas, clases) -> ej. (1, 26, 2)
                            valores_shap_1d = shap_values[0, :, 1]
                        else:
                            # Forma (muestras, caracteristicas) -> ej. (1, 26)
                            valores_shap_1d = shap_values[0]
                            
                        valor_base = explainer.expected_value
                        if isinstance(valor_base, (list, np.ndarray)) and len(valor_base) > 1:
                            valor_base = valor_base[1]
                    
                    # 5. Generar el gráfico de cascada (Waterfall)
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 4))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=valores_shap_1d, 
                            base_values=float(valor_base), 
                            data=X_mat[0],  # Valores clínicos reales, no escalados
                            feature_names=feature_names
                        ),
                        max_display=8,
                        show=False
                    )
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)
                    
                except Exception as e_shap:
                    st.warning(f"No se pudo generar la explicación SHAP para este modelo. Detalle técnico: {e_shap}")
           

            with col_diag:
                if prob_patologia < (umbral_clinico - 0.10):
                    st.markdown('<div class="status-box-normal">✅ Ritmo Fetal Normal (Control)</div>', unsafe_allow_html=True)
                    st.caption(f"Probabilidad de arritmia: **{prob_patologia * 100:.1f}%** | Certeza Normal: **{(1 - prob_patologia) * 100:.1f}%**")
                elif prob_patologia < umbral_clinico:
                    st.markdown('<div class="status-box-warning">🟡 Observación Clínica (Límite)</div>', unsafe_allow_html=True)
                    st.caption(f"Probabilidad de arritmia: **{prob_patologia * 100:.1f}%** | Cercano al umbral ({umbral_clinico * 100:.0f}%).")
                else:
                    st.markdown('<div class="status-box-alert">⚠️ ALERTA: Patología / Arritmia Fetal</div>', unsafe_allow_html=True)
                    st.caption(f"Probabilidad de arritmia: **{prob_patologia * 100:.1f}%** | Supera umbral clínico ({umbral_clinico * 100:.0f}%).")

            with col_bpm:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">FC Fetal (vs Materna {bpm_materno:.0f} bpm)</div>
                        <div class="metric-value">{features_dict.get('BPM_mean', 0):.1f} <span style="font-size: 0.9rem;">bpm</span></div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with col_sdnn:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">SDNN (Variabilidad Total)</div>
                        <div class="metric-value">{features_dict.get('SDNN', 0):.1f} <span style="font-size: 0.9rem;">ms</span></div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with col_ect:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Índice de Ectopia / Irregularidad</div>
                        <div class="metric-value">{features_dict.get('Ectopic_Beat_Ratio', 0):.1f} <span style="font-size: 0.9rem;">%</span></div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Visualización de Señales
            st.subheader("📈 Señales Electrocardiográficas Abdominales y fECG Reconstruido")
            n_pts = int(min(tiempo_visualizar, duracion_sec) * fs)
            t_vec = np.linspace(0, n_pts / fs, n_pts)

            fig_signals = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                subplot_titles=(
                    "Canal Abdominal 1 (Señal Preprocesada)",
                    "Componente Fetal Aislado (TS-SVD + FastICA + Selector bSQI)",
                    "Detección de Complejos fQRS (Pan-Tompkins Fetal Adaptativo)",
                ),
                vertical_spacing=0.08,
            )

            fig_signals.add_trace(
                go.Scatter(x=t_vec, y=sig_filt[:n_pts, 0], line=dict(color="#38bdf8", width=1.2), name="Abdominal Ch1"),
                row=1, col=1,
            )
            fig_signals.add_trace(
                go.Scatter(x=t_vec, y=fecg_ica[:n_pts], line=dict(color="#a855f7", width=1.2), name="fECG Aislado"),
                row=2, col=1,
            )
            fig_signals.add_trace(
                go.Scatter(x=t_vec, y=fecg_ica[:n_pts], line=dict(color="#34d399", width=1.2), name="fQRS"),
                row=3, col=1,
            )

            picos_plot = [p for p in fqrs_peaks if p < n_pts]
            if picos_plot:
                fig_signals.add_trace(
                    go.Scatter(
                        x=np.array(picos_plot) / fs,
                        y=fecg_ica[picos_plot],
                        mode="markers",
                        marker=dict(symbol="circle", size=8, color="#ef4444"),
                        name="Picos fQRS",
                    ),
                    row=3, col=1,
                )

            fig_signals.update_layout(height=550, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            st.plotly_chart(fig_signals, width="stretch")

            st.markdown("---")

            # Gráficas fHRV
            st.subheader("📊 Análisis de Variabilidad del Ritmo Cardíaco Fetal (fHRV)")
            rr_clean = calcular_tacograma_rr(fqrs_peaks, fs)

            col_taco, col_poincare, col_hist = st.columns(3)

            with col_taco:
                fig_taco = go.Figure()
                fig_taco.add_trace(
                    go.Scatter(y=rr_clean, mode="lines+markers", marker=dict(size=3), line=dict(color="#38bdf8", width=1.2))
                )
                fig_taco.update_layout(title="Tacograma RR Fetal", xaxis_title="Latido", yaxis_title="RR (ms)", template="plotly_dark", height=340)
                st.plotly_chart(fig_taco, width="stretch")

            with col_poincare:
                if len(rr_clean) > 2:
                    fig_poinc = go.Figure()
                    fig_poinc.add_trace(
                        go.Scatter(x=rr_clean[:-1], y=rr_clean[1:], mode="markers", marker=dict(color="#f472b6", size=4, opacity=0.7))
                    )
                    fig_poinc.update_layout(title="Diagrama de Poincaré", xaxis_title="RR_n (ms)", yaxis_title="RR_{n+1} (ms)", template="plotly_dark", height=340)
                    st.plotly_chart(fig_poinc, width="stretch")

            with col_hist:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=rr_clean, nbinsx=30, marker_color="#fbbf24", opacity=0.8))
                fig_hist.update_layout(title="Distribución de Intervalos RR", xaxis_title="RR (ms)", yaxis_title="Frecuencia", template="plotly_dark", height=340)
                st.plotly_chart(fig_hist, width="stretch")

            with st.expander("🔬 Ver Tabla Detallada de Biomarcadores Fisiológicos"):
                df_display = pd.DataFrame([features_dict]).T.reset_index()
                df_display.columns = ["Biomarcador / Métrica", "Valor Calculado"]
                st.dataframe(df_display, width="stretch")

    except Exception as err:
        st.error(f"Error procesando la señal: {err}")
else:
    st.warning("Selecciona un registro de la base de datos o sube archivos locales desde la barra lateral izquierda para comenzar.")
