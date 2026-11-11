import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from fqrs_detector import (
    extraer_fqrs_optimo,
    calcular_sqi_senal_fetal,
    detectar_picos_maternos,
)
from feature_extraction import calcular_tacograma_rr, extraer_vector_caracteristicas_completo

MODELS_DIR = PROJECT_ROOT / "models"


class DiagnosticoArritmiaFetalPipeline:
    def __init__(self, models_dir=None):
        base_models = Path(models_dir) if models_dir else MODELS_DIR
        modelo_path = base_models / "detector_arritmias_fetal.pkl"
        scaler_path = base_models / "scaler_fhrv.pkl"
        features_path = base_models / "feature_names.pkl"
        threshold_path = base_models / "decision_threshold.pkl"

        if not all(p.exists() for p in [modelo_path, scaler_path, features_path, threshold_path]):
            raise FileNotFoundError(
                f"Artefactos del modelo no encontrados en '{base_models}'. Ejecute src/model_trainer.py primero."
            )

        self.modelo = joblib.load(modelo_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        self.umbral_decision = float(joblib.load(threshold_path))

    def _predecir_ventana(self, vector_completo):
        x_vec = np.array([[vector_completo.get(f, 0.0) for f in self.feature_names]], dtype=np.float64)
        x_scaled = self.scaler.transform(x_vec)
        return float(self.modelo.predict_proba(x_scaled)[0, 1])

    def diagnosticar(self, sig_cruda_abd, fs=1000.0, thoracic_ref=None):
        """
        Ejecuta el pipeline clínico completo sobre un registro multicanal abdominal:
        1. Extrae complejos fQRS, componente fetal, picos maternos y calidad de señal (SQI).
        2. Si el registro supera los 90 s, evalúa por ventanas homogéneas de 60 s ponderadas
           por SQI (idéntico al protocolo Leave-One-Patient-Out de entrenamiento).
        """
        sig_cruda_abd = np.asarray(sig_cruda_abd, dtype=np.float64)
        fs = float(fs)

        # Detectar frecuencia cardíaca materna de referencia
        picos_mat, _ = detectar_picos_maternos(sig_cruda_abd, fs=fs, thoracic_ref=thoracic_ref)
        bpm_materno = 0.0
        if len(picos_mat) >= 4:
            rr_m = (np.diff(picos_mat) / fs) * 1000.0
            rr_m_val = rr_m[(rr_m >= 380.0) & (rr_m <= 1300.0)]
            if len(rr_m_val) >= 3:
                bpm_materno = float(60000.0 / np.median(rr_m_val))

        # Para visualización e inspección, extraer sobre el primer tramo de hasta 60 s
        n_vis = min(len(sig_cruda_abd), int(60.0 * fs))
        tho_vis = thoracic_ref[:n_vis] if thoracic_ref is not None else None
        picos_fqrs, fecg_ica, fuentes, idx_fetal, sig_filt = extraer_fqrs_optimo(
            sig_cruda_abd[:n_vis], fs, return_intermediates=True, thoracic_ref=tho_vis
        )

        if len(picos_fqrs) < 10:
            return {
                "error": "Señal insuficiente o calidad muy baja para identificar complejos fQRS fetales.",
                "picos_fqrs": picos_fqrs,
                "fecg_ica": fecg_ica,
                "sig_filtrada": sig_filt,
                "tacograma_rr": np.array([]),
                "sqi": 0.0,
                "bpm_materno": bpm_materno,
            }

        _, sqi_vis = calcular_sqi_senal_fetal(fecg_ica, picos_fqrs, fs=fs, bpm_materno=bpm_materno)
        tacograma_rr = calcular_tacograma_rr(picos_fqrs, fs)
        vector_completo = extraer_vector_caracteristicas_completo(picos_fqrs, fs)
        vector_completo["SQI"] = float(sqi_vis)
        vector_completo["BPM_Materno"] = float(bpm_materno)

        # Si el registro es largo (>= 90 s), agregar predicciones de ventanas de 60 s
        win_pts = int(60.0 * fs)
        step_pts = int(30.0 * fs)
        if len(sig_cruda_abd) >= int(90.0 * fs):
            probs_win = []
            sqis_win = []
            for start in range(0, len(sig_cruda_abd) - win_pts + 1, step_pts):
                sub_s = sig_cruda_abd[start : start + win_pts]
                sub_t = thoracic_ref[start : start + win_pts] if thoracic_ref is not None else None
                pks_w, fecg_w, _, _, _ = extraer_fqrs_optimo(
                    sub_s, fs, return_intermediates=True, thoracic_ref=sub_t
                )
                if len(pks_w) < 25:
                    continue
                _, sqi_w = calcular_sqi_senal_fetal(fecg_w, pks_w, fs=fs)
                v_w = extraer_vector_caracteristicas_completo(pks_w, fs)
                if v_w.get("Num_Latidos_Validos", 0) < 25:
                    continue
                probs_win.append(self._predecir_ventana(v_w))
                sqis_win.append(sqi_w)

            if probs_win:
                w = np.clip(np.array(sqis_win), 0.2, 1.0) ** 2
                p_wmean = float(np.sum(np.array(probs_win) * w) / np.sum(w))
                p_p75 = float(np.percentile(probs_win, 75))
                prob_arritmia = 0.65 * p_wmean + 0.35 * p_p75
                sqi_global = float(np.mean(sqis_win))
            else:
                prob_arritmia = self._predecir_ventana(vector_completo)
                sqi_global = float(sqi_vis)
        else:
            prob_arritmia = self._predecir_ventana(vector_completo)
            sqi_global = float(sqi_vis)

        es_arritmia = int(prob_arritmia >= self.umbral_decision)
        diagnostico_texto = (
            "Patológico / Arritmia Fetal" if es_arritmia == 1 else "Ritmo Fetal Normal (Control)"
        )

        return {
            "diagnostico": diagnostico_texto,
            "es_arritmia": es_arritmia,
            "probabilidad_arritmia": prob_arritmia * 100.0,
            "confianza": (prob_arritmia if es_arritmia == 1 else (1.0 - prob_arritmia)) * 100.0,
            "umbral_usado": self.umbral_decision,
            "sqi": sqi_global,
            "bpm_materno": bpm_materno,
            "picos_fqrs": picos_fqrs,
            "fecg_ica": fecg_ica,
            "fuentes": fuentes,
            "idx_fetal": idx_fetal,
            "sig_filtrada": sig_filt,
            "tacograma_rr": tacograma_rr,
            "metricas_fhrv": vector_completo,
        }


if __name__ == "__main__":
    from data_loader import cargar_nifeadb_con_toracico

    pipeline = DiagnosticoArritmiaFetalPipeline()
    for test_rec in ["ARR_02", "NR_02"]:
        sig, fs, _, tho = cargar_nifeadb_con_toracico(test_rec, max_abd_channels=None)
        resultado = pipeline.diagnosticar(sig, fs, thoracic_ref=tho)
        print(f"\n--- RESULTADO INFERENCIA CLÍNICA: {test_rec} ---")
        print(f"Diagnóstico   : {resultado['diagnostico']}")
        print(f"Probabilidad  : {resultado['probabilidad_arritmia']:.2f}% de Arritmia (Umbral: {resultado['umbral_usado']*100:.0f}%)")
        print(f"Calidad (SQI) : {resultado['sqi']*100:.1f}%")
        print(f"BPM Fetal     : {resultado['metricas_fhrv']['BPM_mean']:.1f} bpm | BPM Materno: {resultado['bpm_materno']:.1f} bpm")
        print(f"SDNN          : {resultado['metricas_fhrv']['SDNN']:.1f} ms | Ectopic Ratio: {resultado['metricas_fhrv']['Ectopic_Beat_Ratio']:.1f}%")
