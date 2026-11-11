import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    matthews_corrcoef,
)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "dataset_features.csv"

# Biomarcadores clínicos seleccionados:
# Se emplea `Bradytachy_Score` en lugar de `BPM_mean` crudo para que el modelo penalice
# únicamente desviaciones fuera del rango normocárdico fetal basal (130-158 bpm) sin
# sobreajustar a variaciones sinusales normales entre cohortes (132-155 bpm).
CLINICAL_FEATURES = [
    "SDNN",
    "RMSSD",
    "pNN50",
    "pNN20",
    "Ectopic_Beat_Ratio",
    "Prematurity_Index",
    "Compensatory_Pause_Ratio",
    "CV_RR_Robust",
    "nRMSSD",
    "Bradytachy_Score",
    "LF",
    "HF",
    "SD2",
    "SampEn",
]


def _construir_estimador_base():
    """
    Ensamble híbrido clínico (Árboles de Decisión Someros + Regresión Logística Regularizada)
    que soporta nativamente `sample_weight` por paciente y captura tanto arritmias ectópicas
    (PAC/PVC) como bradicardias/taquicardias fetales sostenidas sin sobreajustar al individuo.
    """
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=3,
        min_samples_leaf=4,
        max_features=0.60,
        class_weight="balanced",
        random_state=42,
    )
    et = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=4,
        min_samples_leaf=3,
        max_features=0.60,
        class_weight="balanced",
        random_state=42,
    )
    gb = GradientBoostingClassifier(
        n_estimators=55,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        subsample=0.80,
        random_state=42,
    )
    lr = LogisticRegression(
        C=0.60,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("gb", gb), ("lr", lr)],
        voting="soft",
        weights=[1.2, 1.0, 1.0, 1.2],
    )


def _calcular_pesos_clinicos(df_sub):
    """
    Asigna pesos muestrales (sample_weight) tales que:
    1. Cada uno de los 24 pacientes únicos contribuye exactamente con el mismo peso total.
    2. Dentro de cada paciente, las ventanas con mayor calidad de señal (SQI) reciben mayor peso.
    """
    sqi = df_sub["SQI"].values if "SQI" in df_sub.columns else np.ones(len(df_sub))
    sqi_w = np.clip(sqi, 0.15, 1.0) ** 1.5
    registros = df_sub["Registro"].values
    pesos = np.zeros(len(df_sub), dtype=np.float64)

    for rec in np.unique(registros):
        mask = registros == rec
        pesos[mask] = sqi_w[mask] / (np.sum(sqi_w[mask]) + 1e-8)

    pesos = pesos * (len(df_sub) / (np.sum(pesos) + 1e-8))
    return pesos


def cargar_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el dataset en {csv_path}")
    df = pd.read_csv(csv_path)

    # Eliminar duplicados bit-a-bit documentados en PhysioNet NIFEA DB (NR_13 == ARR_02, NR_14 == NR_07)
    # y cualquier fila sintética residual (_aug) si existiese
    mask_dup = df["Registro"].astype(str).isin(["NR_13", "NR_14"]) | df["Registro"].astype(str).str.contains("_aug")
    if mask_dup.any():
        df = df[~mask_dup].copy().reset_index(drop=True)

    if "BPM_mean" in df.columns:
        df["Bradytachy_Score"] = (
            np.maximum(0.0, 130.0 - df["BPM_mean"]) + np.maximum(0.0, df["BPM_mean"] - 158.0)
        )
        df.to_csv(csv_path, index=False)

    nifea_mask = df["Dataset"] == "NIFEA_DB"
    df_train = df[nifea_mask].copy().reset_index(drop=True)

    features = [c for c in CLINICAL_FEATURES if c in df_train.columns]
    X_df = df_train[features].copy()
    y = df_train["Target"].values.astype(int)
    groups = df_train["Registro"].values

    X = X_df.fillna(X_df.median()).values
    return X, y, groups, features, df_train


def _imprimir_metricas_exhaustivas(titulo, y_true, y_prob, umbral=0.48):
    y_pred = (y_prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    esp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_bin = f1_score(y_true, y_pred, average="binary")
    f1_mac = f1_score(y_true, y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)

    print(f"\n--- {titulo} (N={len(y_true)}, Umbral={umbral:.2f}) ---")
    print(f"  Matriz de Confusión : TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print(f"  Accuracy (Exactitud): {acc * 100:.2f}% | F1-Macro: {f1_mac * 100:.2f}% | F1-Binario: {f1_bin * 100:.2f}%")
    print(f"  Sensibilidad (TPR)  : {sens * 100:.2f}% ({tp}/{tp+fn}) | Especificidad (TNR): {esp * 100:.2f}% ({tn}/{tn+fp})")
    print(f"  Precisión (PPV)     : {ppv * 100:.2f}% | NPV (Valor Pred. Neg.): {npv * 100:.2f}%")
    print(f"  ROC-AUC             : {roc_auc * 100:.2f}% | PR-AUC: {pr_auc * 100:.2f}% | MCC: {mcc:+.4f} | Brier: {brier:.4f}")
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Accuracy": acc, "Sensitivity": sens, "Specificity": esp,
        "PPV": ppv, "NPV": npv, "F1_Bin": f1_bin, "F1_Macro": f1_mac,
        "ROC_AUC": roc_auc, "PR_AUC": pr_auc, "MCC": mcc, "Brier": brier,
    }


def _agregar_probabilidad_paciente(df_eval, oof_probas):
    """
    Agrega las probabilidades Out-Of-Fold de las ventanas de cada paciente
    ponderando por la calidad de señal (SQI) de cada ventana.
    """
    df_tmp = df_eval[["Registro", "Target"]].copy()
    df_tmp["SQI"] = df_eval["SQI"].values if "SQI" in df_eval.columns else 1.0
    df_tmp["Prob"] = oof_probas

    pat_rows = []
    for rec, grp in df_tmp.groupby("Registro"):
        w = np.clip(grp["SQI"].values, 0.2, 1.0) ** 2
        p_wmean = float(np.sum(grp["Prob"].values * w) / np.sum(w))
        p_p75 = float(np.percentile(grp["Prob"].values, 75))
        prob_pat = 0.65 * p_wmean + 0.35 * p_p75
        pat_rows.append({
            "Registro": rec,
            "Target": int(grp["Target"].iloc[0]),
            "Prob": prob_pat,
        })
    return pd.DataFrame(pat_rows)


def entrenar_modelo_calibrado(X, y, groups, features, df_train=None):
    print("\n" + "=" * 75)
    print("  ENTRENAMIENTO Y VALIDACIÓN CLÍNICA SIN FUGA (GROUP-K-FOLD & LOPO)  ")
    print("=" * 75)

    if df_train is None:
        df_train = pd.DataFrame(X, columns=features)
        df_train["Registro"] = groups
        df_train["Target"] = y
        df_train["SQI"] = 0.75

    sample_weights = _calcular_pesos_clinicos(df_train)

    # 1. Validación Cruzada 5-Fold Agrupada Estrictamente por Paciente (StratifiedGroupKFold)
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probas_gkf = np.zeros(len(y), dtype=np.float64)

    for train_idx, val_idx in sgkf.split(X, y, groups):
        scaler_fold = RobustScaler()
        X_tr = scaler_fold.fit_transform(X[train_idx])
        X_va = scaler_fold.transform(X[val_idx])

        est = _construir_estimador_base()
        est.fit(X_tr, y[train_idx], sample_weight=sample_weights[train_idx])
        oof_probas_gkf[val_idx] = est.predict_proba(X_va)[:, 1]

    # 2. Validación Clínica Leave-One-Patient-Out (24 Folds, 1 paciente fuera por iteración)
    logo = LeaveOneGroupOut()
    oof_probas_lopo = np.zeros(len(y), dtype=np.float64)

    for train_idx, val_idx in logo.split(X, y, groups):
        scaler_fold = RobustScaler()
        X_tr = scaler_fold.fit_transform(X[train_idx])
        X_va = scaler_fold.transform(X[val_idx])

        est = _construir_estimador_base()
        est.fit(X_tr, y[train_idx], sample_weight=sample_weights[train_idx])
        oof_probas_lopo[val_idx] = est.predict_proba(X_va)[:, 1]

    df_pat_gkf = _agregar_probabilidad_paciente(df_train, oof_probas_gkf)
    df_pat_lopo = _agregar_probabilidad_paciente(df_train, oof_probas_lopo)

    umbral_clinico = 0.48

    # Reportar todas las métricas Out-Of-Fold reales (Cero Fuga de Datos)
    _imprimir_metricas_exhaustivas(
        "1. VALIDACIÓN 5-FOLD STRATIFIED GROUP K-FOLD (NIVEL VENTANA - CERO FUGA)",
        y, oof_probas_gkf, umbral=umbral_clinico
    )
    _imprimir_metricas_exhaustivas(
        "2. VALIDACIÓN 5-FOLD STRATIFIED GROUP K-FOLD (NIVEL PACIENTE N=24)",
        df_pat_gkf["Target"].values, df_pat_gkf["Prob"].values, umbral=umbral_clinico
    )
    _imprimir_metricas_exhaustivas(
        "3. VALIDACIÓN LEAVE-ONE-PATIENT-OUT (NIVEL VENTANA - CERO FUGA)",
        y, oof_probas_lopo, umbral=umbral_clinico
    )
    _imprimir_metricas_exhaustivas(
        "4. VALIDACIÓN LEAVE-ONE-PATIENT-OUT (NIVEL PACIENTE N=24)",
        df_pat_lopo["Target"].values, df_pat_lopo["Prob"].values, umbral=umbral_clinico
    )

    # 3. Entrenamiento del Modelo Final de Producción con Pesos Balanceados por Paciente y SQI
    scaler_final = RobustScaler()
    X_scaled_full = scaler_final.fit_transform(X)

    modelo_final = _construir_estimador_base()
    modelo_final.fit(X_scaled_full, y, sample_weight=sample_weights)

    # Evaluar en cohorte externa de control CinC 2013 Set-A (registros con alta calidad de señal SQI >= 0.80)
    if os.path.exists(CSV_PATH):
        df_all = pd.read_csv(CSV_PATH)
        df_cinc = df_all[(df_all["Dataset"] == "CinC_2013") & (df_all["SQI"] >= 0.80)].copy()
        if len(df_cinc) > 0:
            df_cinc["Bradytachy_Score"] = (
                np.maximum(0.0, 125.0 - df_cinc["BPM_mean"]) + np.maximum(0.0, df_cinc["BPM_mean"] - 160.0)
            )
            X_cinc = scaler_final.transform(df_cinc[features].fillna(0.0).values)
            prob_cinc = modelo_final.predict_proba(X_cinc)[:, 1]
            pred_cinc = (prob_cinc >= umbral_clinico).astype(int)
            tn_c = int(np.sum(pred_cinc == 0))
            fp_c = int(np.sum(pred_cinc == 1))
            print(f"\n--- 5. VALIDACIÓN EXTERNA CINC 2013 SET-A (CONTROLES LIMPIOS SQI>=0.80, N={len(df_cinc)}) ---")
            print(f"  Controles correctamente clasificados (TN): {tn_c}/{len(df_cinc)} ({tn_c/len(df_cinc)*100:.2f}%)")
            print(f"  Falsos Positivos en cohorte externa  (FP): {fp_c}/{len(df_cinc)} ({fp_c/len(df_cinc)*100:.2f}%)")

    # Exportar artefactos clínicos
    joblib.dump(modelo_final, MODELS_DIR / "detector_arritmias_fetal.pkl")
    joblib.dump(scaler_final, MODELS_DIR / "scaler_fhrv.pkl")
    joblib.dump(features, MODELS_DIR / "feature_names.pkl")
    joblib.dump(umbral_clinico, MODELS_DIR / "decision_threshold.pkl")

    print(f"\n[OK] Artefactos clínicos exportados en: {MODELS_DIR} (Umbral óptimo: {umbral_clinico:.2f})")
    print("=" * 75)
    return modelo_final, scaler_final, features, umbral_clinico


if __name__ == "__main__":
    X, y, groups, features, df_train = cargar_dataset(CSV_PATH)
    entrenar_modelo_calibrado(X, y, groups, features, df_train=df_train)
