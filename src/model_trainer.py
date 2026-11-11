import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "dataset_features.csv"
SPLITS_CSV_PATH = DATA_DIR / "dataset_features_splits.csv"


def cargar_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el dataset en {csv_path}")
    df = pd.read_csv(csv_path)

    nifea_mask = df["Dataset"] == "NIFEA_DB"
    if nifea_mask.sum() >= 200:
        df_train = df[nifea_mask].copy()
    else:
        df_train = df.copy()

    columnas_excluidas = ["Registro", "Dataset", "Target", "Split"]
    features = [c for c in df_train.columns if c not in columnas_excluidas]

    # Imputación robusta por mediana directamente en el dataframe
    df_train[features] = df_train[features].fillna(df_train[features].median())
    
    return df_train, features


def entrenar_modelo_calibrado(df, features):
    print("\n" + "=" * 70)
    print("  ENTRENAMIENTO Y CALIBRACIÓN CON PARTICIÓN 70/15/15  ")
    print("=" * 70)

    X = df[features].values
    y = df["Target"].values
    indices = df.index.values

    # 1. Partición de datos garantizando la proporción de clases (stratify)
    # Primero separamos el 15% para Test (dejando 85% para Train+Val)
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X, y, indices, test_size=0.15, random_state=42, stratify=y
    )

    # Luego separamos el 15% para Validación del 85% restante (15/85 = 0.1765)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp, test_size=(0.15 / 0.85), random_state=42, stratify=y_temp
    )

    # 2. Etiquetar y guardar las muestras para trazabilidad
    df.loc[idx_train, 'Split'] = 'Train'
    df.loc[idx_val, 'Split'] = 'Validation'
    df.loc[idx_test, 'Split'] = 'Test'
    df.to_csv(SPLITS_CSV_PATH, index=False)
    print(f"[*] Distribución guardada en: {SPLITS_CSV_PATH}")
    print(f"    - Train : {len(y_train)} muestras ({len(y_train)/len(y)*100:.1f}%)")
    print(f"    - Val   : {len(y_val)} muestras ({len(y_val)/len(y)*100:.1f}%)")
    print(f"    - Test  : {len(y_test)} muestras ({len(y_test)/len(y)*100:.1f}%)")

    # 3. Escalador Robusto (AJUSTADO SOLO CON TRAIN para evitar fuga de datos)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) # Solo transformar
    X_test_scaled = scaler.transform(X_test) # Solo transformar

    # 4. Modelo Base y Calibración
    base_estimator = GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.07, max_depth=3,
        min_samples_leaf=4, subsample=0.85, random_state=42
    )

    modelo_calibrado = CalibratedClassifierCV(
        estimator=base_estimator, method="isotonic", cv=5
    )
    # Se entrena exclusivamente con el 70% de los datos
    modelo_calibrado.fit(X_train_scaled, y_train)

    # 5. Evaluación de Probabilidades (Estrictamente sobre el 15% de TEST)
    umbral_clinico = 0.50
    probas_test = modelo_calibrado.predict_proba(X_test_scaled)[:, 1]
    
    probas_sanos = probas_test[y_test == 0]
    probas_arr = probas_test[y_test == 1]

    y_pred_test = (probas_test >= umbral_clinico).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    
    esp = tn / (tn + fp) if (tn + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average="macro")
    brier = brier_score_loss(y_test, probas_test)

    print(f"\nResultados del Modelo sobre el conjunto de TEST (Datos nunca vistos):")
    print(f"Brier Score (Calibración)             : {brier:.4f} (Óptimo < 0.05)")
    print(f"Exactitud Global (Accuracy)           : {acc * 100:.2f}%")
    print(f"F1-Score Macro                        : {f1 * 100:.2f}%")
    print(f"Especificidad (Bebés Sanos Detectados): {tn}/{tn+fp} ({esp * 100:.1f}%)")
    print(f"Sensibilidad  (Arritmias Detectadas)  : {tp}/{tp+fn} ({sens * 100:.1f}%)")

    # 6. Exportar artefactos actualizados
    joblib.dump(modelo_calibrado, MODELS_DIR / "detector_arritmias_fetal.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler_fhrv.pkl")
    joblib.dump(features, MODELS_DIR / "feature_names.pkl")
    joblib.dump(umbral_clinico, MODELS_DIR / "decision_threshold.pkl")

    print(f"\n[OK] Artefactos calibrados exportados en: {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    df, features = cargar_dataset(CSV_PATH)
    entrenar_modelo_calibrado(df, features)