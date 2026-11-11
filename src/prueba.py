# ============================================================
# VISUALIZACIÓN DE LA CARACTERIZACIÓN DE SEÑALES Y ANÁLISIS ESTADÍSTICO
# ============================================================

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, mannwhitneyu, ttest_ind

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "dataset_features.csv"
RESULTADOS = PROJECT_ROOT / "resultados_caracterizacion"
RESULTADOS.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACIÓN DE CARACTERÍSTICAS
# ============================================================
MAX_MUESTRAS = 100
FEATURES_TEMPORALES = ["BPM_mean", "SDNN", "RMSSD", "pNN50"]
FEATURES_FRECUENCIALES = ["VLF", "LF", "HF", "LF_HF_ratio"]
FEATURES_NO_LINEALES = ["SD1", "SD2", "SD1_SD2_ratio", "SampEn", "DFA_alpha1"]
FEATURES = FEATURES_TEMPORALES + FEATURES_FRECUENCIALES + FEATURES_NO_LINEALES

print("=" * 70)
print("ANÁLISIS ESTADÍSTICO Y VISUALIZACIÓN DE CARACTERÍSTICAS")
print("=" * 70)

# ============================================================
# COMPROBAR Y CARGAR DATASET
# ============================================================
if not CSV_PATH.exists():
    print(f"\n❌ No se encontró el dataset: {CSV_PATH}")
    raise SystemExit

df = pd.read_csv(CSV_PATH)
features_disponibles = [f for f in FEATURES if f in df.columns]

# Asegurar limpieza numérica
df_features = df[features_disponibles + ["Target"]].copy()
df_features[features_disponibles] = df_features[features_disponibles].apply(pd.to_numeric, errors="coerce")
df_features = df_features.dropna()

if len(df_features) > MAX_MUESTRAS:
    df_plot = df_features.sample(n=MAX_MUESTRAS, random_state=42)
else:
    df_plot = df_features.copy()

print(f"Muestras utilizadas para el análisis: {len(df_plot)}")

# ============================================================
# FUNCIONES ESTADÍSTICAS (Shapiro-Wilk y Cohen's d)
# ============================================================
def cohens_d(grupo1, grupo2):
    n1, n2 = len(grupo1), len(grupo2)
    var1, var2 = np.var(grupo1, ddof=1), np.var(grupo2, ddof=1)
    # Varianza agrupada (pooled variance)
    s = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(grupo1) - np.mean(grupo2)) / s
    return d

print("\n" + "=" * 70)
print("PARTE 1 — ANÁLISIS DE NORMALIDAD Y COMPARACIÓN DE CLASES")
print("=" * 70)

resultados_estadisticos = []

# Separar por clases: 0 = Normal, 1 = Arritmia
sanos = df_plot[df_plot["Target"] == 0]
arritmias = df_plot[df_plot["Target"] == 1]

for feature in features_disponibles:
    data_sanos = sanos[feature].values
    data_arritmias = arritmias[feature].values
    
    if len(data_sanos) < 3 or len(data_arritmias) < 3:
        continue

    # Prueba de normalidad de Shapiro-Wilk
    _, p_shapiro_sanos = shapiro(data_sanos)
    _, p_shapiro_arritmias = shapiro(data_arritmias)
    
    es_normal_sanos = p_shapiro_sanos > 0.05
    es_normal_arritmias = p_shapiro_arritmias > 0.05
    ambos_normales = es_normal_sanos and es_normal_arritmias
    
    # Elección de prueba estadística según normalidad
    if ambos_normales:
        stat, p_valor = ttest_ind(data_sanos, data_arritmias, equal_var=False)
        prueba_usada = "T-Test (Welch)"
    else:
        stat, p_valor = mannwhitneyu(data_sanos, data_arritmias, alternative='two-sided')
        prueba_usada = "Mann-Whitney U"
        
    # Tamaño del efecto (Cohen's d)
    d_cohen = cohens_d(data_sanos, data_arritmias)
    
    resultados_estadisticos.append({
        "Característica": feature,
        "Norm. Sanos (p-val)": round(p_shapiro_sanos, 4),
        "Norm. Arritmias (p-val)": round(p_shapiro_arritmias, 4),
        "Prueba Comparativa": prueba_usada,
        "P-valor (Diferencia)": round(p_valor, 4),
        "Cohen's d": round(d_cohen, 4),
        "Relevancia": "Alta" if abs(d_cohen) > 0.8 else ("Media" if abs(d_cohen) > 0.5 else "Baja")
    })

# Convertir resultados a DataFrame y mostrar
df_stats = pd.DataFrame(resultados_estadisticos)
print("\n--- Resultados del Análisis Estadístico ---")
print(df_stats.to_string(index=False))

df_stats.to_csv(RESULTADOS / "analisis_estadistico_shapiro_cohen.csv", index=False)
print(f"\n[OK] Análisis exportado a: {RESULTADOS / 'analisis_estadistico_shapiro_cohen.csv'}")

# ============================================================
# PARTE 2 — BOXPLOTS COMPARATIVOS POR CLASE
# ============================================================
print("\n" + "=" * 70)
print("PARTE 2 — GENERANDO BOXPLOTS COMPARATIVOS POR CLASE")
print("=" * 70)

def crear_boxplot_comparativo(columnas, titulo, nombre_archivo):
    columnas_plot = [c for c in columnas if c in features_disponibles]
    if not columnas_plot:
        return

    n_cols = len(columnas_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(columnas_plot):
        data_to_plot = [sanos[col].values, arritmias[col].values]
        # Corrección: Se usa tick_labels en lugar de labels
        axes[i].boxplot(data_to_plot, tick_labels=['Normal (0)', 'Arritmia (1)'], showmeans=True)
        axes[i].set_title(col, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)

    plt.suptitle(titulo, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTADOS / nombre_archivo, dpi=300, bbox_inches="tight")
    plt.show()

crear_boxplot_comparativo(FEATURES_TEMPORALES, "Comparación por Clases - Temporales", "07_boxplot_comp_temporales.png")
crear_boxplot_comparativo(FEATURES_FRECUENCIALES, "Comparación por Clases - Frecuenciales", "08_boxplot_comp_frecuenciales.png")
crear_boxplot_comparativo(FEATURES_NO_LINEALES, "Comparación por Clases - No Lineales", "09_boxplot_comp_no_lineales.png")

print("\n" + "=" * 70)
print("✓ PROCESO COMPLETADO")
print("=" * 70)