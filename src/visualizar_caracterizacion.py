# ============================================================
# VISUALIZACIÓN DE LA CARACTERIZACIÓN DE SEÑALES Y ANÁLISIS ESTADÍSTICO
# ============================================================

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, mannwhitneyu, ttest_ind  # type: ignore

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
FEATURES_TEMPORALES = [
    "BPM_mean", "SDNN", "RR_variance", "RR_min", "RR_max", "RR_range", "RMSSD", "pNN50", "pNN20"
]
FEATURES_ENERGIA_FORMA = ["Signal_Energy", "Signal_Power", "RR_skewness", "RR_kurtosis"]
FEATURES_FRECUENCIALES = ["VLF", "LF", "HF", "LF_HF_ratio"]
FEATURES_TIEMPO_FRECUENCIA = [
    "STFT_Power_Mean", "STFT_Power_Var", "STFT_LF_HF_Mean", "STFT_Spectral_Entropy"
]
FEATURES_NO_LINEALES = ["SD1", "SD2", "SD1_SD2_ratio", "SampEn", "DFA_alpha1"]
FEATURES = (
    FEATURES_TEMPORALES
    + FEATURES_ENERGIA_FORMA
    + FEATURES_FRECUENCIALES
    + FEATURES_TIEMPO_FRECUENCIA
    + FEATURES_NO_LINEALES
)

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
# FUNCIONES ESTADÍSTICAS (Shapiro-Wilk y Cohen's d) PARA 3 CLASES
# ============================================================
def cohens_d(grupo1, grupo2):
    n1, n2 = len(grupo1), len(grupo2)
    if n1 < 2 or n2 < 2: 
        return 0.0
    var1, var2 = np.var(grupo1, ddof=1), np.var(grupo2, ddof=1)
    # Varianza agrupada (pooled variance)
    s = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if s == 0: 
        return 0.0
    d = (np.mean(grupo1) - np.mean(grupo2)) / s
    return d

def evaluar_pares(feature, nombre_g1, data_g1, p_shap_g1, nombre_g2, data_g2, p_shap_g2):
    if len(data_g1) < 3 or len(data_g2) < 3:
        return None
        
    es_normal_g1 = p_shap_g1 > 0.05
    es_normal_g2 = p_shap_g2 > 0.05
    ambos_normales = es_normal_g1 and es_normal_g2
    
    # Elección de prueba estadística según normalidad
    if ambos_normales:
        stat, p_valor = ttest_ind(data_g1, data_g2, equal_var=False)
        prueba_usada = "T-Test (Welch)"
    else:
        stat, p_valor = mannwhitneyu(data_g1, data_g2, alternative='two-sided')
        prueba_usada = "Mann-Whitney U"
        
    # Tamaño del efecto (Cohen's d)
    d_cohen = cohens_d(data_g1, data_g2)
    
    return {
        "Característica": feature,
        "Comparación": f"{nombre_g1} vs {nombre_g2}",
        "Prueba": prueba_usada,
        "P-valor": round(p_valor, 4),
        "Cohen's d": round(d_cohen, 4),
        "Relevancia": "Alta" if abs(d_cohen) > 0.8 else ("Media" if abs(d_cohen) > 0.5 else "Baja")
    }

print("\n" + "=" * 70)
print("PARTE 1 — ANÁLISIS DE NORMALIDAD Y COMPARACIÓN DE 3 CLASES")
print("=" * 70)

resultados_estadisticos = []

# Separar por clases y subdividir la patológica
sanos = df_plot[df_plot["Target"] == 0]
patologicos = df_plot[df_plot["Target"] == 1]

# Usamos BPM_mean para identificar bradicardia (<115) y taquicardia (>=115)
if "BPM_mean" in df_plot.columns:
    bradicardias = patologicos[patologicos["BPM_mean"] < 115]
    taquicardias = patologicos[patologicos["BPM_mean"] >= 115]
else:
    bradicardias = patologicos
    taquicardias = pd.DataFrame(columns=patologicos.columns)

for feature in features_disponibles:
    data_sanos = sanos[feature].values
    data_bradi = bradicardias[feature].values
    data_taqui = taquicardias[feature].values
    
    # Pruebas de Shapiro-Wilk
    p_shap_sanos = shapiro(data_sanos)[1] if len(data_sanos) >= 3 else 0
    p_shap_bradi = shapiro(data_bradi)[1] if len(data_bradi) >= 3 else 0
    p_shap_taqui = shapiro(data_taqui)[1] if len(data_taqui) >= 3 else 0
    
    # Comparación 1: Normal vs Bradicardia
    res_bradi = evaluar_pares(feature, "Normal", data_sanos, p_shap_sanos, "Bradicardia", data_bradi, p_shap_bradi)
    if res_bradi: 
        resultados_estadisticos.append(res_bradi)
        
    # Comparación 2: Normal vs Taquicardia
    res_taqui = evaluar_pares(feature, "Normal", data_sanos, p_shap_sanos, "Taquicardia", data_taqui, p_shap_taqui)
    if res_taqui: 
        resultados_estadisticos.append(res_taqui)

# Convertir resultados a DataFrame y mostrar
df_stats = pd.DataFrame(resultados_estadisticos)
print("\n--- Resultados del Análisis Estadístico (3 Clases) ---")
print(df_stats.to_string(index=False))

df_stats.to_csv(RESULTADOS / "analisis_estadistico_shapiro_cohen_3clases_nuevo.csv", index=False)
print(f"\n[OK] Análisis exportado a: {RESULTADOS / 'analisis_estadistico_shapiro_cohen_3clases_nuevo.csv'}")

# ============================================================
# PARTE 2 — BOXPLOTS COMPARATIVOS POR 3 CLASES (CON COLORES)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 2 — GENERANDO BOXPLOTS COMPARATIVOS POR 3 CLASES")
print("=" * 70)

# Separar en 3 grupos clínicos basados en Target y BPM_mean
# Feto normal: ~110 a 160 BPM. 
sanos = df_plot[df_plot["Target"] == 0]
patologicos = df_plot[df_plot["Target"] == 1]

# Subdividir el grupo patológico en Bradicardia y Taquicardia/Arritmia rápida
bradicardias = patologicos[patologicos["BPM_mean"] < 115] # Ajusta el umbral si es necesario
taquicardias = patologicos[patologicos["BPM_mean"] >= 115]

print(f"Muestras Normales    : {len(sanos)}")
print(f"Muestras Bradicardia : {len(bradicardias)}")
print(f"Muestras Taquicardia : {len(taquicardias)}")

def crear_boxplot_3_clases(columnas, titulo, nombre_archivo):
    columnas_plot = [c for c in columnas if c in features_disponibles]
    if not columnas_plot:
        return

    n_total = len(columnas_plot)
    n_cols = min(5, n_total)
    n_rows = int(np.ceil(n_total / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5.2 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    # Paleta de colores: Verde (Normal), Azul (Bradicardia), Rojo (Taquicardia)
    colores = ['#4CAF50', '#2196F3', '#F44336']

    for i, col in enumerate(columnas_plot):
        data_to_plot = [sanos[col].values, bradicardias[col].values, taquicardias[col].values]

        bplot = axes_flat[i].boxplot(
            data_to_plot,
            patch_artist=True,
            tick_labels=['Normal', 'Bradicardia', 'Taquicardia'],
            showmeans=True
        )

        for patch, color in zip(bplot['boxes'], colores):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes_flat[i].set_title(col, fontweight="bold")
        axes_flat[i].grid(axis="y", alpha=0.3)
        axes_flat[i].tick_params(axis='x', rotation=25)

    for j in range(n_total, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(titulo, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTADOS / nombre_archivo, dpi=300, bbox_inches="tight")
    plt.show()

# Generar las gráficas actualizadas con sufijo _nuevo
crear_boxplot_3_clases(
    FEATURES_TEMPORALES,
    "Comparación 3 Clases - Temporales (Incl. Varianza, Min, Max, Rango)",
    "07_boxplot_comp_temporales_nuevo.png",
)
crear_boxplot_3_clases(
    FEATURES_ENERGIA_FORMA,
    "Comparación 3 Clases - Energía, Potencia, Asimetría y Curtosis",
    "14_boxplot_comp_energia_forma_nuevo.png",
)
crear_boxplot_3_clases(
    FEATURES_FRECUENCIALES,
    "Comparación 3 Clases - Frecuenciales (Welch)",
    "08_boxplot_comp_frecuenciales_nuevo.png",
)
crear_boxplot_3_clases(
    FEATURES_TIEMPO_FRECUENCIA,
    "Comparación 3 Clases - Tiempo-Frecuencia (STFT)",
    "15_boxplot_comp_stft_tiempo_frecuencia_nuevo.png",
)
crear_boxplot_3_clases(
    FEATURES_NO_LINEALES,
    "Comparación 3 Clases - No Lineales",
    "09_boxplot_comp_no_lineales_nuevo.png",
)

# ============================================================
# PARTE 3 — MATRIZ DE CORRELACIÓN
# ============================================================
print("\n" + "=" * 70)
print("PARTE 3 — MATRIZ DE CORRELACIÓN")
print("=" * 70)

# Calcular correlación solo con las características disponibles (sin el Target)
df_corr = df_plot[features_disponibles].copy()
corr = df_corr.corr(method="pearson")

print("\nMatriz de correlación calculada.")

# Configurar la gráfica
plt.figure(figsize=(16, 14))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlación de Pearson")

# Ajustar etiquetas de los ejes
cols_corr = list(corr.columns)
corr_vals = corr.to_numpy(dtype=np.float64)
plt.xticks(range(len(cols_corr)), cols_corr, rotation=45, ha='right', fontsize=9)
plt.yticks(range(len(cols_corr)), cols_corr, fontsize=9)
plt.title("Matriz de Correlación de Características (Con Nuevas Métricas)", fontsize=16, fontweight="bold")

# Escribir los valores numéricos dentro de la matriz
for i in range(len(cols_corr)):
    for j in range(len(cols_corr)):
        valor = float(corr_vals[i, j])
        if not np.isnan(valor):
            color_texto = "white" if abs(valor) > 0.6 else "black"
            plt.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=6.5, color=color_texto)

plt.tight_layout()
plt.savefig(RESULTADOS / "10_matriz_correlacion_nuevo.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# PARTE 6 — COMPARACIÓN BIDIMENSIONAL Y BARRAS POR CLASE
# ============================================================
print("\n" + "=" * 70)
print("PARTE 6 — DIAGRAMAS DE DISPERSIÓN Y BARRAS POR CLASE")
print("=" * 70)

# Definir las características más significativas a comparar (incluyendo nuevas métricas)
top_features = ["BPM_mean", "SDNN", "RMSSD", "LF_HF_ratio", "RR_range", "STFT_Spectral_Entropy"]
top_disponibles = [f for f in top_features if f in df_plot.columns]

colores = ['#4CAF50', '#2196F3', '#F44336'] # Verde (Normal), Azul (Bradi), Rojo (Taqui)
etiquetas = ['Normal', 'Bradicardia', 'Taquicardia']
grupos = [sanos, bradicardias, taquicardias]

# ------------------------------------------------------------
# 1. DIAGRAMAS DE DISPERSIÓN (SCATTER PLOTS)
# ------------------------------------------------------------
print("\nGenerando diagramas de dispersión...")

if len(top_disponibles) >= 2:
    # Cruce 1: Frecuencia (BPM_mean) vs Variabilidad a corto plazo (RMSSD)
    f1, f2 = "BPM_mean", "RMSSD"
    if f1 in df_plot.columns and f2 in df_plot.columns:
        plt.figure(figsize=(9, 6))
        for grupo, color, etiqueta in zip(grupos, colores, etiquetas):
            if not grupo.empty:
                plt.scatter(grupo[f1], grupo[f2], c=color, label=etiqueta, alpha=0.7, edgecolors='k', s=60)
                
        plt.title(f"Dispersión Clínica: {f1} vs {f2}", fontsize=14, fontweight="bold")
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.legend(title="Clase Clínica")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTADOS / f"11_dispersion_{f1}_vs_{f2}_nuevo.png", dpi=300, bbox_inches="tight")
        plt.show()

    # Cruce 2: Variabilidad Global (SDNN) vs Balance Simpatovagal (LF_HF_ratio)
    f3, f4 = "SDNN", "LF_HF_ratio"
    if f3 in df_plot.columns and f4 in df_plot.columns:
        plt.figure(figsize=(9, 6))
        for grupo, color, etiqueta in zip(grupos, colores, etiquetas):
            if not grupo.empty:
                plt.scatter(grupo[f3], grupo[f4], c=color, label=etiqueta, alpha=0.7, edgecolors='k', s=60)
                
        plt.title(f"Dispersión Clínica: {f3} vs {f4}", fontsize=14, fontweight="bold")
        plt.xlabel(f3)
        plt.ylabel(f4)
        plt.legend(title="Clase Clínica")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTADOS / f"12_dispersion_{f3}_vs_{f4}_nuevo.png", dpi=300, bbox_inches="tight")
        plt.show()

    # Cruce 3 (Nuevo): Entropía Espectral STFT vs Asimetría RR (Skewness)
    f5, f6 = "STFT_Spectral_Entropy", "RR_skewness"
    if f5 in df_plot.columns and f6 in df_plot.columns:
        plt.figure(figsize=(9, 6))
        for grupo, color, etiqueta in zip(grupos, colores, etiquetas):
            if not grupo.empty:
                plt.scatter(grupo[f5], grupo[f6], c=color, label=etiqueta, alpha=0.7, edgecolors='k', s=60)
                
        plt.title(f"Dispersión Clínica (Nuevas Métricas): {f5} vs {f6}", fontsize=14, fontweight="bold")
        plt.xlabel(f5)
        plt.ylabel(f6)
        plt.legend(title="Clase Clínica")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTADOS / f"16_dispersion_{f5}_vs_{f6}_nuevo.png", dpi=300, bbox_inches="tight")
        plt.show()

# ------------------------------------------------------------
# 2. DIAGRAMA DE BARRAS AGRUPADAS (MEDIAS POR CLASE)
# ------------------------------------------------------------
print("\nGenerando diagrama de barras comparativas...")

bar_features = [f for f in ["BPM_mean", "SDNN", "RMSSD", "RR_range", "RR_kurtosis", "STFT_Power_Mean"] if f in df_plot.columns]
if bar_features:
    medias_sanos = [sanos[f].mean() if not sanos.empty else 0 for f in bar_features]
    medias_bradi = [bradicardias[f].mean() if not bradicardias.empty else 0 for f in bar_features]
    medias_taqui = [taquicardias[f].mean() if not taquicardias.empty else 0 for f in bar_features]

    std_sanos = [sanos[f].std() if not sanos.empty else 0 for f in bar_features]
    std_bradi = [bradicardias[f].std() if not bradicardias.empty else 0 for f in bar_features]
    std_taqui = [taquicardias[f].std() if not taquicardias.empty else 0 for f in bar_features]

    x = np.arange(len(bar_features))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, medias_sanos, width, yerr=std_sanos, label='Normal', color=colores[0], alpha=0.8, capsize=5, edgecolor='black')
    ax.bar(x, medias_bradi, width, yerr=std_bradi, label='Bradicardia', color=colores[1], alpha=0.8, capsize=5, edgecolor='black')
    ax.bar(x + width, medias_taqui, width, yerr=std_taqui, label='Taquicardia', color=colores[2], alpha=0.8, capsize=5, edgecolor='black')

    ax.set_ylabel('Valor Promedio')
    ax.set_title('Comparación de Características Clave por Clase (Incl. Nuevas Métricas)', fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_features, fontsize=11, rotation=15)
    ax.legend(title="Clase Clínica")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTADOS / "13_barras_comparativas_nuevo.png", dpi=300, bbox_inches="tight")
    plt.show()

print("\n" + "=" * 70)
print("✓ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 70)