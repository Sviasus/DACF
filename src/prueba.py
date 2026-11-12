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
print("ANÁLISIS ESTADÍSTICO Y VISUALIZACIÓN DE CARACTERÍSTICAS (BINARIO)")
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
    if n1 < 2 or n2 < 2: 
        return 0.0
    var1, var2 = np.var(grupo1, ddof=1), np.var(grupo2, ddof=1)
    s = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if s == 0: 
        return 0.0
    d = (np.mean(grupo1) - np.mean(grupo2)) / s
    return d

print("\n" + "=" * 70)
print("PARTE 1 — ANÁLISIS DE NORMALIDAD Y COMPARACIÓN DE 2 CLASES")
print("=" * 70)

resultados_estadisticos = []

# Separar por clases binarias
sanos = df_plot[df_plot["Target"] == 0]
patologicos = df_plot[df_plot["Target"] == 1]

for feature in features_disponibles:
    data_sanos = sanos[feature].values
    data_pato = patologicos[feature].values
    
    if len(data_sanos) < 3 or len(data_pato) < 3:
        continue

    # Prueba de normalidad de Shapiro-Wilk
    p_shap_sanos = shapiro(data_sanos)[1]
    p_shap_pato = shapiro(data_pato)[1]
    
    es_normal_sanos = p_shap_sanos > 0.05
    es_normal_pato = p_shap_pato > 0.05
    ambos_normales = es_normal_sanos and es_normal_pato
    
    # Elección de prueba estadística según normalidad
    if ambos_normales:
        stat, p_valor = ttest_ind(data_sanos, data_pato, equal_var=False)
        prueba_usada = "T-Test (Welch)"
    else:
        stat, p_valor = mannwhitneyu(data_sanos, data_pato, alternative='two-sided')
        prueba_usada = "Mann-Whitney U"
        
    # Tamaño del efecto (Cohen's d)
    d_cohen = cohens_d(data_sanos, data_pato)
    
    resultados_estadisticos.append({
        "Característica": feature,
        "Norm. Sanos (p-val)": round(p_shap_sanos, 4),
        "Norm. Enfermos (p-val)": round(p_shap_pato, 4),
        "Prueba Comparativa": prueba_usada,
        "P-valor (Diferencia)": round(p_valor, 4),
        "Cohen's d": round(d_cohen, 4),
        "Relevancia": "Alta" if abs(d_cohen) > 0.8 else ("Media" if abs(d_cohen) > 0.5 else "Baja")
    })

# Convertir resultados a DataFrame y mostrar
df_stats = pd.DataFrame(resultados_estadisticos)
print("\n--- Resultados del Análisis Estadístico (Sano vs Enfermo) ---")
print(df_stats.to_string(index=False))

# Convertir 'Relevancia' a categoría ordenada y ordenar el DataFrame
orden = ['Alta', 'Media', 'Baja']
df_stats['Relevancia'] = pd.Categorical(df_stats['Relevancia'], categories=orden, ordered=True)
df_stats['Abs_Cohen'] = df_stats["Cohen's d"].abs()
df_stats = df_stats.sort_values(by=['Relevancia', 'Abs_Cohen'], ascending=[True, False]).drop(columns=['Abs_Cohen'])

df_stats.to_csv(RESULTADOS / "analisis_estadistico_shapiro_cohen_binario.csv", index=False)
print(f"\n[OK] Análisis exportado a: {RESULTADOS / 'analisis_estadistico_shapiro_cohen_binario.csv'}")

# ============================================================
# PARTE 2 — BOXPLOTS COMPARATIVOS BINARIOS (NORMAL VS ENFERMO)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 2 — GENERANDO BOXPLOTS COMPARATIVOS BINARIOS")
print("=" * 70)

def crear_boxplot_binario(columnas, titulo, nombre_archivo):
    columnas_plot = [c for c in columnas if c in features_disponibles]
    if not columnas_plot:
        return

    n_total = len(columnas_plot)
    n_cols = min(5, n_total)
    n_rows = int(np.ceil(n_total / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5.2 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    # Colores: Verde (Normal), Naranja (Enfermo)
    colores = ['#4CAF50', '#FF9800']

    for i, col in enumerate(columnas_plot):
        data_to_plot = [sanos[col].values, patologicos[col].values]

        bplot = axes_flat[i].boxplot(
            data_to_plot,
            patch_artist=True,
            tick_labels=['Normal (0)', 'Enfermo (1)'],
            showmeans=True
        )

        for patch, color in zip(bplot['boxes'], colores):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes_flat[i].set_title(col, fontweight="bold")
        axes_flat[i].grid(axis="y", alpha=0.3)
        axes_flat[i].tick_params(axis='x', rotation=15)

    for j in range(n_total, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(titulo, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTADOS / nombre_archivo, dpi=300, bbox_inches="tight")
    plt.show()

# Generar gráficas binarias
crear_boxplot_binario(FEATURES_TEMPORALES, "Comparación Binaria - Temporales", "07_boxplot_comp_temporales_binario.png")
crear_boxplot_binario(FEATURES_ENERGIA_FORMA, "Comparación Binaria - Energía y Forma", "14_boxplot_comp_energia_forma_binario.png")
crear_boxplot_binario(FEATURES_FRECUENCIALES, "Comparación Binaria - Frecuenciales", "08_boxplot_comp_frecuenciales_binario.png")
crear_boxplot_binario(FEATURES_TIEMPO_FRECUENCIA, "Comparación Binaria - STFT", "15_boxplot_comp_stft_binario.png")
crear_boxplot_binario(FEATURES_NO_LINEALES, "Comparación Binaria - No Lineales", "09_boxplot_comp_no_lineales_binario.png")


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
plt.title("Matriz de Correlación de Características", fontsize=16, fontweight="bold")

# Escribir los valores numéricos dentro de la matriz
for i in range(len(cols_corr)):
    for j in range(len(cols_corr)):
        valor = float(corr_vals[i, j])
        if not np.isnan(valor):
            color_texto = "white" if abs(valor) > 0.6 else "black"
            plt.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=6.5, color=color_texto)

plt.tight_layout()
plt.savefig(RESULTADOS / "10_matriz_correlacion_binario.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# PARTE 6 — COMPARACIÓN BIDIMENSIONAL Y BARRAS BINARIAS
# ============================================================
print("\n" + "=" * 70)
print("PARTE 6 — DIAGRAMAS DE DISPERSIÓN Y BARRAS BINARIAS")
print("=" * 70)

top_features = ["BPM_mean", "SDNN", "RMSSD", "LF_HF_ratio", "RR_range", "STFT_Spectral_Entropy"]
top_disponibles = [f for f in top_features if f in df_plot.columns]

colores_bin = ['#4CAF50', '#FF9800'] 
etiquetas_bin = ['Normal', 'Enfermo']
grupos_bin = [sanos, patologicos]

print("\nGenerando diagramas de dispersión binarios...")
if len(top_disponibles) >= 2:
    f1, f2 = "BPM_mean", "RMSSD"
    if f1 in df_plot.columns and f2 in df_plot.columns:
        plt.figure(figsize=(9, 6))
        for grupo, color, etiqueta in zip(grupos_bin, colores_bin, etiquetas_bin):
            if not grupo.empty:
                plt.scatter(grupo[f1], grupo[f2], c=color, label=etiqueta, alpha=0.7, edgecolors='k', s=60)
        plt.title(f"Dispersión Binaria: {f1} vs {f2}", fontsize=14, fontweight="bold")
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.legend(title="Clase Clínica")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTADOS / f"11_dispersion_{f1}_vs_{f2}_binario.png", dpi=300)
        plt.show()

    f3, f4 = "SDNN", "LF_HF_ratio"
    if f3 in df_plot.columns and f4 in df_plot.columns:
        plt.figure(figsize=(9, 6))
        for grupo, color, etiqueta in zip(grupos_bin, colores_bin, etiquetas_bin):
            if not grupo.empty:
                plt.scatter(grupo[f3], grupo[f4], c=color, label=etiqueta, alpha=0.7, edgecolors='k', s=60)
        plt.title(f"Dispersión Binaria: {f3} vs {f4}", fontsize=14, fontweight="bold")
        plt.xlabel(f3)
        plt.ylabel(f4)
        plt.legend(title="Clase Clínica")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTADOS / f"12_dispersion_{f3}_vs_{f4}_binario.png", dpi=300)
        plt.show()

print("\nGenerando diagrama de barras comparativas binario...")
bar_features = [f for f in ["BPM_mean", "SDNN", "RMSSD", "RR_range", "RR_kurtosis", "STFT_Power_Mean"] if f in df_plot.columns]
if bar_features:
    medias_sanos = [sanos[f].mean() if not sanos.empty else 0 for f in bar_features]
    medias_pato = [patologicos[f].mean() if not patologicos.empty else 0 for f in bar_features]
    
    std_sanos = [sanos[f].std() if not sanos.empty else 0 for f in bar_features]
    std_pato = [patologicos[f].std() if not patologicos.empty else 0 for f in bar_features]

    x = np.arange(len(bar_features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, medias_sanos, width, yerr=std_sanos, label='Normal', color=colores_bin[0], alpha=0.8, capsize=5, edgecolor='black')
    ax.bar(x + width/2, medias_pato, width, yerr=std_pato, label='Enfermo', color=colores_bin[1], alpha=0.8, capsize=5, edgecolor='black')

    ax.set_ylabel('Valor Promedio')
    ax.set_title('Comparación de Características Clave (Sano vs Enfermo)', fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_features, fontsize=11, rotation=15)
    ax.legend(title="Clase Clínica")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTADOS / "13_barras_comparativas_binario.png", dpi=300)
    plt.show()

print("\n" + "=" * 70)
print("✓ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 70)