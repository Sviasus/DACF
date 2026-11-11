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

df_stats.to_csv(RESULTADOS / "analisis_estadistico_shapiro_cohen_3clases.csv", index=False)
print(f"\n[OK] Análisis exportado a: {RESULTADOS / 'analisis_estadistico_shapiro_cohen_3clases.csv'}")

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

    n_cols = len(columnas_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    # Paleta de colores: Verde (Normal), Azul (Bradicardia), Rojo (Taquicardia)
    colores = ['#4CAF50', '#2196F3', '#F44336']

    for i, col in enumerate(columnas_plot):
        # Ignorar grupos vacíos si la muestra extraída no tiene alguno de los 3
        data_to_plot = [sanos[col].values, bradicardias[col].values, taquicardias[col].values]
        
        # Crear boxplot habilitando patch_artist para poder colorear
        bplot = axes[i].boxplot(
            data_to_plot, 
            patch_artist=True, 
            tick_labels=['Normal', 'Bradicardia', 'Taquicardia'], 
            showmeans=True
        )
        
        # Aplicar colores a cada caja
        for patch, color in zip(bplot['boxes'], colores):
            patch.set_facecolor(color)
            patch.set_alpha(0.7) # Transparencia para ver la media/mediana

        axes[i].set_title(col, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)
        # Rotar etiquetas para que no se superpongan
        axes[i].tick_params(axis='x', rotation=25)

    plt.suptitle(titulo, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTADOS / nombre_archivo, dpi=300, bbox_inches="tight")
    plt.show()

# Generar las gráficas actualizadas
crear_boxplot_3_clases(FEATURES_TEMPORALES, "Comparación 3 Clases - Temporales", "07_boxplot_comp_temporales.png")
crear_boxplot_3_clases(FEATURES_FRECUENCIALES, "Comparación 3 Clases - Frecuenciales", "08_boxplot_comp_frecuenciales.png")
crear_boxplot_3_clases(FEATURES_NO_LINEALES, "Comparación 3 Clases - No Lineales", "09_boxplot_comp_no_lineales.png")

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
plt.figure(figsize=(12, 10))
# Usamos un mapa de colores 'coolwarm' para distinguir correlaciones positivas (rojo) y negativas (azul)
plt.imshow(corr, cmap="coolwarm", interpolation="nearest", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlación de Pearson")

# Ajustar etiquetas de los ejes
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Matriz de Correlación de Características", fontsize=16, fontweight="bold")

# Escribir los valores numéricos dentro de la matriz
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        valor = corr.iloc[i, j]
        if not np.isnan(valor):
            # Cambiar el color del texto a blanco si el fondo es muy oscuro para mejor lectura
            color_texto = "white" if abs(valor) > 0.6 else "black"
            plt.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8, color=color_texto)

plt.tight_layout()
plt.savefig(RESULTADOS / "10_matriz_correlacion.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# PARTE 6 — COMPARACIÓN BIDIMENSIONAL Y BARRAS POR CLASE
# ============================================================
print("\n" + "=" * 70)
print("PARTE 6 — DIAGRAMAS DE DISPERSIÓN Y BARRAS POR CLASE")
print("=" * 70)

# Definir las características más significativas a comparar
top_features = ["BPM_mean", "SDNN", "RMSSD", "LF_HF_ratio"]
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
    if f1 in top_disponibles and f2 in top_disponibles:
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
        plt.savefig(RESULTADOS / f"11_dispersion_{f1}_vs_{f2}.png", dpi=300, bbox_inches="tight")
        plt.show()

    # Cruce 2: Variabilidad Global (SDNN) vs Balance Simpatovagal (LF_HF_ratio)
    f3, f4 = "SDNN", "LF_HF_ratio"
    if f3 in top_disponibles and f4 in top_disponibles:
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
        plt.savefig(RESULTADOS / f"12_dispersion_{f3}_vs_{f4}.png", dpi=300, bbox_inches="tight")
        plt.show()

# ------------------------------------------------------------
# 2. DIAGRAMA DE BARRAS AGRUPADAS (MEDIAS POR CLASE)
# ------------------------------------------------------------
print("\nGenerando diagrama de barras comparativas...")

if top_disponibles:
    # Calcular medias para cada grupo ignorando NaN
    medias_sanos = [sanos[f].mean() if not sanos.empty else 0 for f in top_disponibles]
    medias_bradi = [bradicardias[f].mean() if not bradicardias.empty else 0 for f in top_disponibles]
    medias_taqui = [taquicardias[f].mean() if not taquicardias.empty else 0 for f in top_disponibles]

    # Calcular desviaciones estándar para las barras de error
    std_sanos = [sanos[f].std() if not sanos.empty else 0 for f in top_disponibles]
    std_bradi = [bradicardias[f].std() if not bradicardias.empty else 0 for f in top_disponibles]
    std_taqui = [taquicardias[f].std() if not taquicardias.empty else 0 for f in top_disponibles]

    x = np.arange(len(top_disponibles))
    width = 0.25  # Ancho de las barras

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar las barras con sus respectivos errores (desviación estándar)
    ax.bar(x - width, medias_sanos, width, yerr=std_sanos, label='Normal', color=colores[0], alpha=0.8, capsize=5, edgecolor='black')
    ax.bar(x, medias_bradi, width, yerr=std_bradi, label='Bradicardia', color=colores[1], alpha=0.8, capsize=5, edgecolor='black')
    ax.bar(x + width, medias_taqui, width, yerr=std_taqui, label='Taquicardia', color=colores[2], alpha=0.8, capsize=5, edgecolor='black')

    ax.set_ylabel('Valor Promedio')
    ax.set_title('Comparación de Características Clave por Clase', fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(top_disponibles, fontsize=12)
    ax.legend(title="Clase Clínica")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTADOS / "13_barras_comparativas.png", dpi=300, bbox_inches="tight")
    plt.show()

print("\n" + "=" * 70)
print("✓ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 70)