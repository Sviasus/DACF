# ============================================================
# VISUALIZACIÓN DEL PIPELINE
# Detector de Arritmias Cardíacas Fetales
#
# Este archivo NO modifica el detector.
# Utiliza las funciones existentes de fqrs_detector.py
# para visualizar las etapas del procesamiento.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import wfdb

# Importamos las funciones REALES del proyecto
from fqrs_detector import (
    cancelar_complejos_maternos,
    aislar_componente_fecg,
    detector_pan_tompkins_fetal,
)


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

REGISTRO = "a01"

# Canal que utilizaremos para las gráficas iniciales
CANAL = 0

# Segundos que queremos visualizar
DURACION_SEGUNDOS = 10


# ============================================================
# 2. UBICACIÓN DEL PROYECTO
# ============================================================

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

DATA_DIR = ROOT / "data" / "cinc2013_real" / "set-a"

RESULTADOS = ROOT / "resultados_preprocesamiento"

RESULTADOS.mkdir(exist_ok=True)


print("=" * 70)
print("VISUALIZACIÓN DEL PIPELINE")
print("DETECTOR DE ARRITMIAS CARDÍACAS FETALES")
print("=" * 70)

print(f"\nRegistro: {REGISTRO}")
print(f"Datos: {DATA_DIR}")


# ============================================================
# 3. COMPROBAR ARCHIVOS
# ============================================================

archivo_hea = DATA_DIR / f"{REGISTRO}.hea"
archivo_dat = DATA_DIR / f"{REGISTRO}.dat"

if not archivo_hea.exists():
    print("\n❌ No existe:")
    print(archivo_hea)
    raise SystemExit

if not archivo_dat.exists():
    print("\n❌ No existe:")
    print(archivo_dat)
    raise SystemExit

print("\n✓ Archivos encontrados.")


# ============================================================
# 4. CARGAR REGISTRO
# ============================================================

print("\nCargando registro...")

registro = wfdb.rdrecord(
    str(DATA_DIR / REGISTRO)
)

fs = float(registro.fs)

signals = registro.p_signal

print("✓ Registro cargado.")

print(f"\nFrecuencia de muestreo: {fs} Hz")
print(f"Número de muestras: {signals.shape[0]}")
print(f"Número de canales: {signals.shape[1]}")

if registro.sig_name:
    print(f"Canales: {registro.sig_name}")


# ============================================================
# 5. SELECCIONAR LOS 10 SEGUNDOS DE EJEMPLO
# ============================================================

n_muestras = min(
    int(DURACION_SEGUNDOS * fs),
    signals.shape[0]
)

signals_demo = signals[:n_muestras, :]

tiempo = np.arange(n_muestras) / fs

print(
    f"\nDuración utilizada: "
    f"{n_muestras / fs:.2f} segundos"
)


# ============================================================
# ============================================================
#                 PARTE A
#          PREPROCESAMIENTO VISUAL
# ============================================================
# ============================================================

print("\n")
print("=" * 70)
print("PARTE A — PREPROCESAMIENTO")
print("=" * 70)


# ------------------------------------------------------------
# IMPORTANTE:
#
# Aquí usamos el preprocesamiento REAL del proyecto.
# No implementamos filtros diferentes.
# ------------------------------------------------------------

from preprocessing import preprocesar_senal_multicanal


print("\nAplicando preprocesamiento real del proyecto...")

sig_filtrada = preprocesar_senal_multicanal(
    signals_demo,
    fs=fs,
    aplicar_notch=True
)

print("✓ Preprocesamiento terminado.")


# ============================================================
# 6. GRÁFICA — SEÑAL ORIGINAL VS PREPROCESADA
# ============================================================

senal_original = signals_demo[:, CANAL]

senal_preprocesada = sig_filtrada[:, CANAL]


fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 8),
    sharex=True
)

# Señal original

axes[0].plot(
    tiempo,
    senal_original
)

axes[0].set_title(
    f"Señal abdominal original — {REGISTRO}",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_ylabel("Amplitud")

axes[0].grid(alpha=0.3)


# Señal preprocesada

axes[1].plot(
    tiempo,
    senal_preprocesada
)

axes[1].set_title(
    "Señal después del preprocesamiento",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_xlabel("Tiempo (s)")
axes[1].set_ylabel("Amplitud")

axes[1].grid(alpha=0.3)


fig.suptitle(
    f"Comparación del preprocesamiento — {REGISTRO}",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.savefig(
    RESULTADOS /
    f"01_original_vs_preprocesada_{REGISTRO}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ============================================================
#              PARTE B
#        CANCELACIÓN MATERNA
# ============================================================
# ============================================================

print("\n")
print("=" * 70)
print("PARTE B — CANCELACIÓN DE COMPLEJOS MATERNOS")
print("=" * 70)


print("\nDetectando y cancelando complejos maternos...")


sig_sin_materno, picos_maternos = (
    cancelar_complejos_maternos(
        sig_filtrada,
        fs=fs
    )
)


print("✓ Cancelación materna terminada.")

print(
    f"Complejos maternos detectados: "
    f"{len(picos_maternos)}"
)


# ============================================================
# 7. GRÁFICA — CANCELACIÓN MATERNA
# ============================================================

senal_antes_materno = sig_filtrada[:, CANAL]

senal_despues_materno = sig_sin_materno[:, CANAL]


fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 8),
    sharex=True
)


# Antes

axes[0].plot(
    tiempo,
    senal_antes_materno
)

axes[0].set_title(
    "Antes de la cancelación de complejos maternos",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_ylabel("Amplitud")

axes[0].grid(alpha=0.3)


# Después

axes[1].plot(
    tiempo,
    senal_despues_materno
)

if len(picos_maternos) > 0:

    picos_visibles = [
        p for p in picos_maternos
        if p < len(senal_despues_materno)
    ]

    if len(picos_visibles) > 0:

        axes[1].scatter(
            np.array(picos_visibles) / fs,
            senal_despues_materno[picos_visibles],
            marker="o",
            label="mQRS detectados"
        )

        axes[1].legend()


axes[1].set_title(
    "Después de la cancelación de complejos maternos",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_xlabel("Tiempo (s)")
axes[1].set_ylabel("Amplitud")

axes[1].grid(alpha=0.3)


fig.suptitle(
    "Cancelación adaptativa de complejos maternos",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.savefig(
    RESULTADOS /
    f"02_cancelacion_materna_{REGISTRO}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ============================================================
#                 PARTE C
#              FastICA
# ============================================================
# ============================================================

print("\n")
print("=" * 70)
print("PARTE C — AISLAMIENTO DE LA COMPONENTE FETAL")
print("=" * 70)


print("\nEjecutando FastICA...")


fecg_aislado, fuentes, idx_fetal = (
    aislar_componente_fecg(
        sig_sin_materno,
        fs=fs
    )
)


print("✓ FastICA terminado.")

print(
    f"Componente fetal seleccionada: "
    f"{idx_fetal}"
)

print(
    f"Número de componentes: "
    f"{fuentes.shape[1]}"
)


# ============================================================
# 8. GRÁFICA — COMPONENTES ICA
# ============================================================

fig, axes = plt.subplots(
    fuentes.shape[1],
    1,
    figsize=(14, 3 * fuentes.shape[1]),
    sharex=True
)


# Cuando solamente existe una componente,
# matplotlib no devuelve una lista de axes.

if fuentes.shape[1] == 1:
    axes = [axes]


for i in range(fuentes.shape[1]):

    axes[i].plot(
        tiempo,
        fuentes[:, i]
    )

    if i == idx_fetal:

        axes[i].set_title(
            f"Componente ICA {i} — COMPONENTE FETAL SELECCIONADA",
            fontweight="bold"
        )

    else:

        axes[i].set_title(
            f"Componente ICA {i}"
        )

    axes[i].set_ylabel(
        "Amplitud"
    )

    axes[i].grid(
        alpha=0.3
    )


axes[-1].set_xlabel(
    "Tiempo (s)"
)


fig.suptitle(
    "Componentes obtenidas mediante FastICA",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.savefig(
    RESULTADOS /
    f"03_componentes_ICA_{REGISTRO}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. GRÁFICA — fECG AISLADO
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    tiempo,
    fecg_aislado
)

plt.title(
    "Componente fECG fetal aislada mediante FastICA",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    "Tiempo (s)"
)

plt.ylabel(
    "Amplitud"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTADOS /
    f"04_fECG_aislado_{REGISTRO}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ============================================================
#              PARTE D
#          PAN-TOMPKINS FETAL
# ============================================================
# ============================================================

print("\n")
print("=" * 70)
print("PARTE D — DETECCIÓN DE fQRS")
print("=" * 70)


print("\nEjecutando detector Pan-Tompkins fetal...")

picos_fqrs = detector_pan_tompkins_fetal(
    fecg_aislado,
    fs=fs
)


print("✓ Detección fQRS terminada.")

print(
    f"Complejos fQRS detectados: "
    f"{len(picos_fqrs)}"
)


# ============================================================
# 10. RECONSTRUIR LAS ETAPAS INTERNAS
#     EXACTAMENTE COMO ESTÁN EN fqrs_detector.py
# ============================================================

from scipy.signal import butter, filtfilt, find_peaks


# ------------------------------------------------------------
# ETAPA 1 — FILTRO 10–35 Hz
# ------------------------------------------------------------

nyq = 0.5 * fs

low = 10.0 / nyq

high = min(
    35.0 / nyq,
    0.99
)

b, a = butter(
    2,
    [low, high],
    btype="bandpass"
)

sig_band = filtfilt(
    b,
    a,
    fecg_aislado
)


# ------------------------------------------------------------
# ETAPA 2 — DERIVADA
# ------------------------------------------------------------

sig_diff = np.gradient(
    sig_band
)


# ------------------------------------------------------------
# ETAPA 3 — ELEVACIÓN AL CUADRADO
# ------------------------------------------------------------

sig_sq = (
    sig_diff ** 2
)


# ------------------------------------------------------------
# ETAPA 4 — INTEGRACIÓN 70 ms
# ------------------------------------------------------------

w_len = max(
    1,
    int(0.07 * fs)
)

kernel = (
    np.ones(w_len)
    / w_len
)

sig_integ = np.convolve(
    sig_sq,
    kernel,
    mode="same"
)


# ------------------------------------------------------------
# ETAPA 5 — VENTANAS ADAPTATIVAS DE 3 SEGUNDOS
# ------------------------------------------------------------

dist_minima = int(
    0.24 * fs
)

win_size = int(
    3.0 * fs
)

todos_picos_integ = []


for start in range(
    0,
    len(fecg_aislado),
    win_size
):

    end = min(
        start + win_size,
        len(fecg_aislado)
    )

    segmento = sig_integ[
        start:end
    ]

    if len(segmento) < dist_minima:
        continue

    mediana_local = np.median(
        segmento
    )

    max_local = np.percentile(
        segmento,
        95
    )

    umbral_local = (
        mediana_local
        + 0.25
        * (
            max_local
            - mediana_local
        )
    )

    picos_seg, _ = find_peaks(
        segmento,
        height=umbral_local,
        distance=dist_minima
    )

    todos_picos_integ.extend(
        start + picos_seg
    )


todos_picos_integ = sorted(
    list(
        set(
            todos_picos_integ
        )
    )
)


# ============================================================
# 11. GRÁFICAS DEL PAN-TOMPKINS
# ============================================================


# ------------------------------------------------------------
# GRÁFICA 5 — FILTRO 10–35 Hz
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 8),
    sharex=True
)

axes[0].plot(
    tiempo,
    fecg_aislado
)

axes[0].set_title(
    "Entrada: fECG aislado",
    fontweight="bold"
)

axes[0].set_ylabel(
    "Amplitud"
)

axes[0].grid(alpha=0.3)


axes[1].plot(
    tiempo,
    sig_band
)

axes[1].set_title(
    "Después del filtro pasa-banda 10–35 Hz",
    fontweight="bold"
)

axes[1].set_xlabel(
    "Tiempo (s)"
)

axes[1].set_ylabel(
    "Amplitud"
)

axes[1].grid(alpha=0.3)


fig.suptitle(
    "Pan-Tompkins — Filtro pasa-banda",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.savefig(
    RESULTADOS /
    f"05_filtro_pan_tompkins_{REGISTRO}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# GRÁFICA 6 — DERIVADA
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    tiempo,
    sig_diff
)

plt.title(
    "Pan-Tompkins — Derivada",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    "Tiempo (s)"
)

plt.ylabel(
    "Amplitud"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTADOS /
    f"06_derivada_{REGISTRO}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# GRÁFICA 7 — ELEVACIÓN AL CUADRADO
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    tiempo,
    sig_sq
)

plt.title(
    "Pan-Tompkins — Elevación al cuadrado",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    "Tiempo (s)"
)

plt.ylabel(
    "Amplitud²"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTADOS /
    f"07_elevacion_cuadrado_{REGISTRO}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# GRÁFICA 8 — INTEGRACIÓN
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    tiempo,
    sig_integ
)

plt.title(
    "Pan-Tompkins — Integración por media móvil (~70 ms)",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    "Tiempo (s)"
)

plt.ylabel(
    "Energía integrada"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTADOS /
    f"08_integracion_{REGISTRO}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 12. GRÁFICA 9 — PUNTOS DETECTADOS EN LA SEÑAL INTEGRADA
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    tiempo,
    sig_integ,
    label="Señal integrada"
)

if len(todos_picos_integ) > 0:

    tiempos_picos_integracion = (
        np.array(todos_picos_integ)
        / fs
    )

    amplitudes_picos_integracion = (
        sig_integ[todos_picos_integ]
    )

    plt.scatter(
        tiempos_picos_integracion,
        amplitudes_picos_integracion,
        marker="o",
        label="Picos candidatos"
    )


plt.title(
    "Pan-Tompkins — Detección adaptativa de candidatos",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    "Tiempo (s)"
)

plt.ylabel(
    "Señal integrada"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTADOS /
    f"09_candidatos_qrs_{REGISTRO}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. RESULTADO FINAL — fQRS
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    tiempo,
    fecg_aislado,
    label="fECG fetal aislado"
)


if len(picos_fqrs) > 0:

    picos_visibles = [
        p for p in picos_fqrs
        if p < len(fecg_aislado)
    ]

    tiempos_fqrs = (
        np.array(picos_visibles)
        / fs
    )

    amplitudes_fqrs = (
        fecg_aislado[picos_visibles]
    )

    plt.scatter(
        tiempos_fqrs,
        amplitudes_fqrs,
        marker="o",
        s=50,
        label="fQRS detectados"
    )


plt.title(
    "Resultado final — Complejos fQRS detectados",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Tiempo (s)"
)

plt.ylabel(
    "Amplitud"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTADOS /
    f"10_fQRS_finales_{REGISTRO}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. RESUMEN
# ============================================================

print("\n")
print("=" * 70)
print("✓ PIPELINE COMPLETO VISUALIZADO")
print("=" * 70)

print(
    f"\nRegistro utilizado: {REGISTRO}"
)

print(
    f"Frecuencia de muestreo: {fs} Hz"
)

print(
    f"Duración analizada: "
    f"{n_muestras / fs:.2f} s"
)

print(
    f"Complejos maternos detectados: "
    f"{len(picos_maternos)}"
)

print(
    f"Componente fetal seleccionada: "
    f"{idx_fetal}"
)

print(
    f"Complejos fQRS detectados: "
    f"{len(picos_fqrs)}"
)

print(
    "\nResultados guardados en:"
)

print(RESULTADOS)

print("\nArchivos:")

for archivo in sorted(
    RESULTADOS.glob("*.png")
):

    print(
        f"  ✓ {archivo.name}"
    )

print("\n")