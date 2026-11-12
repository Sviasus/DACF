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
import wfdb  # type: ignore

# Importamos las funciones REALES del proyecto
from fqrs_detector import (
    cancelar_complejos_maternos,
    aislar_componente_fecg,
    detector_pan_tompkins_fetal,
    extraer_fqrs_optimo,
)
from preprocessing import preprocesar_senal_multicanal
from feature_extraction import (
    calcular_tacograma_rr,
    extraer_vector_caracteristicas_completo,
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

signals = np.asarray(registro.p_signal, dtype=np.float64)

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

# El preprocesamiento real se importa al inicio desde preprocessing.py.


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
    rect=(0.0, 0.0, 1.0, 0.95)
)

plt.savefig(
    RESULTADOS /
    f"01_original_vs_preprocesada_{REGISTRO}_nuevo.png",
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
    rect=(0.0, 0.0, 1.0, 0.95)
)

plt.savefig(
    RESULTADOS /
    f"02_cancelacion_materna_{REGISTRO}_nuevo.png",
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

axes_ica = list(np.atleast_1d(axes).ravel())

for i in range(fuentes.shape[1]):

    axes_ica[i].plot(
        tiempo,
        fuentes[:, i]
    )

    if i == idx_fetal:

        axes_ica[i].set_title(
            f"Componente ICA {i} — COMPONENTE FETAL SELECCIONADA",
            fontweight="bold"
        )

    else:

        axes_ica[i].set_title(
            f"Componente ICA {i}"
        )

    axes_ica[i].set_ylabel(
        "Amplitud"
    )

    axes_ica[i].grid(
        alpha=0.3
    )


axes_ica[-1].set_xlabel(
    "Tiempo (s)"
)


fig.suptitle(
    "Componentes obtenidas mediante FastICA",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout(
    rect=(0.0, 0.0, 1.0, 0.96)
)

plt.savefig(
    RESULTADOS /
    f"03_componentes_ICA_{REGISTRO}_nuevo.png",
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
    f"04_fECG_aislado_{REGISTRO}_nuevo.png",
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

from scipy.signal import butter, filtfilt, find_peaks  # type: ignore


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
    rect=(0.0, 0.0, 1.0, 0.95)
)

plt.savefig(
    RESULTADOS /
    f"05_filtro_pan_tompkins_{REGISTRO}_nuevo.png",
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
    f"06_derivada_{REGISTRO}_nuevo.png",
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
    f"07_elevacion_cuadrado_{REGISTRO}_nuevo.png",
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
    f"08_integracion_{REGISTRO}_nuevo.png",
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
    f"09_candidatos_qrs_{REGISTRO}_nuevo.png",
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
    f"10_fQRS_finales_{REGISTRO}_nuevo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ============================================================
#              PARTE E
#    CARACTERIZACIÓN Y TIEMPO-FRECUENCIA (NUEVAS MÉTRICAS)
# ============================================================
# ============================================================

print("\n")
print("=" * 70)
print("PARTE E — EXTRACCIÓN DE NUEVAS CARACTERÍSTICAS (TIEMPO, ENERGÍA, FORMA, STFT)")
print("=" * 70)

from scipy import signal as sp_signal  # type: ignore

# Evaluar el registro completo (60 s) para tener resolución suficiente de STFT y distribución RR
picos_full = extraer_fqrs_optimo(signals, fs=fs)
rr_ms_full = calcular_tacograma_rr(picos_full, fs=fs)
feats_full = extraer_vector_caracteristicas_completo(picos_full, fs=fs)

print("\nResumen de Nuevas Características Extraídas (Registro completo):")
for clave in [
    "BPM_mean", "SDNN", "RR_variance", "RR_min", "RR_max", "RR_range",
    "Signal_Energy", "Signal_Power", "RR_skewness", "RR_kurtosis",
    "STFT_Power_Mean", "STFT_Power_Var", "STFT_LF_HF_Mean", "STFT_Spectral_Entropy",
]:
    print(f"  - {clave:24s}: {feats_full.get(clave, 0.0):.4f}")

if len(rr_ms_full) >= 15:
    # ------------------------------------------------------------
    # GRÁFICA 11 — DISTRIBUCIÓN RR, ESTADÍSTICAS Y FORMA (ASIMETRÍA / CURTOSIS)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    t_rr = np.cumsum(rr_ms_full) / 1000.0
    axes[0].plot(t_rr, rr_ms_full, marker="o", markersize=3, linewidth=1.2, color="#1f77b4", label="Intervalo RR (ms)")
    axes[0].axhline(feats_full["RR_min"], color="#2ca02c", linestyle="--", linewidth=1.2, label=f"RR_min = {feats_full['RR_min']:.1f} ms")
    axes[0].axhline(feats_full["RR_max"], color="#d62728", linestyle="--", linewidth=1.2, label=f"RR_max = {feats_full['RR_max']:.1f} ms")
    axes[0].fill_between(t_rr, feats_full["RR_min"], feats_full["RR_max"], color="#ff7f0e", alpha=0.08, label=f"RR_range = {feats_full['RR_range']:.1f} ms")
    axes[0].set_title(f"Tacograma RR — {REGISTRO} (BPM_mean={feats_full['BPM_mean']:.1f})", fontweight="bold")
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_ylabel("Intervalo RR (ms)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3)

    axes[1].hist(rr_ms_full, bins=18, density=True, color="#4C72B0", alpha=0.75, edgecolor="black")
    axes[1].axvline(np.mean(rr_ms_full), color="#d62728", linestyle="-", linewidth=1.5, label=f"Media RR = {np.mean(rr_ms_full):.1f} ms")
    info_txt = (
        f"SDNN = {feats_full['SDNN']:.2f} ms\n"
        f"Varianza = {feats_full['RR_variance']:.2f} ms²\n"
        f"Asimetría (Skewness) = {feats_full['RR_skewness']:.3f}\n"
        f"Curtosis (Kurtosis) = {feats_full['RR_kurtosis']:.3f}"
    )
    axes[1].text(
        0.97, 0.95, info_txt, transform=axes[1].transAxes,
        ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    axes[1].set_title("Distribución de Intervalos RR y Forma", fontweight="bold")
    axes[1].set_xlabel("Intervalo RR (ms)")
    axes[1].set_ylabel("Densidad")
    axes[1].legend(loc="upper left", fontsize=8.5)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Estadísticas Básicas y Forma de Distribución — {REGISTRO}", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    plt.savefig(RESULTADOS / f"11_distribucion_rr_forma_{REGISTRO}_nuevo.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------
    # GRÁFICA 12 — ENERGÍA Y POTENCIA PROMEDIO DE VARIACIÓN
    # ------------------------------------------------------------
    diff_rr = np.diff(rr_ms_full)
    energia_inst = diff_rr ** 2
    energia_acum = np.cumsum(energia_inst)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    idx_lat = np.arange(1, len(energia_inst) + 1)

    axes[0].bar(idx_lat, energia_inst, color="#e67e22", alpha=0.8, label="Energía instantánea |ΔRR[n]|²")
    axes[0].axhline(feats_full["Signal_Power"], color="#c0392b", linestyle="--", linewidth=1.6,
                    label=f"Potencia promedio (Signal_Power) = {feats_full['Signal_Power']:.2f} ms²")
    axes[0].set_title("Potencia Instantánea vs Potencia Promedio Latido a Latido", fontweight="bold")
    axes[0].set_ylabel("Energía por latido (ms²)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(idx_lat, energia_acum, color="#2c3e50", linewidth=2,
                 label=f"Energía acumulada total (Signal_Energy) = {feats_full['Signal_Energy']:.1f} ms²")
    axes[1].fill_between(idx_lat, 0, energia_acum, color="#3498db", alpha=0.15)
    axes[1].set_title("Energía Total Acumulada de Variabilidad (Signal_Energy)", fontweight="bold")
    axes[1].set_xlabel("Índice de latido (n)")
    axes[1].set_ylabel("Energía acumulada (ms²)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Medidas Energéticas de la Señal RR — {REGISTRO}", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.savefig(RESULTADOS / f"12_energia_potencia_senal_{REGISTRO}_nuevo.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------
    # GRÁFICA 13 — DOMINIO TIEMPO-FRECUENCIA (ESPECTROGRAMA STFT)
    # ------------------------------------------------------------
    fs_interp = 4.0
    tiempo_acum = np.cumsum(rr_ms_full) / 1000.0
    tiempo_rel = tiempo_acum - tiempo_acum[0]
    t_interp = np.arange(0.0, tiempo_rel[-1], 1.0 / fs_interp)

    if len(t_interp) >= 24:
        rr_interp = np.asarray(np.interp(t_interp, tiempo_rel, rr_ms_full), dtype=np.float64)
        rr_detrend = sp_signal.detrend(rr_interp, type="linear")
        nperseg = max(8, min(len(rr_detrend), int(fs_interp * 16)))
        noverlap = int(nperseg * 0.75)
        nfft = max(256, nperseg)

        f_stft, t_stft, zxx = sp_signal.stft(
            rr_detrend, fs=fs_interp, window="hann",
            nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary=None,
        )
        potencia_tf = np.abs(zxx) ** 2
        potencia_db = 10.0 * np.log10(potencia_tf + 1e-12)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), gridspec_kw={"height_ratios": [2.2, 1]})
        mask_f = f_stft <= 1.0
        pcm = axes[0].pcolormesh(t_stft, f_stft[mask_f], potencia_db[mask_f, :], shading="gouraud", cmap="viridis")
        axes[0].axhline(0.04, color="white", linestyle="--", linewidth=1.0, label="Límite VLF/LF (0.04 Hz)")
        axes[0].axhline(0.20, color="#ffeb3b", linestyle="--", linewidth=1.2, label="Límite LF/HF (0.20 Hz)")
        axes[0].set_title(f"Espectrograma STFT (Tiempo-Frecuencia) — {REGISTRO}", fontweight="bold")
        axes[0].set_ylabel("Frecuencia (Hz)")
        axes[0].legend(loc="upper right", fontsize=8.5)
        fig.colorbar(pcm, ax=axes[0], label="Potencia Espectral (dB)")

        lf_mask = (f_stft >= 0.04) & (f_stft < 0.20)
        hf_mask = (f_stft >= 0.20) & (f_stft <= 1.00)
        lf_t = np.sum(potencia_tf[lf_mask, :], axis=0)
        hf_t = np.sum(potencia_tf[hf_mask, :], axis=0)
        ratio_t = lf_t / (hf_t + 1e-8)

        axes[1].plot(t_stft, ratio_t, color="#8e44ad", linewidth=1.8, marker="o", markersize=3,
                     label=f"Balance LF/HF en el tiempo (Media = {feats_full['STFT_LF_HF_Mean']:.2f})")
        stft_box = (
            f"STFT_Power_Mean = {feats_full['STFT_Power_Mean']:.2f}\n"
            f"STFT_Power_Var = {feats_full['STFT_Power_Var']:.2f}\n"
            f"STFT_Spectral_Entropy = {feats_full['STFT_Spectral_Entropy']:.3f}"
        )
        axes[1].text(
            0.98, 0.92, stft_box, transform=axes[1].transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )
        axes[1].set_title("Dinámica Temporal del Balance LF/HF (STFT)", fontweight="bold")
        axes[1].set_xlabel("Tiempo (s)")
        axes[1].set_ylabel("Ratio LF / HF")
        axes[1].legend(loc="upper left", fontsize=9)
        axes[1].grid(alpha=0.3)

        fig.suptitle(f"Análisis Tiempo-Frecuencia STFT — {REGISTRO}", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.savefig(RESULTADOS / f"13_stft_tiempo_frecuencia_{REGISTRO}_nuevo.png", dpi=300, bbox_inches="tight")
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
    RESULTADOS.glob("*_nuevo.png")
):

    print(
        f"  ✓ {archivo.name}"
    )

print("\n")