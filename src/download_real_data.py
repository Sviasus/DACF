import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CINC_DIR = PROJECT_ROOT / "data" / "cinc2013_real"
NIFEA_DIR = PROJECT_ROOT / "data" / "nifeadb"

CINC_DIR.mkdir(parents=True, exist_ok=True)
NIFEA_DIR.mkdir(parents=True, exist_ok=True)


def descargar_cinc2013_set_a():
    """
    Asegura que los 75 registros de CinC Challenge 2013 (Set-A) tengan sus archivos
    .hea, .dat y las anotaciones médicas de verdad de campo (.fqrs) en disco local.
    """
    set_a_dir = CINC_DIR / "set-a"
    set_a_dir.mkdir(parents=True, exist_ok=True)

    faltantes = []
    for i in range(1, 76):
        rec = f"a{i:02d}"
        for ext in [".hea", ".dat", ".fqrs"]:
            if not (set_a_dir / f"{rec}{ext}").exists():
                faltantes.append(f"set-a/{rec}{ext}")

    if not faltantes:
        print("[OK] Los 75 registros (.hea, .dat, .fqrs) de CinC 2013 Set-A ya existen en disco.")
        return

    print(f"Descargando {len(faltantes)} archivos faltantes de CinC 2013 Set-A (incluyendo .fqrs)...")
    wfdb.dl_files(db="challenge-2013", dl_dir=str(CINC_DIR), files=faltantes)
    print("[OK] Descarga de CinC 2013 Set-A completada.")


def descargar_nifeadb():
    """
    Descarga y persiste en disco local los 26 registros clínicos de NIFEA DB
    (ARR_01..ARR_12 y NR_01..NR_14) para garantizar reproducibilidad offline.
    """
    registros = [f"ARR_{i:02d}" for i in range(1, 13)] + [f"NR_{i:02d}" for i in range(1, 15)]
    faltantes = []
    for rec in registros:
        for ext in [".hea", ".dat"]:
            if not (NIFEA_DIR / f"{rec}{ext}").exists():
                faltantes.append(f"{rec}{ext}")

    if not faltantes:
        print("[OK] Los 26 registros de NIFEA DB ya existen en disco local.")
        return

    print(f"Descargando {len(faltantes)} archivos clínicos de NIFEA DB en {NIFEA_DIR}...")
    wfdb.dl_files(db="nifeadb", dl_dir=str(NIFEA_DIR), files=faltantes)
    print("[OK] Los 26 registros de NIFEA DB se descargaron correctamente.")


if __name__ == "__main__":
    descargar_cinc2013_set_a()
    descargar_nifeadb()
