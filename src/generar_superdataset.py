"""
Módulo saneado (Fase 4 del Plan de Acción):
Se desactiva la generación de clones sintéticos corruptos (`_aug`) antes del split
de validación para erradicar la fuga de datos (Data Leakage) y las inconsistencias
algebraicas entre biomarcadores. Delega la construcción al pipeline clínico real.
"""
from build_dataset import construir_dataset_nifeadb

if __name__ == "__main__":
    construir_dataset_nifeadb(incluir_cinc2013=True)
