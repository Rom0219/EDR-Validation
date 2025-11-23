#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sparc_residuals_hist_galaxies.py
------------------------------------------------
Genera histogramas de residuales para cada galaxia
usando los archivos .npy guardados en results/residuals (Análisis de Diagnóstico).
"""

import os
import numpy as np
# Importamos la función centralizada de ploteo
from sparc_utils import plot_residual_histogram_single

# --- CONFIGURACIÓN DE RUTAS (Ajustadas para la estructura de carpetas) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Asumimos que este script está en scripts/analysis/ o similar, subimos a la raíz
ROOT_DIR = os.path.join(BASE_DIR, "..", "..") 

RESID_DIR = os.path.join(ROOT_DIR, "results", "residuals")
# Usamos un directorio más específico para los histogramas individuales
HIST_DIR  = os.path.join(ROOT_DIR, "results", "histograms_individual")

os.makedirs(HIST_DIR, exist_ok=True)

# Lista de galaxias (puedes ampliarla a las 100 si es necesario)
GALAXIES = [
    "NGC3198", "NGC2403", "NGC2841", "NGC6503",
    "NGC3521", "DDO154", "NGC3741", "IC2574",
    "NGC3109", "NGC2976"
]

print("=============================================")
print("    HISTOGRAMAS DE RESIDUALES POR GALAXIA")
print("=============================================\n")

for g in GALAXIES:
    f = os.path.join(RESID_DIR, f"{g}_residuals.npy")

    if not os.path.exists(f):
        print(f"[SKIP] No residuals: {g}. Ejecuta el fit primero.")
        continue

    try:
        residuals = np.load(f)
    except Exception as e:
        print(f"[FAIL] Error cargando residuales de {g}: {e}")
        continue

    # >>> LLAMADA A LA FUNCIÓN CENTRALIZADA DE UTILITY <<<
    out_path = os.path.join(HIST_DIR, f"{g}_hist.png")
    plot_residual_histogram_single(residuals, out_path, galaxy_name=g)

    print(f"[OK] Histograma guardado → {out_path}")

print("\n>>> PROCESO COMPLETADO <<<")
