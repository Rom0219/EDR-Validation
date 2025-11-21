#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sparc_residuals_hist_galaxies.py
------------------------------------------------
Genera histogramas de residuales para cada galaxia
usando los archivos .npy guardados en results/residuals.

Salida:
    EDR/data/sparc/results/histograms/galaxies/<GALAXY>_hist.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESID_DIR = os.path.join(BASE_DIR, "results", "residuals")
HIST_DIR  = os.path.join(BASE_DIR, "results", "histograms", "galaxies")

os.makedirs(HIST_DIR, exist_ok=True)

GALAXIES = [
    "NGC3198", "NGC2403", "NGC2841", "NGC6503",
    "NGC3521", "DDO154", "NGC3741", "IC2574",
    "NGC3109", "NGC2976"
]

print("=============================================")
print("   HISTOGRAMAS DE RESIDUALES POR GALAXIA")
print("=============================================\n")

for g in GALAXIES:
    f = os.path.join(RESID_DIR, f"{g}_residuals.npy")

    if not os.path.exists(f):
        print(f"[SKIP] No residuals: {g}")
        continue

    residuals = np.load(f)

    plt.figure(figsize=(7,5))
    plt.hist(residuals, bins=25, alpha=0.85, color="darkblue")
    plt.axvline(np.mean(residuals), color="red", linestyle="--", label="Media")
    plt.title(f"Histograma de Residuales — {g}")
    plt.xlabel("Residual (Vobs - Vmodel)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(HIST_DIR, f"{g}_hist.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Histograma guardado → {out_path}")

print("\n>>> PROCESO COMPLETADO <<<")
print(f"Histogramas individuales en: {HIST_DIR}")
