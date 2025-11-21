#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
residuals_analysis.py
-------------------------------------
Genera:

✔ Histograma global de residuales (todas las galaxias)
✔ Histograma por galaxia
✔ Distribución normal ajustada
✔ Estadísticos: media, sigma, skewness, kurtosis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

from sparc_fit import (
    load_rotmod_generic,
    fit_galaxy
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

GALAXIES = [
    "NGC3198",
    "NGC2403",
    "NGC2841",
    "NGC6503",
    "NGC3521",
    "DDO154",
    "NGC3741",
    "IC2574",
    "NGC3109",
    "NGC2976"
]

all_residuals = []


def get_residuals(galaxy):
    """Retorna residuales observados - modelo para una galaxia."""
    fname = f"{galaxy}_rotmod.dat"
    path = os.path.join(BASE_DIR, fname)

    try:
        data = load_rotmod_generic(path)
        result, Vmodel_plot, _ = fit_galaxy(data, galaxy_name=galaxy)
    except Exception as e:
        print(f"[FAIL] Residuales {galaxy}: {e}")
        return None

    if not result["ok"]:
        print(f"[FAIL] Ajuste fallido para {galaxy}")
        return None

    # Interpolo para obtener el modelo en puntos observados
    Vmodel_obs = np.interp(data["r"], result["r_plot"], Vmodel_plot)

    residuals = data["Vobs"] - Vmodel_obs
    return residuals


# ------------------------------
# 1) Recolectar residuales globales
# ------------------------------
for g in GALAXIES:
    res = get_residuals(g)
    if res is not None:
        all_residuals.extend(res)


all_residuals = np.array(all_residuals)
mu = np.mean(all_residuals)
sigma = np.std(all_residuals)
sk = skew(all_residuals)
kt = kurtosis(all_residuals)

print("\n=========================================")
print("  ESTADÍSTICAS GLOBALES DE RESIDUALES")
print("=========================================")
print(f"Media         = {mu:.4f} km/s")
print(f"Sigma         = {sigma:.4f} km/s")
print(f"Skewness      = {sk:.4f}")
print(f"Kurtosis      = {kt:.4f}")
print("=========================================\n")


# ------------------------------
# 2) Histograma global
# ------------------------------
x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
pdf_vals = norm.pdf(x_vals, mu, sigma)

plt.figure(figsize=(8,6))
plt.hist(all_residuals, bins=40, density=True, alpha=0.6, color="steelblue", label="Residuales")
plt.plot(x_vals, pdf_vals, "r--", label=f"N({mu:.2f}, {sigma:.2f}) ajustada")
plt.title("Histograma global de residuales (SPARC + EDR)")
plt.xlabel("Residual (km/s)")
plt.ylabel("Densidad")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "global_hist_residuals.png"), dpi=200)
plt.close()

print(f"[OK] Histograma global generado en: {PLOTS_DIR}/global_hist_residuals.png")


# ------------------------------
# 3) Histograma por galaxia
# ------------------------------
for g in GALAXIES:
    res = get_residuals(g)
    if res is None:
        continue

    plt.figure(figsize=(7,5))
    plt.hist(res, bins=20, alpha=0.7, color="darkgreen")
    plt.title(f"Residuales para {g}")
    plt.xlabel("Residual (km/s)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{g}_hist_residuals.png"), dpi=200)
    plt.close()

print("[OK] Histogramas individuales generados.")
print("[DONE] Residual analysis completo.")
