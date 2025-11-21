#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_selected_sparc_100.py
------------------------------------
Procesamiento masivo de SPARC100 con el modelo EDR.
Usa sparc_fit_100.py
"""

import os
import csv
import numpy as np
from sparc_fit_100 import (
    load_rotmod_generic,
    fit_galaxy,
    plot_fit_with_residuals,
    plot_residual_histogram
)

# ----------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# ----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "SPARC100")     # <-- donde metiste las 100 galaxias
OUT_DIR = os.path.join(BASE_DIR, "SPARC100_results")

PLOTS_DIR = os.path.join(OUT_DIR, "plots")
HIST_DIR = os.path.join(OUT_DIR, "hist")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

print("=============================================")
print("       PROCESO SPARC100 + EDR — MASIVO       ")
print("=============================================")


# ----------------------------------------------------------
# CSV de salida
# ----------------------------------------------------------

csv_path = os.path.join(OUT_DIR, "sparc100_results.csv")

fieldnames = [
    "Galaxy", "fit_ok", "mode",
    "A", "Aerr",
    "R0", "R0err",
    "Yd", "Yderr",
    "Yb", "Yberr",
    "chi2", "chi2_red",
    "sigma_extra",
    "Ndata", "Ndof"
]

f_csv = open(csv_path, "w", newline="", encoding="utf-8")
writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
writer.writeheader()


# ----------------------------------------------------------
# PROCESAMIENTO DE CADA GALAXIA
# ----------------------------------------------------------

all_residuals = []

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_rotmod.dat")])

for filename in files:
    galaxy = filename.replace("_rotmod.dat", "")
    fullpath = os.path.join(DATA_DIR, filename)

    print(f"\n[OK] Leyendo {fullpath}")

    try:
        data = load_rotmod_generic(fullpath)
    except Exception as e:
        print(f"[FAIL] {galaxy}: error al leer archivo → {e}")
        continue

    # ---------- FIT ----------
    result, Vmodel_plot, sigma_extra, residuals = fit_galaxy(data, galaxy_name=galaxy)

    if not result["ok"]:
        print(f"[FAIL] {galaxy}: ajuste falló → {result['error']}")
        continue

    print(f"[OK] Ajuste completo para {galaxy}")

    # Acumular residuales
    if residuals is not None:
        all_residuals.extend(residuals.tolist())

    # ---------- GRÁFICO RESIDUALES ----------
    plot_path = os.path.join(PLOTS_DIR, f"{galaxy}.png")
    plot_fit_with_residuals(data, Vmodel_plot, result, plot_path, galaxy_name=galaxy)

    # ---------- HISTOGRAMA INDIVIDUAL ----------
    hist_path = os.path.join(HIST_DIR, f"{galaxy}_hist.png")
    plot_residual_histogram(residuals, hist_path, galaxy)

    # ---------- GUARDAR CSV ----------
    writer.writerow({
        "Galaxy": galaxy,
        "fit_ok": result["ok"],
        "mode": result["mode"],
        "A": result["A"], "Aerr": result["Aerr"],
        "R0": result["R0"], "R0err": result["R0err"],
        "Yd": result["Yd"], "Yderr": result["Yderr"],
        "Yb": result["Yb"], "Yberr": result["Yberr"],
        "chi2": result["chi2"],
        "chi2_red": result["chi2_red"],
        "sigma_extra": result["sigma_extra"],
        "Ndata": result["Ndata"],
        "Ndof": result["Ndof"]
    })

f_csv.close()


# ----------------------------------------------------------
# HISTOGRAMA GLOBAL DE RESIDUALES
# ----------------------------------------------------------

import matplotlib.pyplot as plt

global_png = os.path.join(OUT_DIR, "residuals_global.png")

plt.figure(figsize=(7, 5))
plt.hist(all_residuals, bins=25, color="gray", edgecolor="black")
plt.axvline(0, color="red", linestyle="--")
plt.title("Histograma Global de Residuales — SPARC100")
plt.xlabel("Residual (km/s)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.savefig(global_png, dpi=240, bbox_inches="tight")
plt.close()


print("\n>>> PROCESO COMPLETADO <<<")
print(f"CSV en: {csv_path}")
print(f"Plots en: {PLOTS_DIR}")
print(f"Hist individuales en: {HIST_DIR}")
print(f"Histograma global: {global_png}")
