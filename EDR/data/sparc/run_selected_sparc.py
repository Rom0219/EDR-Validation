#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_selected_sparc.py — validación SPARC + EDR
------------------------------------------------
Produce:
    - Ajustes individuales
    - Plots normales y con residuales
    - Archivo CSV con estadísticas
    - Archivo .npy con residuales por galaxia
    - Histogramas globales de residuales
"""

import os
import csv
import numpy as np
from sparc_fit import (
    load_rotmod_generic,
    fit_galaxy,
    plot_fit,
    plot_fit_with_residuals
)

# ===========================================================
# CONFIGURACIÓN
# ===========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
RESID_DIR = os.path.join(RESULTS_DIR, "residuals")
HIST_DIR = os.path.join(RESULTS_DIR, "histograms")

for d in [RESULTS_DIR, PLOTS_DIR, RESID_DIR, HIST_DIR]:
    os.makedirs(d, exist_ok=True)

# Lista fija
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

CSV_OUT = os.path.join(RESULTS_DIR, "sparc_results.csv")
SUMMARY_OUT = os.path.join(RESULTS_DIR, "sparc_run_summary.csv")

# ===========================================================
# PROCESAR UNA GALAXIA
# ===========================================================

def process_galaxy(name):
    file_path = os.path.join(DATA_DIR, f"{name}_rotmod.dat")
    print(f"[OK] Leyendo {file_path}")

    try:
        data = load_rotmod_generic(file_path)
    except Exception as e:
        return {"Galaxy": name, "fit_ok": False, "error": f"read_error: {e}"}

    # ---- Ejecutar el ajuste ----
    result, Vmodel_plot, sigma_extra, residuals_obs = fit_galaxy(data, galaxy_name=name)

    if not result.get("ok", False):
        return {
            "Galaxy": name,
            "fit_ok": False,
            "error": result.get("error", "unknown_fit_error")
        }

    # Guardar residuales como archivo numpy
    np.save(os.path.join(RESID_DIR, f"{name}_residuals.npy"), residuals_obs)

    # ---- Hacer plots ----
    plot_fit(data, Vmodel_plot, result,
             fname=os.path.join(PLOTS_DIR, f"{name}.png"),
             galaxy_name=name)

    plot_fit_with_residuals(data, Vmodel_plot, result,
             fname=os.path.join(PLOTS_DIR, f"{name}_residuals.png"),
             galaxy_name=name)

    # ---- Stats de residuales ----
    resid_mean = float(np.mean(residuals_obs))
    resid_std = float(np.std(residuals_obs))
    resid_maxabs = float(np.max(np.abs(residuals_obs)))

    # ---- Preparar salida ----
    out = {
        "Galaxy": name,
        "fit_ok": True,
        "mode": result["mode"],
        "A": result["A"],
        "Aerr": result["Aerr"],
        "R0": result["R0"],
        "R0err": result["R0err"],
        "Yd": result["Yd"],
        "Yderr": result["Yderr"],
        "Yb": result["Yb"],
        "Yberr": result["Yberr"],
        "chi2": result["chi2"],
        "chi2_red": result["chi2_red"],
        "sigma_extra": sigma_extra,
        "residual_mean": resid_mean,
        "residual_std": resid_std,
        "residual_maxabs": resid_maxabs
    }

    return out


# ===========================================================
# GENERAR HISTOGRAMAS GLOBALES DE RESIDUALES
# ===========================================================

def make_global_residual_histogram():
    import matplotlib.pyplot as plt

    all_residuals = []

    for g in GALAXIES:
        f = os.path.join(RESID_DIR, f"{g}_residuals.npy")
        if os.path.exists(f):
            all_residuals.extend(np.load(f))

    if len(all_residuals) == 0:
        print("No hay residuales para histograma global")
        return

    all_residuals = np.array(all_residuals)

    plt.figure(figsize=(7, 5))
    plt.hist(all_residuals, bins=30, alpha=0.8)
    plt.xlabel("Residual (Vobs - Vmodel)")
    plt.ylabel("Frecuencia")
    plt.title("Histograma Global de Residuales — SPARC + EDR")
    plt.grid(True)
    plt.savefig(os.path.join(HIST_DIR, "global_residuals_hist.png"), dpi=200, bbox_inches="tight")
    plt.close()

# ===========================================================
# MAIN
# ===========================================================

print("=============================================")
print("   PROCESO SPARC + EDR — VALIDACIÓN MASIVA")
print("=============================================\n")

rows = []

for g in GALAXIES:
    out = process_galaxy(g)
    rows.append(out)

# Guardar CSV principal
FIELDNAMES = list(rows[0].keys())

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# Crear histograma global
make_global_residual_histogram()

print(">>> PROCESO COMPLETADO <<<")
print(f"Resultados en: {CSV_OUT}")
print(f"Plots en: {PLOTS_DIR}")
print(f"Residuales en: {RESID_DIR}")
print(f"Histogramas en: {HIST_DIR}")
