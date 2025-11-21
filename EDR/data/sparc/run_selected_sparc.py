#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selected_sparc.py — pipeline final para SPARC + EDR

- Ejecuta fit en la lista GALAXIES
- Guarda resultados CSV
- Guarda residuales por galaxia (.npy)
- Genera plots con residuales y histogramas por galaxia
- Genera histograma global grande (1500x1200)
"""

import os
import csv
import numpy as np
from sparc_fit import (
    load_rotmod_generic,
    fit_galaxy,
    plot_fit_with_residuals,
    plot_residual_histogram_single,
    plot_residuals_hist_global
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
HIST_DIR = os.path.join(RESULTS_DIR, "histograms")
RESID_DIR = os.path.join(RESULTS_DIR, "residuals")
GLOBAL_DIR = os.path.join(RESULTS_DIR, "summary_plots")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(RESID_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)

# Lista de galaxias (ya usadas)
GALAXIES = [
    "NGC3198", "NGC2403", "NGC2841", "NGC6503",
    "NGC3521", "DDO154", "NGC3741", "IC2574",
    "NGC3109", "NGC2976"
]

OUT_CSV = os.path.join(RESULTS_DIR, "sparc_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "sparc_run_summary.csv")

# Campos CSV
FIELDNAMES = [
    "Galaxy","A","R0","Yd","Yb","chi2","chi2_red","sigma_extra","fit_ok","mode","Ndata","Ndof"
]

rows = []
residuals_for_global = []

print("=============================================")
print("   PROCESO SPARC + EDR — VALIDACIÓN MASIVA")
print("   Referencia PDF:", "/mnt/data/FORMULAS_V2.pdf")
print("=============================================\n")

for g in GALAXIES:
    fpath = os.path.join(DATA_DIR, f"{g}_rotmod.dat")
    print(f"[OK] Leyendo {fpath}")
    try:
        data = load_rotmod_generic(fpath)
    except Exception as e:
        print(f"[FAIL] {g}: lectura -> {e}")
        rows.append({
            "Galaxy": g, "A": "", "R0": "", "Yd": "", "Yb": "",
            "chi2": "", "chi2_red": "", "sigma_extra": "", "fit_ok": False, "mode": "", "Ndata": "", "Ndof": ""
        })
        continue

    result, Vmodel_plot, sigma_extra, residuals = fit_galaxy(data, galaxy_name=g)

    if (not result) or (not result.get("ok", False)):
        print(f"[FAIL] {g}: fit -> {result.get('error','unknown')}")
        rows.append({
            "Galaxy": g, "A": "", "R0": "", "Yd": "", "Yb": "",
            "chi2": "", "chi2_red": "", "sigma_extra": "", "fit_ok": False, "mode": "", "Ndata": "", "Ndof": ""
        })
        continue

    # Guardar residuales por galaxia
    resid_file = os.path.join(RESID_DIR, f"{g}_residuals.npy")
    np.save(resid_file, residuals)
    residuals_for_global.append(residuals)

    # Guardar plots
    out_plot = os.path.join(PLOTS_DIR, f"{g}.png")
    plot_fit_with_residuals(data, Vmodel_plot, result, out_plot, galaxy_name=g)

    out_hist = os.path.join(HIST_DIR, f"{g}_hist.png")
    plot_residual_histogram_single(residuals, out_hist, galaxy_name=g)

    Ndata = len(data["r"])
    Ndof = max(Ndata - 4, 0)

    rows.append({
        "Galaxy": g,
        "A": result["A"],
        "R0": result["R0"],
        "Yd": result["Yd"],
        "Yb": result["Yb"],
        "chi2": result["chi2"],
        "chi2_red": result["chi2_red"],
        "sigma_extra": sigma_extra,
        "fit_ok": True,
        "mode": result["mode"],
        "Ndata": Ndata,
        "Ndof": Ndof
    })

    print(f"[OK] Ajuste completo para {g} (mode={result['mode']}, sigma_extra={sigma_extra:.6g})")
    print(f"     → Plot: {out_plot}")
    print(f"     → Residuales: {resid_file}")
    print(f"     → Hist: {out_hist}\n")

# Escribir CSV final
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    import csv
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    for r in rows:
        w.writerow(r)

with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
    import csv
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# Histograma global grande (tamaño pedido)
global_hist_path = os.path.join(GLOBAL_DIR, "global_residuals_hist.png")
plot_residuals_hist_global(residuals_for_global, fname=global_hist_path, figsize=(15, 12))
print(f"[OK] Histograma global guardado en: {global_hist_path}")

print("\n>>> PROCESO COMPLETADO <<<")
print(f"Resultados en: {OUT_CSV}")
print(f"Resumen en: {SUMMARY_CSV}")
print(f"Plots en: {PLOTS_DIR}")
print(f"Histogramas por galaxia en: {HIST_DIR}")
print(f"Residuales (.npy) en: {RESID_DIR}")
print(f"Histograma global en: {global_hist_path}")
