#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selected_sparc.py — pipeline final para SPARC + EDR (Validación Local)

- Ejecuta fit en la lista GALAXIES usando archivos *_rotmod.dat (derivados de Table 2)
"""
import os
import csv
import numpy as np
from sparc_utils import (
    load_rotmod_generic, fit_galaxy, plot_fit_with_residuals, 
    plot_residual_histogram_single, plot_residuals_hist_global
)

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ajustar rutas relativas al directorio 'scripts'
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "sparc", "datafiles") 
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
HIST_DIR = os.path.join(RESULTS_DIR, "histograms")
RESID_DIR = os.path.join(RESULTS_DIR, "residuals")
GLOBAL_DIR = os.path.join(RESULTS_DIR, "summary_plots")

# Crear directorios de salida
os.makedirs(DATA_DIR, exist_ok=True) 
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(RESID_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)

# Lista de galaxias
GALAXIES = ["NGC3198", "NGC2403", "NGC2841", "NGC6503", "NGC3521", "DDO154", "NGC3741", "IC2574", "NGC3109", "NGC2976"]

OUT_CSV = os.path.join(RESULTS_DIR, "sparc_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "sparc_run_summary.csv")
FIELDNAMES = ["Galaxy","A","R0","Yd","Yb","chi2","chi2_red","sigma_extra","fit_ok","mode","Ndata","Ndof"]

rows = []
residuals_for_global = []

print("=============================================")
print("     PROCESO SPARC + EDR — VALIDACIÓN LOCAL")
print("=============================================\n")

for g in GALAXIES:
    fpath = os.path.join(DATA_DIR, f"{g}_rotmod.dat")
    
    if not os.path.exists(fpath):
        print(f"[SKIP] {g}: Archivo {fpath} no encontrado.")
        continue

    try:
        data = load_rotmod_generic(fpath)
    except Exception as e:
        print(f"[FAIL] {g}: lectura -> {e}")
        # Lógica para registrar fallos
        continue

    result, Vmodel_plot, sigma_extra, residuals = fit_galaxy(data, galaxy_name=g)

    if (not result) or (not result.get("ok", False)):
        print(f"[FAIL] {g}: fit -> {result.get('error','unknown')}")
        continue

    # Guardar resultados
    resid_file = os.path.join(RESID_DIR, f"{g}_residuals.npy")
    np.save(resid_file, residuals)
    residuals_for_global.append(residuals)
    
    out_plot = os.path.join(PLOTS_DIR, f"{g}.png")
    plot_fit_with_residuals(data, Vmodel_plot, result, out_plot, galaxy_name=g)

    out_hist = os.path.join(HIST_DIR, f"{g}_hist.png")
    plot_residual_histogram_single(residuals, out_hist, galaxy_name=g)

    Ndata = data.get("N_valid", 0)
    Ndof = max(Ndata - 4, 0)

    rows.append({"Galaxy": g, "A": result["A"], "R0": result["R0"], "Yd": result["Yd"], "Yb": result["Yb"],
                 "chi2": result["chi2"], "chi2_red": result["chi2_red"], "sigma_extra": sigma_extra,
                 "fit_ok": True, "mode": result["mode"], "Ndata": Ndata, "Ndof": Ndof})

    print(f"[OK] Ajuste completo para {g} (chi2_red={result['chi2_red']:.2f})")

# Escribir CSV final y resumen
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    w.writerows([r for r in rows if r["fit_ok"]])

with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    w.writerows(rows)

# Histograma global
if residuals_for_global:
    global_hist_path = os.path.join(GLOBAL_DIR, "global_residuals_hist.png")
    plot_residuals_hist_global(residuals_for_global, fname=global_hist_path, figsize=(15, 12))

print("\n>>> PROCESO COMPLETADO <<<")
