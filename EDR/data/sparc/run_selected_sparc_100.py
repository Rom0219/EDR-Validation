#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_selected_sparc_100.py — Procesamiento masivo SPARC100
Lee todos los archivos *_rotmod.dat dentro de SPARC100/
Ejecuta ajustes SPARC + EDR usando sparc_fit_100.py
"""

import os
import csv
from sparc_fit_100 import (
    load_rotmod_generic,
    fit_galaxy,
    plot_fit_with_residuals,
    plot_global_residuals
)

# -----------------------------------------------------------
# DIRECTORIOS
# -----------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
SPARC100_DIR = os.path.join(BASE_DIR, "SPARC100")

RESULTS_DIR = os.path.join(BASE_DIR, "results_100")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
RESIDUALS_DIR = os.path.join(RESULTS_DIR, "residuals")

for d in [RESULTS_DIR, PLOTS_DIR, RESIDUALS_DIR]:
    os.makedirs(d, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULTS_DIR, "sparc100_results.csv")

# -----------------------------------------------------------
# ARCHIVOS SPARC100
# -----------------------------------------------------------

galaxy_files = sorted([
    f for f in os.listdir(SPARC100_DIR)
    if f.endswith("_rotmod.dat")
])

print("===============================================")
print("       PROCESO SPARC100 + EDR — MASIVO         ")
print("===============================================")

print(f"\nTotal identificado: {len(galaxy_files)} galaxias\n")

# -----------------------------------------------------------
# PREPARAR CSV DE RESULTADOS
# -----------------------------------------------------------

fieldnames = [
    "Galaxy", "fit_ok", "mode",
    "A", "Aerr", "R0", "R0err",
    "Yd", "Yderr", "Yb", "Yberr",
    "chi2", "chi2_red", "sigma_extra",
    "Ndata", "Ndof", "error"
]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# Acumulador para histograma global
all_residuals = []

# -----------------------------------------------------------
# LOOP PRINCIPAL
# -----------------------------------------------------------

for fname in galaxy_files:

    galaxy = fname.replace("_rotmod.dat", "")
    print(f"[OK] Leyendo {galaxy}")

    path = os.path.join(SPARC100_DIR, fname)

    try:
        data = load_rotmod_generic(path)
    except Exception as e:
        print(f"[FAIL] {galaxy}: Error al leer archivo: {e}")
        row = {
            "Galaxy": galaxy, "fit_ok": False, "error": str(e)
        }
        with open(OUTPUT_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        continue

    # ---------- FIT ----------
    try:
        result, Vmodel_plot, sigma_extra, residuals = fit_galaxy(data, galaxy_name=galaxy)
    except Exception as e:
        print(f"[FAIL] {galaxy}: excepción durante fit_galaxy -> {e}")
        row = {
            "Galaxy": galaxy, "fit_ok": False, "error": str(e)
        }
        with open(OUTPUT_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        continue

    if not result["ok"]:
        print(f"[FAIL] {galaxy}: {result['error']}")
        row = {
            "Galaxy": galaxy, "fit_ok": False, "error": result["error"]
        }
        with open(OUTPUT_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        continue

    print(f"[OK] Ajuste completo para {galaxy}")

    # ---------- Guardar plot ----------
    plot_path = os.path.join(PLOTS_DIR, f"{galaxy}.png")
    plot_fit_with_residuals(data, Vmodel_plot, result, fname=plot_path, galaxy_name=galaxy)

    # ---------- Guardar residuales ----------
    res_path = os.path.join(RESIDUALS_DIR, f"{galaxy}_residuals.txt")
    with open(res_path, "w") as fr:
        for r in residuals:
            fr.write(f"{r}\n")

    all_residuals.extend(residuals)

    # ---------- Guardar CSV ----------
    row = {
        "Galaxy": galaxy,
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
        "sigma_extra": result["sigma_extra"],
        "Ndata": len(data["r"]),
        "Ndof": len(data["r"]) - 4,
        "error": ""
    }

    with open(OUTPUT_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

# -----------------------------------------------------------
# HISTOGRAMA GLOBAL
# -----------------------------------------------------------

if len(all_residuals) > 0:
    print("\n[OK] Generando histograma global de residuales...")
    hist_path = os.path.join(RESULTS_DIR, "global_residuals.png")
    plot_global_residuals(all_residuals, fname=hist_path)

print("\n>>> PROCESO COMPLETADO <<<")
print(f"Resultados CSV: {OUTPUT_CSV}")
print(f"Plots por galaxia: {PLOTS_DIR}")
print(f"Residuals por galaxia: {RESIDUALS_DIR}")
