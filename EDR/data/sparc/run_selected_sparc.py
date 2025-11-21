#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_selected_sparc.py — versión corregida
-----------------------------------------
Soluciona:

✔ broadcasting error (shapes (N,) vs (300,))
✔ error=... en el CSV
✔ usa plot con residuales
"""

import os
import csv
from sparc_fit import (
    load_rotmod_generic,
    fit_galaxy,
    plot_fit_with_residuals
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

GALAXIES = [
    "NGC3198", "NGC2403", "NGC2841", "NGC6503", "NGC3521",
    "DDO154", "NGC3741", "IC2574", "NGC3109", "NGC2976"
]


# ==========================================================
# FUNCIÓN SEGURA PARA ARMAR UN REGISTRO DE CSV SIN ERRORES
# ==========================================================

def empty_record(galaxy, ok=False):
    return {
        "Galaxy": galaxy,
        "A": None,
        "R0": None,
        "Yd": None,
        "Yb": None,
        "chi2": None,
        "chi2_red": None,
        "sigma_extra": None,
        "fit_ok": ok,
        "mode": None
    }


# ==========================================================
# PROCESADOR INDIVIDUAL
# ==========================================================

def process_galaxy(g):
    filename = f"{g}_rotmod.dat"
    path = os.path.join(BASE_DIR, filename)

    print(f"[OK] Leyendo {path}")

    try:
        data = load_rotmod_generic(path)
    except Exception as e:
        print(f"[FAIL] {g}: error al leer -> {e}")
        return empty_record(g)

    try:
        result, Vmodel, sigma_extra = fit_galaxy(data, galaxy_name=g)
    except Exception as e:
        print(f"[FAIL] {g}: excepción durante fit_galaxy -> {e}")
        return empty_record(g)

    if not result["ok"]:
        print(f"[FAIL] {g}: ajuste fallido")
        return empty_record(g)

    # generar plot con residuales
    try:
        out_plot = os.path.join(PLOTS_DIR, f"{g}.png")
        plot_fit_with_residuals(data, Vmodel, result, fname=out_plot, galaxy_name=g)
        print(f"[OK] Ajuste completo para {g}")
        print(f"     → Plot: {out_plot}")
    except Exception as e:
        print(f"[FAIL] {g}: error al graficar -> {e}")

    # registro final
    return {
        "Galaxy": g,
        "A": result["A"],
        "R0": result["R0"],
        "Yd": result["Yd"],
        "Yb": result["Yb"],
        "chi2": result["chi2"],
        "chi2_red": result["chi2_red"],
        "sigma_extra": sigma_extra,
        "fit_ok": True,
        "mode": result["mode"]
    }


# ==========================================================
# EJECUCIÓN MASIVA
# ==========================================================

if __name__ == "__main__":

    print("=============================================")
    print("   PROCESO SPARC + EDR — VALIDACIÓN MASIVA")
    print("=============================================\n")

    rows = []

    for g in GALAXIES:
        record = process_galaxy(g)
        rows.append(record)

    out_csv = os.path.join(RESULTS_DIR, "sparc_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Galaxy", "A", "R0", "Yd", "Yb",
                "chi2", "chi2_red", "sigma_extra",
                "fit_ok", "mode"
            ]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(">>> PROCESO COMPLETADO <<<")
    print(f"Resultados en: {out_csv}")
    print(f"Plots en: {PLOTS_DIR}")
