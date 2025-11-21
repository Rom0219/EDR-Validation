#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_selected_sparc.py
--------------------------------------
Pipeline para procesar automáticamente un conjunto de galaxias SPARC
usando el modelo EDR + bariones + jitter + restricciones físicas.

Compatible con fit_galaxy() que ahora retorna:
    result, Vmodel, sigma_extra
"""

import os
import csv
import numpy as np
from sparc_fit import (
    load_rotmod_generic,
    fit_galaxy,
    plot_fit,
)

# -----------------------------------------
# CONFIGURACIÓN DE RUTAS
# -----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------------------
# LISTA DE GALAXIAS A PROCESAR
# -----------------------------------------
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
    "NGC2976",
]


# -------------------------------------------------------
# PROCESADOR DE UNA GALAXIA INDIVIDUAL
# -------------------------------------------------------
def process_galaxy(galaxy):
    """
    Realiza:
    1) Lectura del archivo rotmod
    2) Ajuste con EDR + bariones + jitter
    3) Plot con predicción EDR
    4) Retorno de parámetros para guardarlos en CSV
    """

    filename = f"{galaxy}_rotmod.dat"
    path = os.path.join(DATA_DIR, filename)

    print(f"[OK] Leyendo {path}")

    # --- 1) Carga ---
    try:
        data = load_rotmod_generic(path)
    except Exception as e:
        print(f"[FAIL] {galaxy}: error al leer -> {e}")
        return {"Galaxy": galaxy, "fit_ok": False, "error": str(e)}

    # --- 2) Ajuste ---
    try:
        # AHORA SON 3 VALORES
        result, modelV, sigma_extra = fit_galaxy(data, galaxy_name=galaxy)
    except Exception as e:
        print(f"[FAIL] {galaxy}: excepción durante fit_galaxy -> {e}")
        return {"Galaxy": galaxy, "fit_ok": False, "error": str(e)}

    if not result["ok"]:
        print(f"[FAIL] {galaxy}: ajuste fallido")
        return {"Galaxy": galaxy, "fit_ok": False, "error": "fit_failed"}

    # --- 3) Plot ---
    out_plot = os.path.join(PLOTS_DIR, f"{galaxy}.png")
    try:
        plot_fit(data, modelV, result, fname=out_plot, galaxy_name=galaxy)
    except Exception as e:
        print(f"[FAIL] {galaxy}: error al generar plot -> {e}")

    print(f"[OK] Ajuste completo para {galaxy}")
    print(f"     → Plot: {out_plot}")

    # --- 4) Retornar registro ---
    return {
        "Galaxy": galaxy,
        "A": result["A"],
        "R0": result["R0"],
        "Yd": result["Yd"],
        "Yb": result["Yb"],
        "chi2": result["chi2"],
        "chi2_red": result["chi2_red"],
        "sigma_extra": sigma_extra,
        "fit_ok": True,
        "mode": "EDR_barions_jitter_conditioned"
    }


# -------------------------------------------------------
# EJECUCIÓN MASIVA
# -------------------------------------------------------
if __name__ == "__main__":
    print("=============================================")
    print("   PROCESO SPARC + EDR — VALIDACIÓN MASIVA")
    print("   (usar PDF de referencia en: /mnt/data/FORMULAS_V2.pdf)")
    print("=============================================\n")

    rows = []

    for g in GALAXIES:
        row = process_galaxy(g)
        rows.append(row)

    # Guardar CSV principal
    out_csv = os.path.join(RESULTS_DIR, "sparc_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Galaxy", "A", "R0", "Yd", "Yb",
                "chi2", "chi2_red",
                "sigma_extra", "fit_ok", "mode"
            ]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Guardar resumen simple
    out_summary = os.path.join(RESULTS_DIR, "sparc_run_summary.csv")
    with open(out_summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Galaxy", "fit_ok", "mode"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "Galaxy": r["Galaxy"],
                "fit_ok": r["fit_ok"],
                "mode": r.get("mode", "")
            })

    print(">>> PROCESO COMPLETADO <<<")
    print(f"Resumen guardado en: {out_summary}")
    print(f"Resultados en (CSV principal): {out_csv}")
    print(f"Plots en: {PLOTS_DIR}")
