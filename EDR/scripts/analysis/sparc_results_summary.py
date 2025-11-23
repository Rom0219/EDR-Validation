#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_selected_sparc.py — versión completa
-----------------------------------------
Procesa múltiples galaxias SPARC con modelo EDR + bariones,
generando:

✔ Resultados numéricos
✔ Plots con curva + residuales
✔ CSV con todos los parámetros
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


def process_galaxy(g):
    filename = f"{g}_rotmod.dat"
    path = os.path.join(BASE_DIR, filename)

    print(f"[OK] Leyendo {path}")

    try:
        data = load_rotmod_generic(path)
    except Exception as e:
        print(f"[FAIL] {g}: error al leer -> {e}")
        return {"Galaxy": g, "fit_ok": False}

    try:
        result, Vmodel, sigma_extra = fit_galaxy(data, galaxy_name=g)
    except Exception as e:
        print(f"[FAIL] {g}: excepción durante fit_galaxy -> {e}")
        return {"Galaxy": g, "fit_ok": False}

    if not result["ok"]:
        print(f"[FAIL] {g}: ajuste fallido")
        return {"Galaxy": g, "fit_ok": False}

    plot_file = os.path.join(PLOTS_DIR, f"{g}.png")
    try:
        plot_fit_with_residuals(data, Vmodel, result, fname=plot_file, galaxy_name=g)
    except Exception as e:
        print(f"[FAIL] {g}: error al generar plot -> {e}")

    print(f"[OK] Ajuste completo para {g}")
    print(f"     → Plot: {plot_file}")

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
        "mode": result["mode"],
    }


if __name__ == "__main__":
    print("=============================================")
    print("   PROCESO SPARC + EDR — VALIDACIÓN MASIVA")
    print("=============================================\n")

    rows = []

    for g in GALAXIES:
        out = process_galaxy(g)
        rows.append(out)

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
