#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from sparc_fit import load_rotmod_generic, fit_galaxy, plot_fit_with_residuals

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

GALAXIES = [
    "NGC3198", "NGC2403", "NGC2841", "NGC6503", "NGC3521",
    "DDO154", "NGC3741", "IC2574", "NGC3109", "NGC2976"
]

def get_path(g):
    return os.path.join(BASE, f"{g}_rotmod.dat")

def process_galaxy(gname):
    path = get_path(gname)

    try:
        data = load_rotmod_generic(path)
        print(f"[OK] Leyendo {path}")
    except Exception as e:
        print(f"[FAIL] {gname}: error al leer → {e}")
        return {"Galaxy": gname, "fit_ok": False, "error": f"read: {e}"}

    try:
        result, modelV, sigma_extra = fit_galaxy(data, gname)
    except Exception as e:
        print(f"[FAIL] {gname}: excepción durante fit_galaxy -> {e}")
        return {"Galaxy": gname, "fit_ok": False, "error": f"fit: {e}"}

    if not result["ok"]:
        print(f"[FAIL] {gname}: {result['error']}")
        return {"Galaxy": gname, "fit_ok": False, "error": result["error"]}

    # Plot con residuales
    out_plot = os.path.join(PLOTS_DIR, f"{gname}.png")
    plot_fit_with_residuals(data, modelV, result, out_plot, gname)

    print(f"[OK] Ajuste completo para {gname}")
    print(f"     → Plot: {out_plot}")

    return {
        "Galaxy": gname,
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


print("=============================================")
print("   PROCESO SPARC + EDR — VALIDACIÓN MASIVA")
print("=============================================")

all_results = []

for g in GALAXIES:
    res = process_galaxy(g)
    all_results.append(res)

# Guardar CSV
csv_path = os.path.join(RESULTS_DIR, "sparc_results.csv")

fieldnames = [
    "Galaxy", "A", "R0", "Yd", "Yb",
    "chi2", "chi2_red",
    "sigma_extra", "fit_ok", "mode"
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_results:
        # evitar campos desconocidos
        filtered = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(filtered)

print(">>> PROCESO COMPLETADO <<<")
print(f"Resultados en: {csv_path}")
print(f"Plots en: {PLOTS_DIR}")
