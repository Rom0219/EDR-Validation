#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sparc_fit import (
    load_rotmod_generic,
    fit_galaxy,
    plot_fit_with_residuals,
    plot_residual_histogram_single,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
HIST_DIR = os.path.join(RESULTS_DIR, "histograms")
GLOBAL_DIR = os.path.join(RESULTS_DIR, "summary_plots")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)

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


# ==============================================================
# PROCESO INDIVIDUAL
# ==============================================================

def process_galaxy(gname):
    fname = os.path.join(DATA_DIR, f"{gname}_rotmod.dat")
    print(f"[OK] Leyendo {fname}")

    try:
        data = load_rotmod_generic(fname)
    except Exception as e:
        print(f"[FAIL] {gname}: error en lectura -> {e}")
        return {"Galaxy": gname, "fit_ok": False, "error": str(e)}

    # Ajustar modelo EDR+bariones
    result, Vmodel_plot, sigma_extra, residuals = fit_galaxy(data, galaxy_name=gname)

    if result is None or not result["ok"]:
        print(f"[FAIL] {gname}: excepción durante fit_galaxy -> {result['error']}")
        return {"Galaxy": gname, "fit_ok": False, "error": result["error"]}

    print(f"[OK] Ajuste completo para {gname}")

    # --- PLOT con residuales ---
    out_plot = os.path.join(PLOTS_DIR, f"{gname}.png")
    plot_fit_with_residuals(data, Vmodel_plot, result, out_plot, galaxy_name=gname)
    print(f"     → Plot: {out_plot}")

    # --- Histograma por galaxia ---
    out_hist = os.path.join(HIST_DIR, f"{gname}_hist.png")
    plot_residual_histogram_single(residuals, out_hist, gname)

    # Empaquetar resultado
    return {
        "Galaxy": gname,
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
        "Ndata": len(data["r"]),
        "Ndof": len(data["r"]) - 4,
    }


# ==============================================================
# PROCESAMIENTO GLOBAL
# ==============================================================

all_residuals = []

out_csv = os.path.join(RESULTS_DIR, "sparc_results.csv")
summary_csv = os.path.join(RESULTS_DIR, "sparc_run_summary.csv")

print("=============================================")
print("   PROCESO SPARC + EDR — VALIDACIÓN MASIVA")
print("=============================================")


with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "Galaxy", "fit_ok", "mode",
        "A", "Aerr", "R0", "R0err",
        "Yd", "Yderr", "Yb", "Yberr",
        "chi2", "chi2_red", "sigma_extra",
        "Ndata", "Ndof"
    ])
    writer.writeheader()

    summary = []

    for g in GALAXIES:
        out = process_galaxy(g)
        summary.append(out)

        # Crear CSV principal
        if out.get("fit_ok"):
            writer.writerow(out)
        else:
            writer.writerow({
                "Galaxy": out["Galaxy"],
                "fit_ok": False,
                "mode": "",
                "A": "", "Aerr": "",
                "R0": "", "R0err": "",
                "Yd": "", "Yderr": "",
                "Yb": "", "Yberr": "",
                "chi2": "", "chi2_red": "",
                "sigma_extra": "",
                "Ndata": "", "Ndof": ""
            })


# ==============================================================
# GUARDAR resumen general
# ==============================================================

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print(">>> PROCESO COMPLETADO <<<")
print(f"Resultados en: {out_csv}")
print(f"Resumen en: {summary_csv}")
print(f"Plots en: {PLOTS_DIR}")
print(f"Histogramas por galaxia en: {HIST_DIR}")


# ==============================================================
# HISTOGRAMA GLOBAL DE RESIDUALES
# ==============================================================

print("Generando histograma global de residuales…")

global_hist_path = os.path.join(GLOBAL_DIR, "global_residuals_hist.png")

# Solo cuántas galaxias fueron fit_ok
residuals_list = []
for g in GALAXIES:
    fname = os.path.join(HIST_DIR, f"{g}_hist.png")
# Añadiremos en la versión avanzada recolectar datos directamente

plt.figure(figsize=(7, 5))
plt.title("Distribución global de residuales (todas las galaxias)")
plt.xlabel("Residual (km/s)")
plt.ylabel("Frecuencia")
plt.grid(True)

# Nota: Los residuales se adjuntarán desde el fit
#        (Mantenemos esta parte vacía hasta que agregues colecta directa)
plt.savefig(global_hist_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Histograma global: {global_hist_path}")
