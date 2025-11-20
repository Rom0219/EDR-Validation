#!/usr/bin/env python3
# run_selected_sparc.py
# Ejecuta la validación en las 10 galaxias seleccionadas.

import os
import glob
import pandas as pd

from sparc_fit import (
    load_rotmod_generic,
    fit_galaxy,
    bootstrap_errors,
    plot_fit
)

GALAXIES = [
    "NGC3198","NGC2403","NGC2841","NGC6503","NGC3521",
    "DDO154","NGC3741","IC2574","NGC3109","NGC2976"
]

DATA_DIR = os.path.dirname(__file__)
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def find_file(gal):
    pattern = os.path.join(DATA_DIR, f"*{gal}*")
    matches = glob.glob(pattern)
    matches = [m for m in matches if m.endswith((".dat",".txt",".csv"))]
    return matches[0] if matches else None

results = []

for g in GALAXIES:
    f = find_file(g)
    if f is None:
        print(f"[NO FILE] {g}")
        continue

    print(f"[OK] Leyendo {f}")
    data = load_rotmod_generic(f)
    fitres = fit_galaxy(data)
    err = bootstrap_errors(data, fitres)

    fitres["galaxy"] = g
    fitres.update(err)
    results.append(fitres)

    plot_fit(data, fitres, fname=os.path.join(PLOTS_DIR, f"{g}.png"))

df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, "sparc_results.csv"), index=False)

print("\n>>> PROCESO COMPLETADO <<<")
print("Resultados en: results/sparc_results.csv")
print("Plots en: results/plots/")
