import os
import pandas as pd
from sparc_fit import load_rotmod_generic, fit_galaxy, plot_fit

ROOT = os.path.dirname(__file__)
DATA_DIR = ROOT
OUT_DIR = os.path.join(ROOT, "results")
CSV_OUT = os.path.join(OUT_DIR, "sparc_results.csv")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(OUT_DIR, exist_ok=True)
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
    "NGC2976"
]

print("=============================================")
print("   PROCESO SPARC + EDR — VALIDACIÓN MASIVA")
print("=============================================\n")

rows = []

for g in GALAXIES:
    fname = os.path.join(DATA_DIR, f"{g}_rotmod.dat")
    if not os.path.exists(fname):
        print(f"[NO FILE] {g}")
        continue

    print(f"[OK] Leyendo {fname}")
    try:
        data = load_rotmod_generic(fname)
    except Exception as e:
        print(f"ERROR al cargar {g}: {e}")
        continue

    result = fit_galaxy(data)

    if result["ok"]:
        print(f"[OK] Ajuste completo para {g}")
        plot_fname = os.path.join(PLOTS_DIR, f"{g}.png")
        plot_fit(data, result, fname=plot_fname)
        print(f"     → Plot: {plot_fname}\n")

        p = result["params"]
        e = result["errors"]

        A, R0, Yd, Yb = p

        rows.append({
            "Galaxy": g,
            "A": A,
            "Aerr": e[0],
            "R0": R0,
            "R0err": e[1],
            "Yd": Yd,
            "Yderr": e[2],
            "Yb": Yb,
            "Yberr": e[3],
            "chi2": result["chi2"],
            "chi2_red": result["chi2_red"],
            "sigma_extra": result["sigma_extra"],
            "Ndata": result["Ndata"],
            "Ndof": result["Ndof"],
            "fit_ok": True,
            "mode": "SPARC_restricted"
        })
    else:
        print(f"[FAIL] {g}: {result['error']}\n")
        rows.append({"Galaxy": g, "fit_ok": False})

df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)

print(">>> PROCESO COMPLETADO <<<")
print(f"Resultados en: {CSV_OUT}")
print(f"Plots en: {PLOTS_DIR}")
