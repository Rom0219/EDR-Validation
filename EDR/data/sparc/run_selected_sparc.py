import os
from sparc_fit import (
    load_rotmod_generic,
    fit_edr_rotation_curve,
    plot_fit,
    save_fit_result
)

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATA_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(DATA_DIR, "results")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
CSV_PATH = os.path.join(OUT_DIR, "sparc_results.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Galaxias SPARC estándar + las tuyas
GALAXIES = [
    "NGC3198", "NGC2403", "NGC2841", "NGC6503", "NGC3521",
    "DDO154", "NGC3741", "IC2574", "NGC3109", "NGC2976"
]

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------
print("=============================================")
print("   PROCESO SPARC + EDR — VALIDACIÓN MASIVA")
print("=============================================")

for g in GALAXIES:

    fname = f"{g}_rotmod.dat"
    fpath = os.path.join(DATA_DIR, fname)

    if not os.path.isfile(fpath):
        print(f"[NO FILE] {g}")
        continue

    print(f"\n[OK] Leyendo {fpath}")

    try:
        data = load_rotmod_generic(fpath)
    except Exception as e:
        print(f"[ERROR] Fallo leyendo {g}: {e}")
        continue

    # Ajuste EDR
    fitres = fit_edr_rotation_curve(
        data["R"], data["Vobs"], data["eV"]
    )

    if fitres is None:
        print(f"[ERROR] No se pudo ajustar {g}")
        continue

    # Guardar CSV
    save_fit_result(fitres, g, CSV_PATH)

    # Graficar
    plot_path = os.path.join(PLOTS_DIR, f"{g}.png")
    plot_fit(data, fitres, plot_path)

    print(f"[OK] Ajuste completo para {g}")
    print(f"     → Plot: {plot_path}")

print("\n>>> PROCESO COMPLETADO <<<")
print(f"Resultados en: {CSV_PATH}")
print(f"Plots en: {PLOTS_DIR}")
