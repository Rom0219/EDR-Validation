#!/usr/bin/env python3
"""
run_selected_sparc.py
Nuevo lanzador compatible con la versión final de sparc_fit.py.

- Escanea la lista de galaxias (configurable)
- Carga cada *_rotmod.dat vía load_rotmod_generic
- Llama a fit_galaxy (devuelve result, modelV)
- Dibuja y salva la figura usando plot_fit
- Añade resultados al CSV con append_result
- Maneja gracefully galaxies sin bulbo y errores de ajuste
"""

import os
import sys
import time
import traceback
import pandas as pd

# Ajusta esta lista si quieres otras galaxias o un directorio distinto
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

# Rutas locales (asegúrate de ejecutar desde la raíz del repo)
BASE_DIR = os.path.dirname(__file__)  # EDR/data/sparc
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CSV_OUT = os.path.join(RESULTS_DIR, "sparc_results.csv")

# Ruta al PDF subido (referencia/URL local)
UPLOADED_PDF = "/mnt/data/FORMULAS_V2.pdf"

# Crear directorios si no existen
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Import funciones del módulo sparc_fit.py (asume que está en el mismo folder)
try:
    from sparc_fit import load_rotmod_generic, fit_galaxy, plot_fit, append_result
except Exception as e:
    print("ERROR: no pude importar funciones desde sparc_fit.py. Asegúrate de que el archivo está en EDR/data/sparc/")
    print("Detalle:", e)
    sys.exit(1)


def process_galaxy(galaxy_name):
    fname = os.path.join(BASE_DIR, f"{galaxy_name}_rotmod.dat")
    if not os.path.isfile(fname):
        print(f"[NO FILE] {galaxy_name} — archivo faltante: {fname}")
        return {"Galaxy": galaxy_name, "fit_ok": False, "error": "file_missing"}

    print(f"[OK] Leyendo {fname}")
    try:
        data = load_rotmod_generic(fname)
    except Exception as e:
        print(f"[ERROR] al cargar datos de {galaxy_name}: {e}")
        traceback.print_exc()
        return {"Galaxy": galaxy_name, "fit_ok": False, "error": f"load_error: {e}"}

    # Ajuste
    t0 = time.time()
    try:
        result, modelV = fit_galaxy(data)
    except Exception as e:
        print(f"[FAIL] {galaxy_name}: excepción durante fit_galaxy -> {e}")
        traceback.print_exc()
        return {"Galaxy": galaxy_name, "fit_ok": False, "error": f"fit_exception: {e}"}

    if result is None:
        print(f"[FAIL] {galaxy_name}: ajuste devolvió None y mensaje de error.")
        return {"Galaxy": galaxy_name, "fit_ok": False, "error": "fit_failed_no_result"}

    # Guardado de plot
    plot_path = os.path.join(PLOTS_DIR, f"{galaxy_name}.png")
    try:
        plot_fit(data, modelV, result, fname=plot_path, galaxy_name=galaxy_name)
    except Exception as e:
        print(f"[WARN] {galaxy_name}: fallo al generar plot: {e}")
        traceback.print_exc()

    # Append CSV result (usa append_result del módulo)
    try:
        append_result(CSV_OUT, galaxy_name, result)
    except Exception as e:
        print(f"[WARN] {galaxy_name}: no pude escribir CSV: {e}")
        traceback.print_exc()

    t1 = time.time()
    elapsed = t1 - t0
    print(f"[OK] Ajuste completo para {galaxy_name} (tiempo: {elapsed:.2f}s)")
    print(f"     → Plot: {plot_path}\n")

    # Devolver summary para inspección en memoria
    row = {
        "Galaxy": galaxy_name,
        "A": result.get("A"),
        "R0": result.get("R0"),
        "Yd": result.get("Yd"),
        "Yb": result.get("Yb"),
        "chi2": result.get("chi2"),
        "chi2_red": result.get("chi2_red"),
        "fit_ok": True,
        "mode": "EDR_barions"
    }
    return row


def main():
    print("=============================================")
    print("   PROCESO SPARC + EDR — VALIDACIÓN MASIVA")
    print("   (usar PDF de referencia en: {})".format(UPLOADED_PDF))
    print("=============================================\n")

    summary_rows = []

    for g in GALAXIES:
        try:
            row = process_galaxy(g)
            summary_rows.append(row)
        except Exception as e:
            print(f"[ERROR GRAVE] al procesar {g}: {e}")
            traceback.print_exc()
            summary_rows.append({"Galaxy": g, "fit_ok": False, "error": str(e)})

    # Guardar resumen completo (adicional al CSV individual)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "sparc_run_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(">>> PROCESO COMPLETADO <<<")
    print(f"Resumen guardado en: {summary_path}")
    print(f"Resultados en (CSV principal): {CSV_OUT}")
    print(f"Plots en: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
