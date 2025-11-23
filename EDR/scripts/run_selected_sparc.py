#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selected_sparc.py — Script Principal de Fitting EDR + SPARC
--------------------------------------------------------------
Ejecuta el ajuste del modelo EDR + Bariones para cada galaxia
en la lista usando las utilidades de sparc_utils.py.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# IMPORTAMOS TODAS LAS FUNCIONES DESDE LA LIBRERÍA CONSOLIDADA
from sparc_utils import load_rotmod_generic, fit_galaxy, plot_fit_with_residuals, plot_residual_histogram_single, plot_residuals_hist_global

# --- 1. CONFIGURACIÓN DE RUTAS Y GALAXIAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Asumimos que este script está en scripts/, subimos un nivel a la raíz del proyecto
ROOT_DIR = os.path.join(BASE_DIR, "..") 

# Rutas de datos y resultados
DATA_DIR = os.path.join(ROOT_DIR, "EDR", "data", "sparc", "rotmod_data") # Carpeta donde están los archivos *_rotmod.dat
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "fits_plots")
RESIDUALS_DIR = os.path.join(RESULTS_DIR, "residuals")

# Crear directorios si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESIDUALS_DIR, exist_ok=True)

# Lista de las primeras 10 galaxias (tu muestra de prueba)
GALAXIES_SAMPLE = [
    "NGC3198", "NGC2403", "NGC2841", "NGC6503",
    "NGC3521", "DDO154", "NGC3741", "IC2574",
    "NGC3109", "NGC2976"
]

# Archivo de salida principal
RESULTS_CSV = os.path.join(RESULTS_DIR, "sparc_results.csv")

# --- 2. PIPELINE DE EJECUCIÓN ---
all_results = []
all_residuals = []

print(f"--- INICIANDO FIT DE {len(GALAXIES_SAMPLE)} GALAXIAS ---")

for i, galaxy_name in enumerate(GALAXIES_SAMPLE):
    print(f"\n[{i+1}/{len(GALAXIES_SAMPLE)}] Procesando {galaxy_name}...")
    
    rotmod_path = Path(DATA_DIR) / f"{galaxy_name}_rotmod.dat"
    
    if not rotmod_path.exists():
        print(f"[SKIP] Archivo de datos no encontrado para {galaxy_name} en {rotmod_path}")
        continue

    try:
        # A. Cargar datos
        data = load_rotmod_generic(rotmod_path)
    except Exception as e:
        print(f"[FAIL] Error al cargar {galaxy_name}: {e}")
        continue

    # B. Ejecutar ajuste
    result, Vmodel_plot, sigma_extra, residuals = fit_galaxy(data, galaxy_name=galaxy_name)

    # C. Procesar resultados
    if result["ok"]:
        print(f"[OK] Fit exitoso. Chi2_red: {result['chi2_red']:.3f}, Sigma_extra: {sigma_extra:.3f} km/s")
        
        # Guardar parámetros en el resultado
        row = {"Galaxy": galaxy_name, "N_points": len(data["r"]), "sigma_extra": sigma_extra}
        row.update({k: v for k, v in result.items() if k not in ["ok", "r_plot", "mode"]})
        
        all_results.append(row)
        all_residuals.append(residuals)
        
        # D. Generar plots (Usando las utilidades de sparc_utils)
        plot_fname = os.path.join(PLOTS_DIR, f"{galaxy_name}_fit.png")
        plot_fit_with_residuals(data, Vmodel_plot, result, plot_fname, galaxy_name)
        
        hist_fname = os.path.join(PLOTS_DIR, f"{galaxy_name}_hist.png")
        plot_residual_histogram_single(residuals, hist_fname, galaxy_name)
        
        # E. Guardar residuales por separado para análisis posterior
        np.save(os.path.join(RESIDUALS_DIR, f"{galaxy_name}_residuals.npy"), residuals)

    else:
        print(f"[FAIL] El fit falló para {galaxy_name}. Error: {result['error']}")


# --- 3. RESUMEN Y FINALIZACIÓN ---
if all_results:
    final_df = pd.DataFrame(all_results)
    
    # 3.1. Guardar resultados globales
    final_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n--- ÉXITO ---")
    print(f"Resultados de {len(final_df)} galaxias guardados en: {RESULTS_CSV}")
    
    # 3.2. Generar histograma global de residuales
    global_hist_fname = os.path.join(PLOTS_DIR, "global_residuals_hist.png")
    plot_residuals_hist_global(all_residuals, global_hist_fname)
    print(f"Histograma global de residuales guardado en: {global_hist_fname}")

    # 3.3. Mostrar resumen
    print("\nResumen de parámetros promedio:")
    print(final_df[["A", "R0", "Yd", "Yb", "chi2_red"]].mean().to_string())

else:
    print("\n--- FALLO ---")
    print("No se pudieron procesar galaxias exitosamente.")
