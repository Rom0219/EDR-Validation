#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_btfr_validation.py — Validación de la Relación Tully-Fisher Bariónica (BTFR)
-------------------------------------------------------------------------------
Usa los parámetros del fit (Yd, Yb) junto con los datos de luminosidad (L[3.6])
para calcular la masa bariónica y verificar la BTFR.
"""
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

# --- INICIO DEL FIX PARA EL ERROR ModuleNotFoundError: 'sparc_utils' ---
# El script está en 'validation/', necesitamos acceder a 'scripts/'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
# Añadir la carpeta 'scripts/' al path de Python para que la importación funcione
sys.path.append(parent_dir)
# --- FIN DEL FIX ---

# Ahora importamos las funciones de utilidad
try:
    from sparc_utils import parse_table1, plot_btfr, TABLE1_FILENAME
except ImportError:
    print("[ERROR] El fix de ruta falló o sparc_utils.py no está en EDR/scripts/.")
    exit()

# --- CONFIGURACIÓN DE RUTAS ---
# Subimos a la raíz del proyecto (EDR/)
ROOT_DIR = os.path.join(parent_dir, "..")

# Rutas a los datos necesarios
RESULTS_CSV = os.path.join(ROOT_DIR, "results", "sparc_results.csv")
DATA_DIR = os.path.join(ROOT_DIR, "data", "sparc", "datafiles") # Ruta a la Tabla 1
TABLE1_PATH = os.path.join(DATA_DIR, TABLE1_FILENAME)

# Directorio de salida
OUTDIR = os.path.join(ROOT_DIR, "results", "validation")
os.makedirs(OUTDIR, exist_ok=True)

# CONSTANTES FÍSICAS
# L[3.6] y MHI en Table 1 están en 10^9 M_sol
LUMINOSITY_UNIT = 1e9 

print("=============================================")
print("    VALIDACIÓN: RELACIÓN TULLY-FISHER BARIÓNICA (BTFR)")
print("=============================================\n")

# 1. CARGA DE DATOS DE FIT (Ajuste EDR)
try:
    fit_df = pd.read_csv(RESULTS_CSV)
    # Usamos A (velocidad asintótica EDR) como Vflat de nuestro fit
    fit_df = fit_df[['Galaxy', 'A', 'Yd', 'Yb']].rename(columns={'A': 'Vflat_fit'}) 
except FileNotFoundError:
    print(f"[ERROR] No se encontró el archivo de resultados del fit en: {RESULTS_CSV}")
    exit()

# 2. CARGA DE DATOS GLOBALES (Table 1)
try:
    global_df = parse_table1(TABLE1_PATH)
    # Seleccionamos ID, Luminosidad, Masa de Gas (MHI) y Vflat de la tabla SPARC original
    global_df = global_df[['ID', 'L[3.6]', 'MHI', 'Vflat']].rename(columns={'ID': 'Galaxy', 'Vflat': 'Vflat_tab1'})
except FileNotFoundError:
    print(f"[ERROR] No se encontró Table 1 en: {TABLE1_PATH}. Asegúrate que {TABLE1_FILENAME} esté ahí.")
    exit()
except Exception as e:
    print(f"[ERROR] Falló el parseo de Table 1: {e}")
    exit()


# 3. MERGE Y PREPARACIÓN DE DATOS
# Unir resultados del fit con datos globales.
merged_df = pd.merge(fit_df, global_df, on='Galaxy', how='inner')

if merged_df.empty:
    print("[ERROR] No se pudo hacer merge. Verifica que los nombres de las galaxias coincidan en ambos CSV.")
    exit()

# 4. CÁLCULO DE MASA BARIÓNICA TOTAL (Mbar)
# Mbar = (L[3.6] * Yd) + (L[3.6] * Yb) + MHI
# Todas las componentes de masa se llevan a M_sol
merged_df['M_disk'] = merged_df['L[3.6]'] * merged_df['Yd'] * LUMINOSITY_UNIT
merged_df['M_bulge'] = merged_df['L[3.6]'] * merged_df['Yb'] * LUMINOSITY_UNIT
merged_df['M_gas'] = merged_df['MHI'] * LUMINOSITY_UNIT # Asumimos MHI ya está en unidades de masa*
merged_df['M_bar'] = merged_df['M_disk'] + merged_df['M_bulge'] + merged_df['M_gas']

# 5. EXTRACCIÓN DE V_flat Y LOGARITMOS
# Priorizamos la Vflat de la tabla SPARC (Vflat_tab1) si existe, si no, usamos la A del fit.
merged_df['V_flat_final'] = merged_df['Vflat_tab1'].combine_first(merged_df['Vflat_fit'])

# Filtrar datos válidos
btfr_data = merged_df[
    (merged_df['M_bar'] > 0) & 
    (merged_df['V_flat_final'] > 0)
].copy()

if len(btfr_data) < 3:
    print("[ERROR] Datos insuficientes para regresión BTFR después del merge y filtrado.")
    exit()

# Logaritmos: log(M_bar / 10^9) y log(V_flat)
btfr_data['log_Mbar'] = np.log10(btfr_data['M_bar'] / LUMINOSITY_UNIT)
btfr_data['log_Vflat'] = np.log10(btfr_data['V_flat_final'])

# 6. REGRESIÓN LINEAL (BTFR)
slope, intercept, r_value, p_value, std_err = stats.linregress(
    btfr_data['log_Mbar'], btfr_data['log_Vflat']
)

r_squared = r_value**2

print(f"\n--- Resultados de la Regresión BTFR (N={len(btfr_data)}) ---")
print(f"Ecuación: Log(V_flat) = {slope:.3f} * Log(M_bar) + {intercept:.3f}")
print(f"R^2 (Bondad de ajuste): {r_squared:.3f}")
print(f"P-valor: {p_value:.2g}")


# 7. GUARDAR Y PLOTEAR
# Guardar resultados intermedios
btfr_data[['Galaxy', 'M_bar', 'V_flat_final', 'log_Mbar', 'log_Vflat']].to_csv(
    os.path.join(OUTDIR, "btfr_data_results.csv"), index=False
)

# Generar el gráfico BTFR
out_path_plot = os.path.join(OUTDIR, "btfr_validation.png")
plot_btfr(
    btfr_data['log_Mbar'].values, 
    btfr_data['log_Vflat'].values, 
    slope, 
    intercept, 
    r_squared, 
    out_path_plot
)

print(f"\n[OK] Gráfico BTFR guardado en: {out_path_plot}")
print("\n>>> VALIDACIÓN BTFR COMPLETADA <<<")
