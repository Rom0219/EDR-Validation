"""
run_btfr_validation.py — Pipeline de Validación Global (BTFR)

- Carga la Tabla 1 de SPARC (Datos Globales)
- Realiza la regresión log(Vflat) vs log(Mbar) para validar la Relación Tully-Fisher Bariónica (BTFR)
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
from sparc_utils import parse_table1, plot_btfr, TABLE1_FILENAME

# ---- CONFIG & INPUTS ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "sparc")
TABLE1_PATH = os.path.join(DATA_DIR, TABLE1_FILENAME)

FIT_RESULTS_CSV = os.path.join(BASE_DIR, "..", "results", "sparc_results.csv") 
OUTDIR = os.path.join(BASE_DIR, "..", "results", "btfr_validation")
os.makedirs(OUTDIR, exist_ok=True)

# Factores de M/L canónicos (SPS)
Y_DISK_SPS = 0.5 
Y_BULB_SPS = 0.7 

print("=============================================")
print(f"     PROCESO SPARC + EDR — VALIDACIÓN GLOBAL (BTFR)")
print(f"     Usando archivo: {TABLE1_FILENAME}")
print("=============================================\n")

# ---- PASO 1: CARGAR DATOS GLOBALES (TABLA 1) ----
if not os.path.exists(TABLE1_PATH):
    print(f"[ERROR] Archivo de Tabla 1 no encontrado en: {TABLE1_PATH}. ¡Revisa la ruta!")
    exit()

try:
    df_global = parse_table1(TABLE1_PATH)
    df_global = df_global.rename(columns={"L[3.6]": "L_3p6", "MHI": "M_gas", "Vflat": "V_flat", "ID": "Galaxy"})
except Exception as e:
    print(f"[FATAL] Error al parsear Tabla 1: {e}")
    exit()

# ---- PASO 2: CARGAR FACTORES M/L Y CALCULAR M_bar ----
df_merged = df_global.copy()
use_sps = True

# Intenta usar los Yd/Yb ajustados de los fits locales
try:
    df_fits = pd.read_csv(FIT_RESULTS_CSV)
    df_fits = df_fits[["Galaxy", "Yd", "Yb"]].rename(columns={"Yd": "Yd_fit", "Yb": "Yb_fit"})
    df_merged = pd.merge(df_global, df_fits, on="Galaxy", how="left")
    
    df_merged["Yd_used"] = df_merged["Yd_fit"].fillna(Y_DISK_SPS)
    df_merged["Yb_used"] = df_merged["Yb_fit"].fillna(Y_BULB_SPS)
    
    if df_merged["Yd_fit"].count() > 0:
        use_sps = False
    
except FileNotFoundError:
    pass # Continúa usando SPS si el archivo no existe

if use_sps:
    print(f"[INFO] Factores M/L (SPS) utilizados: Yd={Y_DISK_SPS}, Yb={Y_BULB_SPS}")

df_btfr = df_merged.copy()
# Calculo de M_star asumiendo simplificación: M_star = Yd_used * L_3p6 (Luminosidad total)
df_btfr["M_star"] = df_btfr["L_3p6"] * df_btfr["Yd_used"] 
df_btfr["M_bar"] = df_btfr["M_star"] + df_btfr["M_gas"] 

# Filtrar datos válidos
df_btfr = df_btfr.dropna(subset=["M_bar", "V_flat"]).query("V_flat > 0").copy()
print(f"[INFO] Calculando BTFR con {len(df_btfr)} galaxias con datos completos.")

if len(df_btfr) < 5:
    print("[ERROR] Datos insuficientes para BTFR después de la limpieza. Finalizando.")
    exit()

# ---- PASO 3: REGRESIÓN LOG-LOG ----
log_V = np.log10(df_btfr["V_flat"].values)
log_M = np.log10(df_btfr["M_bar"].values)

slope, intercept, r_value, p_value, std_err = stats.linregress(log_M, log_V)
r_squared = r_value**2

print(f"\n[BTFR RESULTS] r = {r_value:.4f}, r^2 = {r_squared:.4f}")

# ---- PASO 4: PLOT BTFR ----
out_plot = os.path.join(OUTDIR, "btfr_edr_validation.png")
plot_btfr(log_M, log_V, slope, intercept, r_squared, out_plot)

print(f"[OK] Plot BTFR guardado en: {out_plot}")
print("\n>>> VALIDACIÓN GLOBAL (BTFR) COMPLETADA <<<")
