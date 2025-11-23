import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress
import sys 

# --- CONFIGURACIÓN DE RUTAS ---
# ARCHIVOS REQUERIDOS:
# 1. sparc_results.csv: Contiene los mejores ajustes (Yd, Yb) para cada galaxia.
# 2. SPARC_Lelli2016_Table2.txt: Contiene los datos brutos de las curvas de rotación (R, Vobs, Vgas, Vdisk, Vbul).
RESULTS_FILE = Path("EDR/data/sparc/sparc_results.csv")
TABLE2_FILE = Path("EDR/data/sparc/SPARC_Lelli2016_Table2.txt")
OUT_DIR = Path("EDR/results/validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constante MOND en unidades de (km/s)^2 / kpc.
# Usamos el valor canónico para la curva teórica de referencia
A0_GALACTIC = 3.70 

# --- MODELO MOND ---
def mond_interpolation(g_bar, a0):
    """ g_obs = sqrt(g_bar^2 + g_bar * a0) """
    # Se usa np.abs(g_bar) para evitar errores si g_bar fuera negativo
    return np.sqrt(g_bar**2 + np.abs(g_bar) * a0)

# --- CÁLCULO DE ACELERACIONES (g = V^2 / R) ---
def calculate_accelerations(df_rc, Yd, Yb):
    """
    Calcula la aceleración observada (g_obs = Vobs^2 / R) y bariónica (g_bar = Vbar^2 / R).
    Asegura que las columnas sean numéricas y limpia los datos inválidos.
    """
    df_rc = df_rc.copy()
    
    # Asegurar que R y Velocidades sean numéricas (CRÍTICO)
    numeric_cols = ['R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
    for col in numeric_cols:
        # 'coerce' convierte valores problemáticos (como '---' o '-') a NaN
        df_rc[col] = pd.to_numeric(df_rc[col], errors='coerce')
        
    # Limpieza de NaN, radio cero, y componentes inválidas
    df_rc = df_rc.dropna(subset=['R', 'Vobs', 'Vgas', 'Vdisk', 'Vbul'])
    df_rc = df_rc[df_rc['R'] > 0]
    
    if df_rc.empty:
        return df_rc

    # Calcular V^2 y aceleraciones
    df_rc['Vobs_sq'] = df_rc['Vobs']**2
    df_rc['g_obs'] = df_rc['Vobs_sq'] / df_rc['R']
    
    # Calcular la velocidad bariónica al cuadrado (Vbar^2 = Vgas^2 + Yd*Vdisk^2 + Yb*Vbul^2)
    # Nota: Vgas, Vdisk, Vbul son las velocidades de los componentes, por lo que V^2 = V_componente^2
    df_rc['Vbar_sq'] = df_rc['Vgas']**2 + Yd * df_rc['Vdisk']**2 + Yb * df_rc['Vbul']**2
    df_rc['g_bar'] = df_rc['Vbar_sq'] / df_rc['R']
    
    # Filtro final: Aceleraciones positivas y finitas
    df_rc = df_rc[(df_rc['g_obs'] > 0) & (df_rc['g_bar'] > 0)]
    df_rc = df_rc.replace([np.inf, -np.inf], np.nan).dropna(subset=['g_obs', 'g_bar'])

    return df_rc

# --- CARGA Y PREPARACIÓN DE DATOS ---

# Cargar los resultados de los ajustes (donde están Yd y Yb)
try:
    df_res = pd.read_csv(RESULTS_FILE)
    df_res['Galaxy'] = df_res['Galaxy'].astype(str).str.strip()
    fit_params = df_res.set_index('Galaxy')[['Yd', 'Yb']].to_dict('index')
    
    if not fit_params:
        print(f"ERROR: Archivo de resultados '{RESULTS_FILE}' está vacío.")
        sys.exit(1)
    
    galaxies_to_plot = list(fit_params.keys())
    print(f"INFO: Galaxias cargadas para validación (N={len(galaxies_to_plot)}): {', '.join(galaxies_to_plot[:5])}...")
    
except FileNotFoundError:
    print(f"ERROR: Archivo de resultados NO encontrado en: {RESULTS_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR al cargar o procesar {RESULTS_FILE}. Detalle: {e}")
    sys.exit(1)


# Cargar los datos brutos de la curva de rotación (Table 2) - MÉTODO ROBUSTO Y FORZADO
try:
    # Nombres de las columnas del archivo SPARC Table 2
    col_names_full = ['ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
    
    df_t2 = pd.read_csv(
        TABLE2_FILE,
        sep='\s+',  # Separador: uno o más espacios en blanco
        header=None,
        names=col_names_full,
        skiprows=42, # Saltar el encabezado y descripciones (42 líneas)
        engine='python', # Usar el motor Python para mejor control de separación
        skipinitialspace=True,
        # *** FIX CRÍTICO: Forzar la columna ID a ser string para el matching ***
        dtype={'ID': str} 
    )
    
    # Limpiar espacios alrededor del ID
    df_t2['ID'] = df_t2['ID'].str.strip()
    
    # DEBUG CRÍTICO: Imprime las primeras filas del archivo leído
    print("\nDEBUG: Primeras 5 filas del archivo SPARC Table 2 (después de parsear):")
    print(df_t2.head().to_string())
    print("-" * 50)
    
    # Filtrar solo las galaxias que ajustamos previamente
    df_t2 = df_t2[df_t2['ID'].isin(galaxies_to_plot)]

    if df_t2.empty:
        print("\nERROR CRÍTICO: No hay coincidencia de IDs. Esto significa que los nombres en 'sparc_results.csv' no coinciden con los de 'SPARC_Lelli2016_Table2.txt'.")
        print(f"IDs en sparc_results.csv (ajustes): {galaxies_to_plot[:5]}...")
        # Imprimir IDs únicos de la tabla 2 que se lograron leer
        read_ids = pd.read_csv(
            TABLE2_FILE,
            sep='\s+', header=None, names=['ID'] + ['_'] * (len(col_names_full) - 1), 
            skiprows=42, engine='python', skipinitialspace=True, dtype={'ID': str}
        )['ID'].str.strip().unique()
        print(f"IDs leídos en Table2 (muestra): {read_ids[:5]}...")
        sys.exit(1)
    
    print(f"INFO: Puntos de datos brutos cargados y filtrados (N={len(df_t2)}).")
    
except Exception as e:
    print(f"ERROR al parsear {TABLE2_FILE}. Detalle: {e}")
    sys.exit(1)


# --- CÁLCULO DE MDAR PARA TODA LA MUESTRA ---

mdar_data = []

for g, params in fit_params.items():
    Yd = params.get('Yd')
    Yb = params.get('Yb')
    
    # Validación de parámetros
    if Yd is None or Yb is None or np.isnan(Yd) or np.isnan(Yb) or Yd < 0 or Yb < 0:
        continue
        
    df_rc_g = df_t2[df_t2['ID'] == g]
    
    if df_rc_g.empty:
        # Esta galaxia existe en los resultados, pero no en la tabla 2 después del filtrado.
        continue

    df_accel = calculate_accelerations(df_rc_g, Yd, Yb)
    
    if not df_accel.empty:
        df_accel['Galaxy'] = g
        mdar_data.append(df_accel[['Galaxy', 'R', 'g_obs', 'g_bar']])

if not mdar_data:
    print("\nERROR CRÍTICO: Después de calcular g_obs y g_bar, no queda NINGÚN punto válido.")
    print("Causa probable: Los valores R, Vobs, Vgas, Vdisk, Vbul en la Tabla 2 contienen muchos datos faltantes ('---', '-') o valores no positivos que fueron descartados.")
    sys.exit(1)
    
df_mdar = pd.concat(mdar_data)
# Agregar un pequeño valor (epsilon) para evitar log(0)
epsilon = 1e-10 
df_mdar['log_g_obs'] = np.log10(df_mdar['g_obs'] + epsilon)
df_mdar['log_g_bar'] = np.log10(df_mdar['g_bar'] + epsilon)

# --- REGRESIÓN Y CÁLCULO DE CORRELACIÓN ---

g_bar_vals = df_mdar['g_bar'].values
g_obs_vals = df_mdar['g_obs'].values
N_points = len(df_mdar)

print(f"\nINFO: Puntos de datos MDAR finales válidos (N_points): {N_points}")

if N_points == 0:
    print("ERROR CRÍTICO: ¡ydata está vacío! N_points=0. Falló la limpieza de datos en el bucle.")
    sys.exit(1)

r_value_log = np.nan
a0_fit = np.nan
R_squared_fit = np.nan

# 1. Ajuste no lineal MOND-like
try:
    popt, pcov = curve_fit(mond_interpolation, g_bar_vals, g_obs_vals, p0=[A0_GALACTIC], maxfev=5000)
    a0_fit = popt[0]
    
    residuals_fit = g_obs_vals - mond_interpolation(g_bar_vals, a0_fit)
    ss_res_fit = np.sum(residuals_fit**2)
    ss_tot = np.sum((g_obs_vals - np.mean(g_obs_vals))**2)
    R_squared_fit = 1 - (ss_res_fit / ss_tot)
    
except RuntimeError:
    print("WARNING: Falló el ajuste no lineal (curve_fit). Saltando métricas MOND-like.")
    pass
except Exception as e:
    print(f"WARNING: Falló el ajuste MOND por una excepción: {e}")
    pass


# 2. Correlación de Pearson (r) en log-log
try:
    slope, intercept, r_value_log, p_value, std_err = linregress(df_mdar['log_g_bar'], df_mdar['log_g_obs'])
except ValueError:
    print("WARNING: Falló la regresión lineal (linregress). Saltando métrica de correlación.")
    pass


print("----------------------------------------------------------------")
print("           RESULTADOS DE LA VALIDACIÓN MDAR (EDR)")
print("----------------------------------------------------------------")
print(f"Puntos de datos MDAR totales (N): {N_points}")
if not np.isnan(a0_fit):
    print(f"Ajuste MDAR (MOND-like):")
    print(f"  a_0 ajustado: {a0_fit:.3f} (km/s)^2 / kpc")
    print(f"  R^2 vs. curva MOND-like: {R_squared_fit:.4f}")
if not np.isnan(r_value_log):
    print(f"Correlación de Pearson (r) de log(g_obs) vs log(g_bar): {r_value_log:.4f}")
print("----------------------------------------------------------------")


# --- GRÁFICO DE LA RELACIÓN ACELERACIÓN-DISCREPANCIA DE MASA (MDAR) ---

plt.figure(figsize=(8, 7))

# 1. Puntos de Datos EDR
plt.scatter(df_mdar['log_g_bar'], df_mdar['log_g_obs'], s=10, alpha=0.5, label='Modelo EDR (Datos de Curvas Ajustadas)')

# 2. Línea de Identidad (g_obs = g_bar)
# Se usa np.nanmin y np.nanmax para manejar posibles NaNs
log_min_x = np.nanmin(df_mdar['log_g_bar'])
log_max_y = np.nanmax(df_mdar['log_g_obs'])
x_range = np.linspace(min(log_min_x, -2) - 0.5, max(log_max_y, 2) + 0.5, 100)
plt.plot(x_range, x_range, 'k--', label='$g_{obs} = g_{bar}$ (Bariones Dominan)')

# 3. Curva Teórica MOND (Usando a_0 canónico)
g_bar_theory = np.logspace(log_min_x, log_max_y, 100)
g_obs_mond = mond_interpolation(g_bar_theory, A0_GALACTIC)
plt.plot(np.log10(g_bar_theory), np.log10(np.abs(g_obs_mond)), 'r-', linewidth=2, label=f'MDAR Teórico (MOND $a_0 \\approx {A0_GALACTIC:.2f}$)')

plt.xlabel('log$_{10}(g_{bar})$ $[log_{10}((km/s)^2/kpc)]$')
plt.ylabel('log$_{10}(g_{obs})$ $[log_{10}((km/s)^2/kpc)]$')

title_r = f" (r={r_value_log:.4f})" if not np.isnan(r_value_log) else ""
plt.title(f'Relación MDAR - EDR vs. MOND{title_r}')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.gca().set_aspect('equal', adjustable='box') 

mdar_plot_path = OUT_DIR / "MDAR_plot_EDR_vs_MOND.png"
plt.savefig(mdar_plot_path, dpi=200)
plt.close()

print(f"Gráfico MDAR guardado en: {mdar_plot_path}")
print("\n--- Por favor, copia y pega TODO el resultado de la consola (incluyendo INFO y DEBUG) ---")
