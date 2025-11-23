import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress
import sys 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # Ignorar warnings de log(0) temporalmente

# --- CONFIGURACIÓN DE RUTAS ---
RESULTS_FILE = Path("EDR/data/sparc/sparc_results.csv")
TABLE2_FILE = Path("EDR/data/sparc/SPARC_Lelli2016_Table2.txt")
OUT_DIR = Path("EDR/results/validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- FUNCIÓN DE CÁLCULO DE ACELERACIONES ---
def calculate_accelerations(df_rc, Yd, Yb, galaxy_id):
    """
    Calcula la aceleración observada (g_obs) y bariónica (g_bar)
    a partir de los datos de la curva de rotación y los parámetros de ajuste EDR.
    """
    df_rc = df_rc.copy()
    initial_rows = len(df_rc)
    
    # Columnas numéricas clave: 'R' (radio), 'Vobs' (observada), 'Vgas', 'Vdisk', 'Vbul', 'e_Vobs'
    numeric_cols = ['R', 'Vobs', 'Vgas', 'Vdisk', 'Vbul', 'e_Vobs']
    
    # 1. Conversión robusta a numérico
    for col in numeric_cols:
        # Convertir a numérico, forzando cualquier string no numérico (incluyendo '---', '-' o espacios) a NaN.
        df_rc[col] = pd.to_numeric(df_rc[col], errors='coerce')
        
    # 2. Limpieza de NaN estricta en las columnas críticas (solo filas completas)
    df_rc = df_rc.dropna(subset=['R', 'Vobs', 'Vgas', 'Vdisk', 'Vbul'])
    
    # 3. Filtrar radios no positivos (CRÍTICO: R debe ser > 0 para calcular g = V^2/R)
    df_rc = df_rc[df_rc['R'] > 0]
    
    # DEPURACIÓN: Si no queda nada, informamos
    if df_rc.empty:
        print(f"DEBUG: Galaxia {galaxy_id} descartada. Quedan 0 puntos después de limpiar NaN/Radio. (Initial: {initial_rows})")
        return df_rc

    # 4. Calcular aceleraciones
    # g_obs = Vobs^2 / R
    df_rc['Vobs_sq'] = df_rc['Vobs']**2
    df_rc['g_obs'] = df_rc['Vobs_sq'] / df_rc['R']
    
    # Vbar^2 = Vgas^2 + Yd*Vdisk^2 + Yb*Vbul^2
    df_rc['Vbar_sq'] = df_rc['Vgas']**2 + Yd * df_rc['Vdisk']**2 + Yb * df_rc['Vbul']**2
    # g_bar = Vbar^2 / R
    df_rc['g_bar'] = df_rc['Vbar_sq'] / df_rc['R']
    
    # 5. Filtro final: Aceleraciones positivas, finitas y reales
    df_rc = df_rc[(df_rc['g_obs'] > 0) & (df_rc['g_bar'] > 0)]
    df_rc = df_rc.replace([np.inf, -np.inf], np.nan).dropna(subset=['g_obs', 'g_bar'])
    
    final_rows = len(df_rc)
    if final_rows == 0:
        print(f"DEBUG: Galaxia {galaxy_id} descartada. Quedan 0 puntos después de calcular y filtrar aceleraciones.")
    elif final_rows < initial_rows:
        print(f"INFO: Galaxia {galaxy_id}: {final_rows} puntos válidos de {initial_rows} iniciales.")

    return df_rc

# --- CARGA Y PREPARACIÓN DE DATOS ---

# Cargar los resultados de los ajustes (donde están Yd y Yb)
try:
    df_res = pd.read_csv(RESULTS_FILE)
    df_res['Galaxy'] = df_res['Galaxy'].astype(str).str.strip()
    
    # Filtrar resultados con NaN en Yd o Yb (ajustes fallidos)
    df_res = df_res.dropna(subset=['Yd', 'Yb'])
    
    if df_res.empty:
        print(f"ERROR: Archivo de resultados '{RESULTS_FILE}' está vacío o todos los ajustes fallaron (Yd/Yb son NaN). Revise '{RESULTS_FILE}'.")
        sys.exit(1)
        
    fit_params = df_res.set_index('Galaxy')[['Yd', 'Yb']].to_dict('index')
    galaxies_to_plot = list(fit_params.keys())
    
    print(f"INFO: Galaxias con parámetros EDR válidos cargadas (N={len(galaxies_to_plot)}).")
    
except Exception as e:
    print(f"ERROR al cargar o procesar {RESULTS_FILE}. Detalle: {e}")
    sys.exit(1)


# Cargar los datos brutos de la curva de rotación (Table 2) - Usando separador de espacios
try:
    col_names_full = ['ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
    
    df_t2 = pd.read_csv(
        TABLE2_FILE,
        sep='\s+',  # Separador: uno o más espacios en blanco
        header=None,
        names=col_names_full,
        skiprows=42, # Saltar el encabezado (42 es el índice del primer dato si se cuenta desde el inicio del archivo)
        engine='python',
        skipinitialspace=True,
        dtype={'ID': str} 
    )
    
    df_t2['ID'] = df_t2['ID'].str.strip()
    
    # DEBUG CRÍTICO: Imprime las primeras filas del archivo leído
    print("\nDEBUG: Primeras 5 filas del archivo SPARC Table 2 (después de parsear):")
    print(df_t2.head().to_string())
    print("-" * 50)
    
    # Filtrar solo las galaxias que ajustamos previamente
    df_t2 = df_t2[df_t2['ID'].isin(galaxies_to_plot)]

    if df_t2.empty:
        print("\nERROR CRÍTICO: No hay COINCIDENCIA de IDs. Los nombres en 'sparc_results.csv' no coinciden con los de 'SPARC_Lelli2016_Table2.txt'.")
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
        
    # Saltamos si los parámetros no son válidos (ya filtrado, pero por doble seguridad)
    if Yd is None or Yb is None or np.isnan(Yd) or np.isnan(Yb) or Yd < 0 or Yb < 0:
        continue
        
    df_rc_g = df_t2[df_t2['ID'] == g]
    
    if df_rc_g.empty:
        continue 

    # Aquí es donde se calcula g_obs y g_bar
    df_accel = calculate_accelerations(df_rc_g, Yd, Yb, g)
    
    if not df_accel.empty:
        df_accel['Galaxy'] = g
        # Solo necesitamos 'g_obs' y 'g_bar' para la relación MDAR
        mdar_data.append(df_accel[['Galaxy', 'g_obs', 'g_bar']])


# --- RESULTADO FINAL DE CARGA ---

if not mdar_data:
    print("\nERROR CRÍTICO: Después de la limpieza final, NO QUEDÓ NINGÚN punto de dato válido (N=0).")
    print("VERIFICAR: 1. Contenido de Vobs, R, Vgas, Vdisk, Vbul. 2. Parámetros Yd/Yb.")
    sys.exit(1)
    
df_mdar = pd.concat(mdar_data)
N_points = len(df_mdar)

if N_points == 0:
    print("ERROR CRÍTICO: ¡df_mdar está vacío! N_points=0. Falló la concatenación.")
    sys.exit(1)

# Preparar datos para el plot y el cálculo de la correlación
epsilon = 1e-10 
df_mdar['log_g_obs'] = np.log10(df_mdar['g_obs'] + epsilon)
df_mdar['log_g_bar'] = np.log10(df_mdar['g_bar'] + epsilon)

g_bar_vals = df_mdar['g_bar'].values
g_obs_vals = df_mdar['g_obs'].values

print(f"\nINFO: Puntos de datos MDAR finales válidos (N_points): {N_points}")

# --- CÁLCULO DE CORRELACIÓN ---

r_value_log = np.nan

# Correlación de Pearson (r) en log-log
try:
    slope, intercept, r_value_log, p_value, std_err = linregress(df_mdar['log_g_bar'], df_mdar['log_g_obs'])
except ValueError:
    print("WARNING: Falló la regresión lineal (linregress). No hay suficientes datos para calcular la correlación.")
    pass


print("----------------------------------------------------------------")
print("           VALIDACIÓN MDAR (Mass Discrepancy-Acceleration Relation)")
print(f"          Usando parámetros de ajuste de su modelo EDR")
print("----------------------------------------------------------------")
print(f"Puntos de datos MDAR totales (N): {N_points}")
if not np.isnan(r_value_log):
    print(f"Correlación de Pearson (r) de log(g_obs) vs log(g_bar): {r_value_log:.4f}")
print("----------------------------------------------------------------")


# --- GRÁFICO DE LA RELACIÓN ACELERACIÓN-DISCREPANCIA DE MASA (MDAR) ---
if N_points >= 2:
    plt.figure(figsize=(8, 7))

    # 1. Puntos de Datos EDR
    plt.scatter(df_mdar['log_g_bar'], df_mdar['log_g_obs'], s=15, alpha=0.6, label='Su Modelo EDR (Datos de Curvas Ajustadas)', color='darkblue')

    # 2. Línea de Identidad (g_obs = g_bar)
    log_min = min(np.nanmin(df_mdar['log_g_bar']), np.nanmin(df_mdar['log_g_obs']))
    log_max = max(np.nanmax(df_mdar['log_g_bar']), np.nanmax(df_mdar['log_g_obs']))
    
    x_min_plot = log_min - 0.5
    x_max_plot = log_max + 0.5
    x_range = np.linspace(x_min_plot, x_max_plot, 100)
    plt.plot(x_range, x_range, 'k--', linewidth=1, label='$g_{obs} = g_{bar}$ (Sin Materia Oscura)')

    # 3. Curva MOND (como referencia estándar de la MDAR, NO para ajuste de su modelo)
    A0_GALACTIC = 3.70 # Constante MOND en (km/s)^2 / kpc.
    g_bar_theory = np.logspace(x_min_plot, x_max_plot, 100)
    g_obs_mond = np.sqrt(g_bar_theory**2 + np.abs(g_bar_theory) * A0_GALACTIC)
    
    plt.plot(np.log10(g_bar_theory), np.log10(np.abs(g_obs_mond)), 'r-', linewidth=2, alpha=0.7, label=f'Relación MDAR Teórica (MOND)')

    plt.xlabel('log$_{10}(g_{bar})$ $[log_{10}((km/s)^2/kpc)]$')
    plt.ylabel('log$_{10}(g_{obs})$ $[log_{10}((km/s)^2/kpc)]$')

    title_r = f" ($r_{{log}}={r_value_log:.4f}$)" if not np.isnan(r_value_log) else ""
    plt.title(f'Relación Aceleración-Discrepancia de Masa (MDAR) - Modelo EDR{title_r}')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box') 

    mdar_plot_path = OUT_DIR / "MDAR_plot_EDR_validation.png"
    plt.savefig(mdar_plot_path, dpi=200)
    plt.close()

    print(f"Gráfico MDAR guardado en: {mdar_plot_path}")
else:
    print(f"WARNING: No hay suficientes puntos ({N_points}) para generar el gráfico MDAR.")
    
print("\n--- Por favor, copia y pega TODO el resultado de la consola (incluyendo INFO y DEBUG) ---")
