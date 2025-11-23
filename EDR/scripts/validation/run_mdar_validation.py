import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# --- CONFIGURACIÓN DE RUTAS ---
# Archivo de resultados de ajustes individuales (contiene Yd y Yb)
RESULTS_FILE = Path("EDR/data/sparc/sparc_results.csv")
# Archivo de datos de la curva de rotación (Tabla 2 de SPARC)
TABLE2_FILE = Path("EDR/data/sparc/SPARC_Lelli2016_Table2.txt")
# Directorio de salida
OUT_DIR = Path("EDR/results/validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constante de conversión: 1 km^2 s^-2 kpc^-1 a m s^-2
# (1 km/s)^2 / 1 kpc = 1e6 m^2/s^2 / 3.086e19 m = 3.24e-14 m/s^2
# Aceleración característica de MOND (a_0) en unidades de (km/s)^2 / kpc
# a_0 = 1.2 x 10^-10 m/s^2. En unidades galácticas: ~ 3.70 (km/s)^2 / kpc
UNIT_ACCEL_FACTOR = 3.24e-14 # Factor de conversión si se necesitara a_0 en m/s^2, pero lo mantendremos en unidades galácticas para el gráfico
A0_GALACTIC = 3.70 # Valor de a_0 en (km/s)^2 / kpc (aproximado)

# --- MODELO MOND (PARA COMPARACIÓN) ---
# Función de interpolación estándar (ejemplo: función 'simple' o 'canonical' de MOND)
def mond_interpolation(g_bar, a0):
    """
    Función de interpolación MOND (canonical: g_obs = g_bar * mu + g_DM * (1-mu) )
    Aquí usamos la forma más simple: g_obs^2 = g_bar * a0 + g_bar^2
    """
    return np.sqrt(g_bar**2 + g_bar * a0)

# --- CÁLCULO DE ACELERACIONES (g = V^2 / R) ---
def calculate_accelerations(df_rc, Yd, Yb):
    """Calcula la aceleración observada y bariónica para cada punto de datos."""
    df_rc = df_rc.copy()
    
    # Pre-limpieza y manejo de cero
    df_rc['R'] = df_rc['R'].replace(0, np.nan)
    df_rc = df_rc.dropna(subset=['R'])

    # 1. Aceleración Observada (g_obs): g_obs = V_obs^2 / R
    df_rc['Vobs_sq'] = df_rc['Vobs']**2
    df_rc['g_obs'] = df_rc['Vobs_sq'] / df_rc['R']
    
    # 2. Aceleración Bariónica (g_bar): g_bar = (V_gas^2 + Yd*V_disk^2 + Yb*V_bul^2) / R
    
    # NOTA: V_disk y V_bul en el archivo SPARC ya incluyen la normalización para Yd=1, Yb=1.
    # La velocidad bariónica total al cuadrado es:
    # V_bar^2 = V_gas^2 + Yd*V_disk^2 + Yb*V_bul^2
    
    # Los V's en Table 2 son componentes de velocidad (no cuadradas)
    # Vgas, Vdisk, Vbul son las contribuciones a la velocidad circular (ej: Vgas = sqrt(g_gas * R))
    # Por lo tanto, g_bar = g_gas + g_disk + g_bul
    # g_gas = V_gas^2 / R, g_disk = V_disk^2 / R (pero escalado por Yd), g_bul = V_bul^2 / R (escalado por Yb)
    
    # La masa total de bariones es M_bar = Yd*M_disk + Yb*M_bul + M_gas
    # La aceleración total de bariones es g_bar = g_gas + Yd*g_disk + Yb*g_bul (asumiendo que g_disk/g_bul son de Y=1)
    
    # Velocidad al cuadrado de las componentes bariónicas (escaladas por M/L)
    df_rc['Vbar_sq'] = df_rc['Vgas']**2 + Yd * (df_rc['Vdisk']**2) + Yb * (df_rc['Vbul']**2)
    
    # Aceleración Bariónica total
    df_rc['g_bar'] = df_rc['Vbar_sq'] / df_rc['R']
    
    # Filtro: Evitar valores no físicos o infinitos
    df_rc = df_rc[(df_rc['g_obs'] > 0) & (df_rc['g_bar'] > 0)]

    return df_rc

# --- CARGA Y PREPARACIÓN DE DATOS ---

# Cargar los resultados de los ajustes (donde están Yd y Yb)
try:
    df_res = pd.read_csv(RESULTS_FILE)
    df_res['Galaxy'] = df_res['Galaxy'].str.strip()
    fit_params = df_res.set_index('Galaxy')[['Yd', 'Yb']].to_dict('index')
except FileNotFoundError:
    print(f"ERROR: Archivo de resultados no encontrado en: {RESULTS_FILE}")
    print("Asegúrate de haber corrido primero el script de ajuste de curvas de rotación.")
    exit()

# Cargar los datos brutos de la curva de rotación (Table 2)
# Hay que parsear el archivo de texto de formato fijo, no es CSV simple
try:
    # Usamos pandas read_fwf (fixed width file) basado en la descripción de Table2.mrt
    # ID: 1-11, R: 20-25, Vobs: 27-32, Vgas: 40-45, Vdisk: 47-52, Vbul: 54-59
    col_names = ['ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul']
    col_widths = [11, 1, 6, 1, 6, 1, 5, 1, 6, 1, 6, 1, 5, 1, 6, 1, 6]
    # Suma de anchos para las columnas que nos interesan (ver descripción Byte-by-byte)
    # 1-11 ID, 13-18 D, 20-25 R, 27-32 Vobs, 34-38 e_Vobs, 40-45 Vgas, 47-52 Vdisk, 54-59 Vbul
    # Los indices son 0-based, el width de los separadores se incluye en el skip
    
    df_t2 = pd.read_fwf(
        TABLE2_FILE,
        colspecs=[
            (0, 11),  # ID
            (12, 18), # D
            (19, 25), # R
            (26, 32), # Vobs
            (33, 38), # e_Vobs
            (39, 45), # Vgas
            (46, 52), # Vdisk
            (53, 59), # Vbul
        ],
        header=None,
        names=col_names,
        skiprows=42 # Saltar el encabezado de la descripción del archivo
    )
    df_t2['ID'] = df_t2['ID'].str.strip()
    
    # Filtrar solo las galaxias que tenemos en el archivo de resultados (las 10 ajustadas)
    galaxies_to_plot = list(fit_params.keys())
    df_t2 = df_t2[df_t2['ID'].isin(galaxies_to_plot)]

    if df_t2.empty:
        print("ERROR: No se encontraron datos de curva de rotación para las galaxias en el archivo de resultados.")
        exit()

except FileNotFoundError:
    print(f"ERROR: Archivo de datos de curva de rotación no encontrado en: {TABLE2_FILE}")
    exit()
except Exception as e:
    print(f"ERROR al parsear {TABLE2_FILE}. Revisar el formato de ancho fijo.")
    print(e)
    exit()


# --- CÁLCULO DE MDAR PARA TODA LA MUESTRA ---

mdar_data = []

for g, params in fit_params.items():
    Yd = params.get('Yd', 1.0)
    Yb = params.get('Yb', 0.0)
    
    # 1. Obtener los datos de la curva de rotación para la galaxia actual
    df_rc_g = df_t2[df_t2['ID'] == g].copy()
    
    if df_rc_g.empty:
        print(f"Advertencia: Datos de curva de rotación faltantes para {g}")
        continue
        
    # 2. Calcular las aceleraciones usando los Yd/Yb ajustados
    df_accel = calculate_accelerations(df_rc_g, Yd, Yb)
    
    # 3. Guardar los datos para el plot global
    df_accel['Galaxy'] = g
    mdar_data.append(df_accel[['Galaxy', 'R', 'g_obs', 'g_bar']])

if not mdar_data:
    print("No se generaron datos de MDAR. Revise sus archivos de entrada.")
    exit()
    
df_mdar = pd.concat(mdar_data)
df_mdar['log_g_obs'] = np.log10(df_mdar['g_obs'])
df_mdar['log_g_bar'] = np.log10(df_mdar['g_bar'])

# --- REGRESIÓN (Opcional, solo para medir dispersión) ---

# Se ajusta la MDAR observada para encontrar la dispersión
x = df_mdar['log_g_bar'].values
y = df_mdar['log_g_obs'].values
# Fit the power law relationship: log(g_obs) = m * log(g_bar) + c
# La MDAR no es lineal en log-log, pero la dispersión se mide contra el MOND teórico
# En su lugar, comparamos la dispersión de los residuales con la dispersión de MOND.

# Para simplificar, ajustamos la función MOND a los datos (en lugar de una regresión lineal)
# MOND predice g_obs = mond_interpolation(g_bar, a0_fit)
g_bar_vals = df_mdar['g_bar'].values
g_obs_vals = df_mdar['g_obs'].values

# Fit a_0 to the data (this is NOT what MOND does, but useful for comparison)
try:
    popt, pcov = curve_fit(mond_interpolation, g_bar_vals, g_obs_vals, p0=[A0_GALACTIC], maxfev=5000)
    a0_fit = popt[0]
    # Calculate R-squared (coefficient of determination) against the MOND-like fit
    residuals = g_obs_vals - mond_interpolation(g_bar_vals, a0_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((g_obs_vals - np.mean(g_obs_vals))**2)
    R_squared = 1 - (ss_res / ss_tot)
    r_value = np.sqrt(R_squared)
    
    print("----------------------------------------------------------------")
    print("           RESULTADOS DE LA VALIDACIÓN MDAR (EDR)")
    print("----------------------------------------------------------------")
    print(f"Ajuste MDAR (MOND-like, g_obs^2 = g_bar^2 + g_bar * a_0_fit):")
    print(f"  a_0 ajustado: {a0_fit:.3f} (km/s)^2 / kpc")
    print(f"  Correlación (r) vs. curva MOND-like: {r_value:.4f}")
    print(f"  Coef. de Determinación (R^2): {R_squared:.4f}")
    print(f"  Valor MOND canónico (r): ~ 0.98 a 0.99")
    print("----------------------------------------------------------------")

except RuntimeError:
    a0_fit = A0_GALACTIC
    R_squared = np.nan
    r_value = np.nan
    print("Advertencia: Falló el ajuste no lineal. Usando a_0 canónico para la MDAR teórica.")

# --- GRÁFICO DE LA RELACIÓN ACELERACIÓN-DISCREPANCIA DE MASA (MDAR) ---

plt.figure(figsize=(8, 7))

# 1. Puntos de Datos EDR
plt.scatter(df_mdar['log_g_bar'], df_mdar['log_g_obs'], s=10, alpha=0.5, label='Modelo EDR (Datos de Curvas Ajustadas)')

# 2. Línea de Identidad (g_obs = g_bar)
x_range = np.linspace(df_mdar['log_g_bar'].min() - 0.5, df_mdar['log_g_obs'].max() + 0.5, 100)
plt.plot(x_range, x_range, 'k--', label='$g_{obs} = g_{bar}$ (Bariones Dominan)')

# 3. Curva Teórica MOND (Usando a_0 canónico)
g_bar_theory = np.logspace(np.min(x), np.max(x), 100)
g_obs_mond = mond_interpolation(g_bar_theory, A0_GALACTIC)
plt.plot(np.log10(g_bar_theory), np.log10(g_obs_mond), 'r-', linewidth=2, label=f'MDAR Teórico (MOND $a_0 \\approx {A0_GALACTIC:.2f}$)')

plt.xlabel('log$_{10}(g_{bar})$ $[log_{10}((km/s)^2/kpc)]$')
plt.ylabel('log$_{10}(g_{obs})$ $[log_{10}((km/s)^2/kpc)]$')
plt.title('Relación Aceleración-Discrepancia de Masa (MDAR) - EDR vs. MOND')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.axis('equal') # Importante para ver la dispersión

mdar_plot_path = OUT_DIR / "MDAR_plot_EDR_vs_MOND.png"
plt.savefig(mdar_plot_path, dpi=200)
plt.close()

print(f"Gráfico MDAR guardado en: {mdar_plot_path}")
print("\n--- ¡Validación MDAR completada! Revisa el gráfico y el valor de correlación (r). ---\n")
