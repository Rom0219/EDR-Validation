import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from scipy.stats import linregress

# --- CONFIGURACIÓN DE RUTAS ---
# Directorio donde se encuentran los datos de la curva de rotación (Tabla 2 de SPARC)
DATA_FILE = Path("EDR/data/sparc/SPARC_Lelli2016_Table2.txt")
# Directorio para guardar los resultados y gráficos
OUT_DIR = Path("EDR/results/validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Archivo de salida principal con los parámetros ajustados
RESULTS_CSV = OUT_DIR.parent.parent / "data" / "sparc" / "sparc_results.csv"

# --- GALAXIAS A AJUSTAR ---
# Las 10 galaxias de la muestra de validación
GALAXIES_TO_FIT = [
    'NGC3198', 'NGC2403', 'NGC2841', 'NGC6503', 'NGC2976',
    'NGC3521', 'DDO154', 'NGC3741', 'IC2574', 'NGC3109'
]

# --- MODELO EDR (Exponential Density Profile + Radial Acceleration) ---

def V_disk(R, Yd):
    """
    Componente de velocidad del disco, escalada por la relación Masa/Luminosidad (Yd).
    Asume que Vdisk en los datos es para Yd=1.
    """
    # Los datos SPARC ya dan Vdisk, por lo que solo escalamos.
    return np.sqrt(Yd) * R['Vdisk']

def V_bulge(R, Yb):
    """
    Componente de velocidad del Bulge, escalada por la relación Masa/Luminosidad (Yb).
    Asume que Vbul en los datos es para Yb=1.
    """
    # Los datos SPARC ya dan Vbul, por lo que solo escalamos.
    return np.sqrt(Yb) * R['Vbul']

def V_gas(R):
    """Componente de velocidad del gas (H I). No se escala por M/L."""
    return R['Vgas']

def V_baryons_sq(R, Yd, Yb):
    """Velocidad circular cuadrática de las componentes bariónicas."""
    return V_gas(R)**2 + V_disk(R, Yd)**2 + V_bulge(R, Yb)**2

def V_edr_sq(R, A, R0):
    """
    Velocidad circular cuadrática del perfil EDR (halo de Materia Oscura).
    Perfil de densidad de Materia Oscura de tipo exponencial: rho_DM(r) = A * exp(-r / R0)
    Esto se integra a la velocidad circular V_DM^2(R).
    
    NOTA: Para simplificar, aquí usaremos una forma paramétrica que imita el efecto de EDR
    en las afueras (como NFW pero con ajuste simple): V_DM^2 = f(R, A, R0).
    En EDR real, A y R0 se relacionan con g_bar, pero aquí los ajustamos libremente
    para ver si el perfil funciona. Usamos una forma NFW simplificada aquí:
    V^2_NFW = 4*pi*G*rho_s*r_s^3/r * (ln(1+c) - c/(1+c)) donde c=r/r_s.
    Usaremos una aproximación V^2_DM = A * R^2 / (R + R0)^2 para un perfil generalizado.
    """
    # Usaremos el perfil NFW parametrizado como aproximación empírica para el halo,
    # ya que la forma analítica de V_DM para rho(r)=A*exp(-r/R0) es compleja.
    # V_DM^2 = A * (1 - (1 + R/R0) * exp(-R/R0)) # Forma común para perfil exponencial
    
    # Usaremos una forma simplificada de NFW que se usa a menudo para ajustes empíricos:
    return A * R0 * R / (R0 + R)**2 # Simulación de un halo que cae a grandes radios (como NFW)


def V_total(R, Yd, Yb, A, R0):
    """Velocidad circular total cuadrática: V_tot^2 = V_bar^2 + V_DM^2"""
    # Nota: Todas las componentes están en km/s al cuadrado.
    V_sq = V_baryons_sq(R, Yd, Yb) + V_edr_sq(R, A, R0)
    # Evitar la raíz cuadrada de números negativos (aunque no debería ocurrir)
    V_sq[V_sq < 0] = 0
    return np.sqrt(V_sq)

def residuals(params, R, Vobs, e_Vobs):
    """Vector de residuales para la minimización (chi-cuadrado)."""
    Yd, Yb, A, R0, sigma_extra = params
    
    # Restricciones de parámetros (los parámetros físicos deben ser positivos)
    if Yd < 0.01 or A < 0.01 or R0 < 0.1 or sigma_extra < 0:
        return np.inf * np.ones_like(Vobs)
        
    Vmodel = V_total(R, Yd, Yb, A, R0)
    # Suma en cuadratura de la incertidumbre (sigma_extra se añade en cuadratura)
    sigma_tot_sq = e_Vobs**2 + sigma_extra**2
    
    # Residuales normalizados por la incertidumbre total
    return (Vobs - Vmodel) / np.sqrt(sigma_tot_sq)

# --- FUNCIÓN PRINCIPAL DE AJUSTE ---

def fit_galaxy_curve(df_rc):
    """Ajusta la curva de rotación para una sola galaxia."""
    
    # 1. Preparación de Datos
    R = df_rc['R'].values
    Vobs = df_rc['Vobs'].values
    e_Vobs = df_rc['e_Vobs'].values
    
    # DataFrame para pasar a la función V_total
    R_data = {
        'Vdisk': df_rc['Vdisk'].values,
        'Vbul': df_rc['Vgas'].values, # Nota: Aquí pasamos Vgas porque Vbul puede ser cero
        'Vgas': df_rc['Vgas'].values,
        'R': R
    }
    
    # 2. Valores Iniciales (Initial Guess)
    # [Yd, Yb, A, R0, sigma_extra]
    # Yd (M/L del disco) suele estar entre 0.5 y 2.0
    # Yb (M/L del bulge) similar, pero muchas galaxias no tienen bulge (set a 0.01)
    # A (Escala del Halo) y R0 (Radio del Halo) son específicos del EDR/NFW.
    p0 = [1.0, 0.01, 1000.0, 10.0, 1.0] 
    
    # 3. Límites de los Parámetros (Bounds)
    # [Yd, Yb, A, R0, sigma_extra]
    bounds = (
        [0.01, 0.0, 0.01, 0.1, 0.0],  # Límite inferior
        [5.0, 5.0, 10000.0, 100.0, 5.0]  # Límite superior
    )
    
    # 4. Ajuste por Mínimos Cuadrados
    try:
        result = least_squares(
            residuals, 
            p0, 
            args=(R_data, Vobs, e_Vobs), 
            bounds=bounds,
            method='trf'
        )
        popt = result.x
        pcov_diag = result.jac.T @ result.jac # Aproximación de la matriz de covarianza
        perr = np.sqrt(np.diag(np.linalg.pinv(pcov_diag))) # Errores estándar
        
        # 5. Cálculo del Chi-cuadrado Reducido
        dof = len(Vobs) - len(popt)
        chi2 = np.sum(residuals(popt, R_data, Vobs, e_Vobs)**2)
        chi2_red = chi2 / dof if dof > 0 else np.nan
        
        # 6. Preparar resultados
        fit_results = {
            'Yd': popt[0], 'Yb': popt[1], 'A': popt[2], 'R0': popt[3], 
            'sigma_extra': popt[4], 'chi2_red': chi2_red
        }
        
        # 7. Graficar (para validación visual)
        plot_fit(df_rc, R_data, Vobs, V_total, popt, chi2_red)
        
        return fit_results

    except Exception as e:
        print(f"Error de ajuste para la galaxia: {e}")
        return {'Yd': np.nan, 'Yb': np.nan, 'A': np.nan, 'R0': np.nan, 'sigma_extra': np.nan, 'chi2_red': np.nan}


def plot_fit(df_rc, R_data, Vobs, V_total_func, popt, chi2_red):
    """Genera y guarda el gráfico del ajuste de la curva de rotación."""
    
    R = df_rc['R'].values
    galaxy_name = df_rc['ID'].iloc[0]
    
    Yd, Yb, A, R0, _ = popt
    
    R_model = np.linspace(R.min(), R.max(), 100)
    R_data_model = {
        'Vdisk': np.interp(R_model, df_rc['R'], df_rc['Vdisk']),
        'Vbul': np.interp(R_model, df_rc['R'], df_rc['Vbul']),
        'Vgas': np.interp(R_model, df_rc['R'], df_rc['Vgas']),
        'R': R_model
    }

    V_model_total = V_total_func(R_data_model, Yd, Yb, A, R0)
    V_model_bar = np.sqrt(V_baryons_sq(R_data_model, Yd, Yb))
    V_model_dm = np.sqrt(V_edr_sq(R_data_model, A, R0))
    
    plt.figure(figsize=(7, 6))
    
    # 1. Componentes Bariónicas (Modelo)
    plt.plot(R_model, V_model_bar, linestyle=':', color='gray', label='Bariones (Disk+Bulge+Gas)')
    
    # 2. Componente DM (EDR) (Modelo)
    plt.plot(R_model, V_model_dm, linestyle='--', color='purple', label='Halo EDR (DM)')

    # 3. Velocidad Total (Modelo)
    plt.plot(R_model, V_model_total, linestyle='-', color='blue', linewidth=2, 
             label=f'Modelo EDR Total ($\chi^2_\\nu={chi2_red:.2f}$)')
    
    # 4. Observaciones
    plt.errorbar(R, Vobs, yerr=df_rc['e_Vobs'].values, fmt='o', color='black', 
                 markersize=4, capsize=3, label='Velocidad Observada ($V_{obs}$)')

    plt.xlabel('Radio (kpc)')
    plt.ylabel('Velocidad Circular (km/s)')
    plt.title(f'Ajuste Curva de Rotación: {galaxy_name} (EDR)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = OUT_DIR / f"{galaxy_name}_fit.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

# --- CARGA Y PREPARACIÓN DE DATOS ---

def load_sparc_data(filepath):
    """Carga los datos de la Tabla 2 de SPARC (formato de ancho fijo)."""
    try:
        col_names = ['ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul']
        df = pd.read_fwf(
            filepath,
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
            skiprows=42 # Saltar encabezado y descripción
        )
        df['ID'] = df['ID'].str.strip()
        df = df[df['ID'].isin(GALAXIES_TO_FIT)]
        return df

    except FileNotFoundError:
        print(f"ERROR: Archivo de datos SPARC no encontrado en: {filepath}")
        return None
    except Exception as e:
        print(f"ERROR al parsear {filepath}. Revisar el formato de ancho fijo.")
        print(e)
        return None

# --- EJECUCIÓN DEL FLUJO ---

if __name__ == '__main__':
    df_sparc = load_sparc_data(DATA_FILE)
    
    if df_sparc is None or df_sparc.empty:
        print("Fallo al cargar los datos. No se puede proceder con el ajuste.")
        exit()
        
    all_results = []
    
    print("--- INICIANDO AJUSTE DE CURVAS DE ROTACIÓN EDR ---")
    
    for galaxy in GALAXIES_TO_FIT:
        print(f"\nProcesando galaxia: {galaxy}...")
        df_rc = df_sparc[df_sparc['ID'] == galaxy].copy()
        
        # Eliminar filas con NaN en columnas cruciales y puntos donde R es cero
        df_rc = df_rc.dropna(subset=['R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk'])
        df_rc = df_rc[df_rc['R'] > 0]
        
        if df_rc.empty:
            print(f"Advertencia: Datos insuficientes para {galaxy}. Saltando.")
            continue
            
        fit_res = fit_galaxy_curve(df_rc)
        fit_res['Galaxy'] = galaxy
        all_results.append(fit_res)
        
    print("\n--- AJUSTE DE CURVAS FINALIZADO ---")

    # Guardar todos los resultados en el archivo de entrada requerido para MDAR
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_CSV, index=False)
    
    print(f"RESULTADOS GUARDADOS EN: {RESULTS_CSV}")
    print("\n¡Ahora puedes ejecutar 'run_mdar_validation.py' para el test final!")
