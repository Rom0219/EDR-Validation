import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
import warnings
# Ignorar warnings de log(0) temporalmente, comunes con el perfil EDR en R pequeños
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# --- CONFIGURACIÓN DE RUTAS ---
# Archivo de datos SPARC (Tabla 2)
DATA_FILE = Path("EDR/data/sparc/SPARC_Lelli2016_Table2.txt")
# Directorio para guardar resultados
OUT_DIR = Path("EDR/results/validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Archivo de salida con los parámetros ajustados para 175 galaxias
RESULTS_CSV = OUT_DIR.parent.parent / "data" / "sparc" / "sparc_results_175.csv" 
# Flag para saltar la generación de gráficos individuales
SKIP_PLOTTING = True 

# --- GALAXIAS A AJUSTAR ---
GALAXIES_TO_FIT = 'ALL' # Usar la muestra completa de 175

# --- MODELO EDR (Exponential Density Profile + Radial Acceleration) ---

def V_disk(R_data, Yd):
    """Componente de velocidad del disco, escalada por la relación Masa/Luminosidad (Yd)."""
    return np.sqrt(Yd) * R_data['Vdisk']

def V_bulge(R_data, Yb):
    """Componente de velocidad del Bulge, escalada por la relación Masa/Luminosidad (Yb)."""
    return np.sqrt(Yb) * R_data['Vbul']

def V_gas(R_data):
    """Componente de velocidad del gas (H I)."""
    return R_data['Vgas']

def V_baryons_sq(R_data, Yd, Yb):
    """Velocidad circular cuadrática de las componentes bariónicas."""
    return V_gas(R_data)**2 + V_disk(R_data, Yd)**2 + V_bulge(R_data, Yb)**2

def V_edr_sq(R_data, V_max_sq, R_scale):
    """Velocidad circular cuadrática del perfil de Materia Oscura (Halo EDR paramétrico)."""
    R = R_data['R']
    R_scale = np.clip(R_scale, 0.1, R_scale) 
    x = R / R_scale
    # Fórmula del perfil de halo exponencial: V_DM^2 = V_max^2 * (1 - (1 + R/R_scale) * exp(-R/R_scale))
    V_sq = V_max_sq * (1.0 - (1.0 + x) * np.exp(-x))
    V_sq[V_sq < 0] = 0
    return V_sq


def V_total(R_data, Yd, Yb, V_max_sq, R_scale):
    """Velocidad circular total cuadrática: V_tot^2 = V_bar^2 + V_DM^2"""
    V_sq = V_baryons_sq(R_data, Yd, Yb) + V_edr_sq(R_data, V_max_sq, R_scale)
    V_sq[V_sq < 0] = 0
    return np.sqrt(V_sq)

def residuals(params, R_data, Vobs, e_Vobs):
    """Vector de residuales para la minimización (chi-cuadrado)."""
    # Parámetros: [Yd, Yb, V_max_sq, R_scale, sigma_extra]
    Yd, Yb, V_max_sq, R_scale, sigma_extra = params 
    
    # Restricciones de parámetros (positividad y límites físicos)
    if Yd < 0.01 or Yb < 0.0 or V_max_sq < 0.01 or R_scale < 0.1 or sigma_extra < 0:
        return np.inf * np.ones_like(Vobs)
        
    Vmodel = V_total(R_data, Yd, Yb, V_max_sq, R_scale)
    # Incertidumbre total en cuadratura
    sigma_tot_sq = e_Vobs**2 + sigma_extra**2
    
    return (Vobs - Vmodel) / np.sqrt(sigma_tot_sq)

# --- FUNCIÓN PRINCIPAL DE AJUSTE ---

def fit_galaxy_curve(df_rc):
    """Ajusta la curva de rotación para una sola galaxia."""
    
    galaxy_name = df_rc['ID'].iloc[0]
    R = df_rc['R'].values
    Vobs = df_rc['Vobs'].values
    e_Vobs = df_rc['e_Vobs'].values
    
    R_data = {
        'Vdisk': df_rc['Vdisk'].values, 'Vbul': df_rc['Vbul'].values, 
        'Vgas': df_rc['Vgas'].values, 'R': R
    }
    
    # Valores Iniciales: [Yd, Yb, V_max_sq, R_scale, sigma_extra]
    Vobs_max = np.nanmax(Vobs)
    p0 = [1.0, 0.01, Vobs_max**2 * 4.0, 10.0, 1.0] 
    
    # Límites de los Parámetros
    bounds = (
        [0.01, 0.0, 0.01, 0.1, 0.0],  # Límite inferior
        [5.0, 5.0, 500000.0, 100.0, 5.0]  # Límite superior
    )
    
    try:
        result = least_squares(
            residuals, 
            p0, 
            args=(R_data, Vobs, e_Vobs), 
            bounds=bounds,
            method='trf',
            max_nfev=2000 
        )
        popt = result.x
        
        # Cálculo de Chi-cuadrado Reducido
        dof = len(Vobs) - len(popt)
        chi2 = np.sum(residuals(popt, R_data, Vobs, e_Vobs)**2)
        chi2_red = chi2 / dof if dof > 0 else np.nan
        
        # Mapeo de V_max_sq -> A y R_scale -> R0
        fit_results = {
            'Galaxy': galaxy_name, 'Yd': popt[0], 'Yb': popt[1], 
            'A': popt[2], 'R0': popt[3], 'sigma_extra': popt[4], 
            'chi2_red': chi2_red
        }
        
        return fit_results

    except Exception as e:
        # Fallo de ajuste
        return {'Galaxy': galaxy_name, 'Yd': np.nan, 'Yb': np.nan, 'A': np.nan, 'R0': np.nan, 'sigma_extra': np.nan, 'chi2_red': np.nan}


# --- CARGA Y PREPARACIÓN DE DATOS ---

def load_sparc_data(filepath, galaxies_to_fit_list):
    """Carga los datos de la Tabla 2 de SPARC (formato de ancho fijo)."""
    try:
        col_names = ['ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
        
        # Leemos hasta SBbul (columna 69-76) para cargar la información completa
        df = pd.read_fwf(
            filepath,
            colspecs=[
                (0, 11), (12, 18), (19, 25), (26, 32), (33, 38), 
                (39, 45), (46, 52), (53, 59), (60, 67), (68, 76)
            ],
            header=None,
            names=col_names,
            skiprows=42, # Saltar encabezado
            dtype={'ID': str} 
        )
        
        df['ID'] = df['ID'].str.strip()
        numeric_cols = ['D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        if galaxies_to_fit_list != 'ALL':
             df = df[df['ID'].isin(galaxies_to_fit_list)]
             
        return df

    except FileNotFoundError:
        print(f"ERROR: Archivo de datos SPARC no encontrado en: {filepath}")
        return None
    except Exception as e:
        print(f"ERROR al parsear {filepath}. Revisar el formato de ancho fijo. Detalle: {e}")
        return None

# --- EJECUCIÓN DEL FLUJO ---

if __name__ == '__main__':
    
    # Cargar todos los nombres de galaxias únicas
    temp_df = load_sparc_data(DATA_FILE, 'temp') 
    if temp_df is None:
        exit()
        
    galaxy_list = temp_df['ID'].unique().tolist()
    df_sparc = load_sparc_data(DATA_FILE, galaxy_list) 
    
    print(f"--- INICIANDO AJUSTE EDR: MUESTRA COMPLETA SPARC (N={len(galaxy_list)}) ---")
    
    if df_sparc is None or df_sparc.empty:
        print("Fallo al cargar los datos. No se puede proceder con el ajuste.")
        exit()
        
    all_results = []
    success_count = 0
    
    for i, galaxy in enumerate(galaxy_list):
        # Imprimir progreso cada 25 galaxias
        if (i + 1) % 25 == 0 or i == 0 or i == len(galaxy_list) - 1:
            print(f"Progreso: Procesando galaxia {i+1}/{len(galaxy_list)}: {galaxy}...")

        df_rc = df_sparc[df_sparc['ID'] == galaxy].copy()
        
        # Pre-procesamiento: rellenar Bulge y quitar NaNs cruciales
        df_rc['Vbul'] = df_rc['Vbul'].fillna(0.0) 
        df_rc = df_rc.dropna(subset=['R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk'])
        df_rc = df_rc[df_rc['R'] > 0]
        
        if df_rc.empty:
            all_results.append({'Galaxy': galaxy, 'Yd': np.nan, 'Yb': np.nan, 'A': np.nan, 'R0': np.nan, 'sigma_extra': np.nan, 'chi2_red': np.nan})
            continue
            
        fit_res = fit_galaxy_curve(df_rc)
        all_results.append(fit_res) 
        
        if not np.isnan(fit_res['chi2_red']):
            success_count += 1
        
    print(f"\n--- AJUSTE DE CURVAS FINALIZADO ({success_count}/{len(galaxy_list)} galaxias ajustadas) ---")

    # Guardar todos los resultados
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_CSV, index=False)
    
    print(f"RESULTADOS GUARDADOS EN: {RESULTS_CSV}")
    print(f"\n¡Ahora puedes ejecutar 'run_mdar_validation_175.py' para la validación MDAR!")
