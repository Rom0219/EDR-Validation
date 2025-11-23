import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # Ignorar warnings de log(0) temporalmente

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

def V_disk(R_data, Yd):
    """
    Componente de velocidad del disco, escalada por la relación Masa/Luminosidad (Yd).
    """
    return np.sqrt(Yd) * R_data['Vdisk']

def V_bulge(R_data, Yb):
    """
    Componente de velocidad del Bulge, escalada por la relación Masa/Luminosidad (Yb).
    """
    return np.sqrt(Yb) * R_data['Vbul']

def V_gas(R_data):
    """Componente de velocidad del gas (H I). No se escala por M/L."""
    return R_data['Vgas']

def V_baryons_sq(R_data, Yd, Yb):
    """Velocidad circular cuadrática de las componentes bariónicas."""
    return V_gas(R_data)**2 + V_disk(R_data, Yd)**2 + V_bulge(R_data, Yb)**2

def V_edr_sq(R_data, V_max_sq, R_scale):
    """
    Velocidad circular cuadrática del perfil de Materia Oscura (Halo).
    Usando el perfil exponencial simplificado: V_DM^2 = V_max^2 * (1 - (1 + R/R_scale) * exp(-R/R_scale)).
    Este es un modelo empírico de halo comúnmente usado en lugar del NFW completo.
    """
    R = R_data['R']
    
    # Asegurar R_scale no sea cercano a cero para evitar divisiones
    R_scale = np.clip(R_scale, 0.1, R_scale) 
    
    x = R / R_scale # Relación R/R_scale
    
    # Fórmula del perfil de halo exponencial:
    V_sq = V_max_sq * (1.0 - (1.0 + x) * np.exp(-x))
    
    # Evitar V^2 negativo
    V_sq[V_sq < 0] = 0
    return V_sq


def V_total(R_data, Yd, Yb, V_max_sq, R_scale):
    """Velocidad circular total cuadrática: V_tot^2 = V_bar^2 + V_DM^2"""
    # Nota: Todas las componentes están en km/s al cuadrado.
    V_sq = V_baryons_sq(R_data, Yd, Yb) + V_edr_sq(R_data, V_max_sq, R_scale)
    # Evitar la raíz cuadrada de números negativos (aunque no debería ocurrir)
    V_sq[V_sq < 0] = 0
    return np.sqrt(V_sq)

def residuals(params, R_data, Vobs, e_Vobs):
    """Vector de residuales para la minimización (chi-cuadrado)."""
    # Los parámetros del fit ahora son: [Yd, Yb, V_max_sq, R_scale, sigma_extra]
    Yd, Yb, V_max_sq, R_scale, sigma_extra = params 
    
    # Restricciones de parámetros (los parámetros físicos deben ser positivos)
    # Yd y Yb deben ser > 0.01. V_max_sq y R_scale deben ser > 0
    if Yd < 0.01 or Yb < 0.0 or V_max_sq < 0.01 or R_scale < 0.1 or sigma_extra < 0:
        # Se devuelve infinito si los parámetros están fuera de los límites de las restricciones internas
        return np.inf * np.ones_like(Vobs)
        
    Vmodel = V_total(R_data, Yd, Yb, V_max_sq, R_scale)
    # Suma en cuadratura de la incertidumbre (sigma_extra se añade en cuadratura)
    sigma_tot_sq = e_Vobs**2 + sigma_extra**2
    
    # Residuales normalizados por la incertidumbre total
    return (Vobs - Vmodel) / np.sqrt(sigma_tot_sq)

# --- FUNCIÓN PRINCIPAL DE AJUSTE ---

def fit_galaxy_curve(df_rc):
    """Ajusta la curva de rotación para una sola galaxia."""
    
    galaxy_name = df_rc['ID'].iloc[0]
    
    # 1. Preparación de Datos
    R = df_rc['R'].values
    Vobs = df_rc['Vobs'].values
    e_Vobs = df_rc['e_Vobs'].values
    
    # Diccionario de datos para componentes bariónicas (pasado a las funciones V_*)
    R_data = {
        'Vdisk': df_rc['Vdisk'].values,
        'Vbul': df_rc['Vbul'].values, # CORRECCIÓN CLAVE: Ahora usa Vbul
        'Vgas': df_rc['Vgas'].values,
        'R': R
    }
    
    # 2. Valores Iniciales (Initial Guess)
    # [Yd, Yb, V_max_sq, R_scale, sigma_extra]
    # Usamos Vmax_sq = Vobs_max^2 * 4 como estimación de un halo grande
    Vobs_max = np.nanmax(Vobs)
    p0 = [1.0, 0.01, Vobs_max**2 * 4.0, 10.0, 1.0] 
    
    # 3. Límites de los Parámetros (Bounds)
    # [Yd, Yb, V_max_sq, R_scale, sigma_extra]
    bounds = (
        [0.01, 0.0, 0.01, 0.1, 0.0],  # Límite inferior
        [5.0, 5.0, 500000.0, 100.0, 5.0]  # Límite superior: Aumentado V_max_sq de 200000.0 a 500000.0
    )
    
    # 4. Ajuste por Mínimos Cuadrados
    try:
        result = least_squares(
            residuals, 
            p0, 
            args=(R_data, Vobs, e_Vobs), 
            bounds=bounds,
            method='trf',
            max_nfev=2000 # Limitar iteraciones para evitar bucles infinitos en fits difíciles
        )
        popt = result.x
        
        # 5. Cálculo de Chi-cuadrado Reducido (usando popt para calcular residuos finales)
        dof = len(Vobs) - len(popt)
        chi2 = np.sum(residuals(popt, R_data, Vobs, e_Vobs)**2)
        chi2_red = chi2 / dof if dof > 0 else np.nan
        
        # 6. Preparar resultados (Usamos A y R0 en el CSV para mantener la estructura de la MDAR)
        fit_results = {
            'Galaxy': galaxy_name, 
            'Yd': popt[0], 'Yb': popt[1], 
            'A': popt[2], 'R0': popt[3], # Mapeo de V_max_sq -> A y R_scale -> R0
            'sigma_extra': popt[4], 'chi2_red': chi2_red
        }
        
        # 7. Graficar (para validación visual)
        plot_fit(df_rc, R_data, V_total, popt, chi2_red)
        
        # Uso de raw f-string (rf"...") para evitar SyntaxWarning con \chi
        print(rf"INFO: {galaxy_name} - Ajuste exitoso. $\chi^2_\nu={chi2_red:.2f}$, Yd={popt[0]:.2f}, Yb={popt[1]:.2f}")
        return fit_results

    except Exception as e:
        print(f"ERROR: Fallo de ajuste para {galaxy_name}. Detalle: {e}")
        return {'Galaxy': galaxy_name, 'Yd': np.nan, 'Yb': np.nan, 'A': np.nan, 'R0': np.nan, 'sigma_extra': np.nan, 'chi2_red': np.nan}


def plot_fit(df_rc, R_data_obs, V_total_func, popt, chi2_red):
    """Genera y guarda el gráfico del ajuste de la curva de rotación."""
    
    R_obs = df_rc['R'].values
    Vobs = df_rc['Vobs'].values
    e_Vobs = df_rc['e_Vobs'].values
    galaxy_name = df_rc['ID'].iloc[0]
    
    Yd, Yb, V_max_sq, R_scale, _ = popt
    
    # Crear un rango de radio suave para la curva del modelo
    R_model = np.linspace(R_obs.min(), R_obs.max() * 1.1, 100)
    
    # Interpolación para obtener los V's bariónicos en el rango suave del modelo
    R_data_model = {
        'Vdisk': np.interp(R_model, R_obs, R_data_obs['Vdisk']),
        'Vbul': np.interp(R_model, R_obs, R_data_obs['Vbul']),
        'Vgas': np.interp(R_model, R_obs, R_data_obs['Vgas']),
        'R': R_model
    }

    # Calcular las componentes del modelo
    V_model_total = V_total_func(R_data_model, Yd, Yb, V_max_sq, R_scale)
    V_model_bar = np.sqrt(V_baryons_sq(R_data_model, Yd, Yb))
    V_model_dm = np.sqrt(V_edr_sq(R_data_model, V_max_sq, R_scale))
    
    plt.figure(figsize=(7, 6))
    
    # 1. Componentes Bariónicas (Modelo)
    # Uso de raw f-string (rf"...") para evitar SyntaxWarning con \gamma
    plt.plot(R_model, V_model_bar, linestyle=':', color='gray', 
             label=rf'Bariones Total ($\gamma_d={Yd:.2f}, \gamma_b={Yb:.2f}$)')
    
    # 2. Componente DM (EDR) (Modelo)
    plt.plot(R_model, V_model_dm, linestyle='--', color='purple', label='Halo DM (EDR Paramétrico)')

    # 3. Velocidad Total (Modelo)
    # Uso de raw f-string (rf"...") para evitar SyntaxWarning con \chi
    plt.plot(R_model, V_model_total, linestyle='-', color='blue', linewidth=2, 
             label=rf'Modelo EDR Total ($\chi^2_\nu={chi2_red:.2f}$)')
    
    # 4. Observaciones
    plt.errorbar(R_obs, Vobs, yerr=e_Vobs, fmt='o', color='black', 
             markersize=4, capsize=3, label='Velocidad Observada ($V_{obs}$)')

    plt.xlabel('Radio (kpc)')
    plt.ylabel('Velocidad Circular (km/s)')
    plt.title(f'Ajuste Curva de Rotación: {galaxy_name}')
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
        
        # Revisión para evitar errores de caracteres no imprimibles (U+00A0)
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
            skiprows=42, # Saltar encabezado y descripción
            dtype={'ID': str} 
        )
        
        # Limpieza y conversión a numérico:
        df['ID'] = df['ID'].str.strip()
        # Forzar las columnas de datos a numérico, convirtiendo errores (como '---') a NaN
        numeric_cols = ['D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
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
        # Vbul no siempre es crucial si su valor es NaN (la mayoría de veces significa Vbul=0)
        # Por simplicidad, si Vbul es NaN, lo rellenamos con 0, si otras columnas son NaN, descartamos
        df_rc['Vbul'] = df_rc['Vbul'].fillna(0.0) 
        df_rc = df_rc.dropna(subset=['R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk'])
        df_rc = df_rc[df_rc['R'] > 0]
        
        if df_rc.empty:
            print(f"Advertencia: Datos insuficientes para {galaxy}. Saltando.")
            
            # Aseguramos un registro NaN para esta galaxia
            all_results.append({'Galaxy': galaxy, 'Yd': np.nan, 'Yb': np.nan, 'A': np.nan, 'R0': np.nan, 'sigma_extra': np.nan, 'chi2_red': np.nan})
            continue
            
        fit_res = fit_galaxy_curve(df_rc)
        
        # Agregar el resultado, fit_galaxy_curve ya incluye el nombre de la galaxia
        all_results.append(fit_res) 
        
    print("\n--- AJUSTE DE CURVAS FINALIZADO ---")

    # Guardar todos los resultados en el archivo de entrada requerido para MDAR
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_CSV, index=False)
    
    print(f"RESULTADOS GUARDADOS EN: {RESULTS_CSV}")
    print("\n¡Ahora puedes ejecutar 'run_mdar_validation.py' para el test final!")
