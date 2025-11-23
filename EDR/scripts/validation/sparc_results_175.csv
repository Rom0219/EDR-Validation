import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import least_squares
import warnings
# Ignorar warnings de log(0) temporalmente
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# --- CONSTANTES ---
# Aceleración característica de MOND (kg/s^2 * 1e11)
A0 = 1.2e-10 
G = 6.674e-11 # Constante de gravitación universal (m^3 kg^-1 s^-2)
PC_TO_M = 3.08567758e16 # 1 parsec en metros
KPC_TO_M = PC_TO_M * 1000 # 1 kpc en metros
KM_PER_S_TO_M_PER_S = 1000 # 1 km/s en m/s

# --- CONFIGURACIÓN DE RUTAS ACTUALIZADA ---

# Directorio donde se encuentra el CSV de resultados de ajuste (Ruta solicitada: EDR/data/sparc/datafiles175)
CSV_DIR = Path("EDR/data/sparc/datafiles175")
# Archivo de resultados de ajuste EDR (sparc_results_175.csv)
RESULTS_CSV = CSV_DIR / "sparc_results_175.csv"
# Archivo de datos SPARC (Tabla 2)
DATA_FILE = Path("EDR/data/sparc/SPARC_Lelli2016_Table2.txt")
# Directorio para guardar los gráficos de validación MDAR (Gráficos globales)
OUT_DIR = Path("EDR/results/validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- FUNCIÓN DE MODELO EDR (Duplicada para auto-contención) ---

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

# --- CÁLCULO DE ACELERACIONES ---

def calculate_accelerations(df_rc, df_fit_params):
    """Calcula las aceleraciones observadas y bariónicas para una galaxia."""
    
    R = df_rc['R'].values
    Vobs = df_rc['Vobs'].values
    e_Vobs = df_rc['e_Vobs'].values
    
    # Parámetros ajustados (A = V_max_sq, R0 = R_scale)
    Yd = df_fit_params['Yd'].iloc[0]
    Yb = df_fit_params['Yb'].iloc[0]
    A_sq = df_fit_params['A'].iloc[0]
    R0 = df_fit_params['R0'].iloc[0]
    sigma_extra = df_fit_params['sigma_extra'].iloc[0]
    
    R_data = {
        'Vdisk': df_rc['Vdisk'].values, 'Vbul': df_rc['Vbul'].values, 
        'Vgas': df_rc['Vgas'].values, 'R': R
    }
    
    # 1. Aceleración Observada (a_obs)
    # a_obs = V_obs^2 / R
    a_obs = (Vobs**2) / R
    
    # Incertidumbre en a_obs: e_a_obs = 2 * V_obs * e_V_obs / R
    e_a_obs = 2 * Vobs * np.sqrt(e_Vobs**2 + sigma_extra**2) / R

    # 2. Aceleración Bariónica (a_bar)
    # a_bar = V_bar^2 / R
    V_bar_sq = V_baryons_sq(R_data, Yd, Yb)
    a_bar = V_bar_sq / R
    
    # 3. Aceleración DM (a_dm)
    # a_dm = V_DM^2 / R
    V_dm_sq = V_edr_sq(R_data, A_sq, R0)
    a_dm = V_dm_sq / R
    
    # 4. Aceleración Total del Modelo (a_mod)
    # a_mod = (V_bar^2 + V_DM^2) / R = a_bar + a_dm
    a_mod = a_bar + a_dm

    # Generar la tabla de resultados para la galaxia
    df_accel = pd.DataFrame({
        'Galaxy': df_rc['ID'], 'R': R, 
        'a_obs': a_obs, 'e_a_obs': e_a_obs,
        'a_bar': a_bar, 'a_dm': a_dm, 'a_mod': a_mod
    })
    
    return df_accel

# --- FUNCIÓN DE PLOTEO MDAR ---

def plot_mdar(df_mdar_full):
    """Genera el gráfico de la Relación de Aceleración de Masa (MDAR)."""
    
    plt.figure(figsize=(10, 8))
    
    # Filtros para datos válidos
    df_mdar = df_mdar_full.dropna(subset=['a_bar', 'a_obs', 'e_a_obs'])
    df_mdar = df_mdar[(df_mdar['a_bar'] > 0) & (df_mdar['a_obs'] > 0)]
    
    # Curva de MOND (para comparación)
    a_mond = lambda ab: ab * (1.0 - np.exp(-np.sqrt(ab / A0)))**(-1)
    a_bar_grid = np.logspace(np.log10(df_mdar['a_bar'].min()), np.log10(df_mdar['a_bar'].max()), 100)
    
    # --- Gráfico principal: a_obs vs a_bar ---
    
    # Puntos de datos (a_obs vs a_bar)
    plt.errorbar(df_mdar['a_bar'], df_mdar['a_obs'], yerr=df_mdar['e_a_obs'], 
                 fmt='o', color='black', alpha=0.3, label='Datos SPARC (a_obs vs a_bar)')
    
    # Modelo EDR (a_mod vs a_bar)
    # a_mod = a_bar + a_dm. Esto se plotea como la predicción del modelo EDR.
    plt.scatter(df_mdar['a_bar'], df_mdar['a_mod'], 
                s=10, color='red', label='Modelo EDR (a_mod vs a_bar)', zorder=5)

    # Curvas de referencia
    plt.plot(a_bar_grid, a_bar_grid, 'k--', label='$a_{obs} = a_{bar}$ (Newtoniano)', alpha=0.7)
    # MOND - Curva teórica
    plt.plot(a_bar_grid, a_mond(a_bar_grid), 'g-', linewidth=2, label='Predicción MOND', zorder=4)
    
    # Línea A0 de MOND
    plt.axvline(A0, color='gray', linestyle=':', label='$a_0$ MOND', alpha=0.7)
    plt.axhline(A0, color='gray', linestyle=':', alpha=0.7)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$a_{bar}$ (km$^2$ s$^{-2}$ kpc$^{-1}$)')
    plt.ylabel('$a_{obs}$ (km$^2$ s$^{-2}$ kpc$^{-1}$)')
    plt.title('Relación de Aceleración de Masa (MDAR): Modelo EDR vs SPARC (N=175)')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    plot_path = OUT_DIR / "MDAR_plot_EDR_175_sample.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"GRÁFICO MDAR GUARDADO EN: {plot_path}")

# --- FUNCIÓN DE PLOTEO HISTOGRAMA DE RESIDUALES ---

def plot_residuals(df_mdar_full):
    """Genera el histograma de los residuales normalizados."""
    
    df_mdar = df_mdar_full.dropna(subset=['a_obs', 'a_mod', 'e_a_obs'])
    
    # Residuales Normalizados
    residuales = (df_mdar['a_obs'] - df_mdar['a_mod']) / df_mdar['e_a_obs']
    
    if residuales.empty:
        print("AVISO: No hay suficientes datos para generar el histograma de residuales.")
        return

    # Ajuste de la distribución normal a los residuales
    mu, sigma = norm.fit(residuales)

    plt.figure(figsize=(9, 6))
    
    # Histograma de los residuales
    n, bins, patches = plt.hist(residuales, bins=50, density=True, alpha=0.6, color='b',
                                label='Residuales EDR Normalizados')
    
    # Curva de la distribución normal ajustada
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2, label=f'Ajuste Normal ($\mu$={mu:.3f}, $\sigma$={sigma:.3f})')
    
    # Curva de la distribución normal estándar (para comparación)
    p_std = norm.pdf(x, 0, 1)
    plt.plot(x, p_std, 'r--', linewidth=1, label='Normal Estándar ($\mu$=0, $\sigma$=1)')
    
    plt.axvline(0, color='gray', linestyle=':', alpha=0.7)
    
    plt.xlabel('Residual Normalizado: $(a_{obs} - a_{mod}) / \sigma_{a_{obs}}$')
    plt.ylabel('Densidad de Probabilidad')
    plt.title('Histograma de Residuales Normalizados: Modelo EDR (N=175)')
    plt.legend(loc='upper right')
    plt.grid(True, axis='y', ls='--', alpha=0.4)
    
    plot_path = OUT_DIR / "histogram_residuals_EDR_175_sample.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"HISTOGRAMA DE RESIDUALES GUARDADO EN: {plot_path}")


# --- CARGA DE DATOS ---

def load_sparc_data(filepath):
    """Carga los datos de la Tabla 2 de SPARC y devuelve el DataFrame completo."""
    try:
        col_names = ['ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
        
        df = pd.read_fwf(
            filepath,
            colspecs=[
                (0, 11), (12, 18), (19, 25), (26, 32), (33, 38), 
                (39, 45), (46, 52), (53, 59), (60, 67), (68, 76)
            ],
            header=None,
            names=col_names,
            skiprows=42, 
            dtype={'ID': str} 
        )
        
        df['ID'] = df['ID'].str.strip()
        numeric_cols = ['D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Pre-procesamiento para MDAR: llenar Bulge y quitar NaNs cruciales
        df['Vbul'] = df['Vbul'].fillna(0.0) 
        df = df.dropna(subset=['R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk'])
        df = df[df['R'] > 0]
            
        return df

    except FileNotFoundError:
        print(f"ERROR: Archivo de datos SPARC no encontrado en: {filepath}")
        return None
    except Exception as e:
        print(f"ERROR al parsear {filepath}. Revisar el formato de ancho fijo. Detalle: {e}")
        return None


# --- EJECUCIÓN DEL FLUJO ---

if __name__ == '__main__':
    
    df_sparc_full = load_sparc_data(DATA_FILE) 
    
    if df_sparc_full is None or df_sparc_full.empty:
        print("Fallo al cargar los datos de la curva de rotación.")
        exit()
        
    try:
        # Aquí se usa la ruta confirmada: EDR/data/sparc/datafiles175/sparc_results_175.csv
        df_fit_results = pd.read_csv(RESULTS_CSV)
        print(f"Leyendo resultados de ajuste desde: {RESULTS_CSV}")
    except FileNotFoundError:
        print(f"ERROR: Archivo de resultados de ajuste no encontrado en la ruta esperada: {RESULTS_CSV}")
        print("Asegúrate de ejecutar 'run_fit_and_plot_all_175.py' primero.")
        exit()
        
    # Filtrar solo las galaxias que se ajustaron correctamente
    df_fit_results = df_fit_results.dropna(subset=['chi2_red'])
    galaxy_list = df_fit_results['Galaxy'].unique().tolist()
    
    if not galaxy_list:
        print("AVISO: El archivo de resultados de ajuste no contiene galaxias ajustadas exitosamente.")
        exit()

    print(f"--- INICIANDO VALIDACIÓN MDAR PARA {len(galaxy_list)} GALAXIAS ---")
        
    all_accel_data = []
    
    for i, galaxy in enumerate(galaxy_list):
        if (i + 1) % 50 == 0:
            print(f"Progreso: Calculando aceleraciones para galaxia {i+1}/{len(galaxy_list)}: {galaxy}...")

        # Datos de la curva de rotación y parámetros de ajuste
        df_rc = df_sparc_full[df_sparc_full['ID'] == galaxy].copy()
        df_fit_params = df_fit_results[df_fit_results['Galaxy'] == galaxy].copy()
        
        if df_rc.empty or df_fit_params.empty:
            continue
        
        # Calcular aceleraciones y apilar resultados
        df_accel = calculate_accelerations(df_rc, df_fit_params)
        all_accel_data.append(df_accel)
        
    if not all_accel_data:
        print("ERROR: No se pudo generar ningún dato de aceleración para el ploteo MDAR.")
        exit()
        
    df_mdar_full = pd.concat(all_accel_data, ignore_index=True)

    # Ejecutar el ploteo
    print("\nGenerando gráfico MDAR...")
    plot_mdar(df_mdar_full)
    
    print("Generando histograma de residuales...")
    plot_residuals(df_mdar_full)
    
    print("\n--- VALIDACIÓN MDAR EDR FINALIZADA ---")
