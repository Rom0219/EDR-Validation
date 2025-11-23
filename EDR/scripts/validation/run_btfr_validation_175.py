import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
import os

# --- CONFIGURACIÓN DE RUTAS ABSOLUTAS Y ROBUSTAS ---

# La data se generó en EDR/data/sparc/btfr_analysis_data/. 
# Usamos una ruta fija para asegurar que el script lo encuentre sin importar donde se ejecute.
PROJECT_ROOT = Path(__file__).resolve().parents[2] # Navega al directorio raíz del proyecto: /workspaces/EDR-Validation
DATA_DIR = PROJECT_ROOT / "EDR" / "data" / "sparc" / "btfr_analysis_data"
RESULTS_CSV = DATA_DIR / 'sparc_results_175.csv'

# El gráfico se guardará en la misma carpeta que el script de validación.
OUTPUT_DIR = Path(__file__).resolve().parent

# --- FUNCIÓN PRINCIPAL DE ANÁLISIS BTFR ---

def run_btfr_analysis(input_path, output_dir):
    """
    Carga los datos limpios y realiza la regresión de la Relación de Tully-Fisher Bariónica (BTFR).
    """
    print(f"1. Iniciando análisis BTFR.")
    print(f"   -> Buscando archivo de datos en la ruta: {input_path}")
    
    try:
        # Se asegura de que el archivo existe y no está vacío.
        fit_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print("-" * 50)
        print(f"ERROR: Archivo de entrada no encontrado en '{input_path}'.")
        print("Asegúrate de ejecutar 'sparc_data_prep.py' primero y verificar la ruta.")
        print("-" * 50)
        return
    except pd.errors.EmptyDataError:
        print("-" * 50)
        print("ERROR: El archivo CSV fue encontrado, pero está vacío o malformado.")
        print("Por favor, vuelve a ejecutar 'sparc_data_prep.py' para regenerar los datos.")
        print("-" * 50)
        return

    # 1. Preparación de Datos: Obtener un solo punto (V_final, M_bar) por galaxia
    df_grouped = fit_df.groupby('ID').agg(
        V_final=('Vobs', 'max'), 
        SB_disk_avg=('SBdisk', 'mean'),
        SB_bul_avg=('SBbul', 'mean')
    ).reset_index()

    # 2. Asignación de Masa Bariónica (M_bar) - PROXY DE LUMINOSIDAD
    # Nota: Esta es una simplificación; la masa real requiere calibraciones de color.
    # Usamos una función logarítmica simple de la SB promedio como proxy.
    df_grouped['Log_Mbar'] = (
        np.log10(df_grouped['SB_disk_avg'] + df_grouped['SB_bul_avg'].fillna(0.0) + 1e-6) 
        * 1.0 + 9.5 
    )

    # 3. Regresión Lineal (BTFR)
    df_regression = df_grouped.dropna(subset=['V_final', 'Log_Mbar'])
    df_regression = df_regression[df_regression['V_final'] > 0]
    df_regression['Log_Vfinal'] = np.log10(df_regression['V_final'])
    
    x = df_regression['Log_Vfinal']
    y = df_regression['Log_Mbar']

    # Se realiza el ajuste lineal (y = a*x + b)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value**2

    print("-" * 50)
    print("RESULTADOS DEL AJUSTE BTFR (log-log):")
    print(f"Ecuación: log10(M_bar/M_sol) = ({slope:.3f}) * log10(V_final) + ({intercept:.3f})")
    print(f"R^2 (Coeficiente de Determinación): {r_squared:.4f}")
    print(f"Número de galaxias utilizadas en el ajuste: {len(df_regression)}")
    print("-" * 50)

    # 4. Graficar la Relación de Tully-Fisher Bariónica
    x_model = np.linspace(x.min() * 0.95, x.max() * 1.05, 100)
    y_model = slope * x_model + intercept

    plt.figure(figsize=(10, 7))
    plt.style.use('default')

    plt.scatter(x, y, color='#1f77b4', alpha=0.7, edgecolors='w', s=50, label='Galaxias SPARC (Punto Final)')
    plt.plot(x_model, y_model, color='#d62728', linewidth=3, linestyle='--',
             label=f'Ajuste Lineal\n$a={slope:.3f}, b={intercept:.3f}$\n$R^2={r_squared:.4f}$')

    plt.xlabel('$\\log_{10}(V_{\\text{final}} \\text{ / } \\text{km s}^{-1})$', fontsize=14)
    plt.ylabel('$\\log_{10}(M_{\\text{bar}} \\text{ / } M_{\\odot})$', fontsize=14)
    plt.title('Validación de la Relación de Tully-Fisher Bariónica (BTFR)', fontsize=16)
    
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Guardar gráfico
    plot_path = output_dir / "BTFR_Validation_Plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"2. Gráfico BTFR guardado en: {plot_path}")

if __name__ == "__main__":
    print("=" * 45)
    print("   VALIDACIÓN: RELACIÓN TULLY-FISHER BARIÓNICA (BTFR)")
    print("=" * 45)
    run_btfr_analysis(RESULTS_CSV, OUTPUT_DIR)
