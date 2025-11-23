import pandas as pd
from pathlib import Path
import os
# Se eliminó 'requests' ya que no se requiere descarga por internet.

# --- CONFIGURACIÓN DE RUTAS ROBUSTAS ---
DATA_DIR_SPARC = Path(__file__).resolve().parent 
RESULTS_DIR = DATA_DIR_SPARC / "btfr_analysis_data"
RESULTS_CSV = RESULTS_DIR / 'sparc_results_175.csv'

# Ruta al archivo de datos sin procesar que el usuario debe tener localmente
LOCAL_SPARC_FILE = DATA_DIR_SPARC / "SPARC_data.csv"

def prepare_sparc_data_from_local(input_path, output_path):
    """
    Carga el dataset SPARC desde un archivo local ('SPARC_data.csv'), filtra y 
    guarda el archivo CSV de resultados listos para la validación BTFR.
    """
    print("=" * 50)
    print("INICIANDO PREPARACIÓN DE DATOS SPARC (MODO LOCAL)")
    print("-" * 50)
    
    # 1. Asegurar la existencia del directorio de salida
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de salida verificado/creado: {RESULTS_DIR}")

    # 2. Cargar datos desde el archivo local
    if not input_path.exists():
        print("-" * 50)
        print(f"ERROR CRÍTICO: Archivo de datos RAW no encontrado en la ruta esperada.")
        print(f"RUTA ESPERADA: {input_path}")
        print("Asegúrate de que el archivo 'SPARC_data.csv' esté subido y ubicado en EDR/data/sparc/")
        print("-" * 50)
        return

    try:
        print(f"-> Archivo RAW encontrado. Cargando datos desde: {input_path}")
        # Asumiendo que el archivo de datos RAW se llama 'SPARC_data.csv'
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"ERROR AL PROCESAR EL CSV local: {e}")
        return

    # 3. Filtrado y Limpieza de Datos
    
    # Columnas relevantes para BTFR
    required_cols = ['ID', 'Vobs', 'R', 'SBdisk', 'SBbul']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: El archivo SPARC local carece de las columnas clave: {missing_cols}")
        print("El análisis BTFR no puede continuar.")
        return

    df_filtered = df[required_cols].copy()
    
    # Filtrar datos no válidos (Ejemplo: Vobs > 0 y Radios válidos)
    df_filtered = df_filtered[(df_filtered['Vobs'] > 0) & (df_filtered['R'] > 0)].copy()

    # 4. Guardar el archivo de resultados limpio
    if not df_filtered.empty:
        df_filtered.to_csv(output_path, index=False)
        print("-" * 50)
        print(f"¡ÉXITO! Datos procesados y guardados en: {output_path}")
        print(f"Filas guardadas: {len(df_filtered)}")
        print("Listo para la validación BTFR.")
    else:
        print("-" * 50)
        print("ADVERTENCIA: El DataFrame filtrado está vacío después del procesamiento. No se guardó ningún archivo.")
    
    print("=" * 50)

if __name__ == "__main__":
    prepare_sparc_data_from_local(LOCAL_SPARC_FILE, RESULTS_CSV)
