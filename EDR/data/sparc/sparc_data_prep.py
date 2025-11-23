import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
DATA_FILE_NAME = 'SPARC_Lelli2016_Table2.txt'

# Rutas Robustas: Path(__file__).parent apunta al directorio EDR/data/sparc/
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / DATA_FILE_NAME

# Directorio de salida.
OUTPUT_DIR = SCRIPT_DIR / 'btfr_analysis_data'
OUTPUT_FILE = OUTPUT_DIR / 'sparc_results_175.csv'

# Definición de Anchos de Columna y Nombres (Crucial para leer el archivo .txt)
colspecs = [
    (0, 11),  # ID
    (12, 18), # D (Distancia)
    (19, 25), # R (Radio)
    (26, 32), # Vobs (Velocidad Observada)
    (33, 38), # e_Vobs (Error de Vobs)
    (39, 45), # Vgas (Gas)
    (46, 52), # Vdisk (Disco)
    (53, 59), # Vbul (Bulbo)
    (60, 67), # SBdisk (Brillo Superficial Disco)
    (68, 76), # SBbul (Bulge surface brightness)
]

names = [
    'ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'
]

def prepare_sparc_data():
    """Carga los datos SPARC, limpia y calcula Vbar."""
    print(f"1. Iniciando la carga de datos.")
    print(f"   -> Buscando archivo de datos en la ruta: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Usamos read_fwf (fixed-width file) ya que es el formato original de SPARC Table 2
        df = pd.read_fwf(
            INPUT_FILE,
            colspecs=colspecs,
            names=names,
            skiprows=8, # Omitir las primeras 8 líneas de metadatos del TXT
            engine='python' 
        )
    except FileNotFoundError:
        print(f"¡ERROR! No se encontró el archivo de entrada en la ruta esperada: {INPUT_FILE}")
        print("ACCIÓN REQUERIDA: Si el archivo está ahí, el error es de permisos o ruta.")
        return
    except Exception as e:
        print(f"Error al leer el archivo (Revisa el formato): {e}")
        return
    
    # --- Limpieza y Cálculo ---
    # Reemplazar valores nulos de texto (-) por NaN
    df = df.replace(to_replace='-', value=np.nan) 
    
    # Convertir a numérico (los valores perdidos ahora serán NaN)
    numeric_cols = ['D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Asumir Bulge nulo (Vbul=0) si el valor falta (NaN)
    df['Vbul'] = df['Vbul'].fillna(0.0)

    # Filtrar filas con valores NaN en las columnas críticas (Vobs, Vgas, Vdisk, D)
    df_clean = df.dropna(subset=['Vobs', 'Vgas', 'Vdisk', 'D']).copy()

    # Calcular la Velocidad de Componentes Bariónicos (Vbar)
    df_clean['Vbar_sq'] = df_clean['Vgas']**2 + df_clean['Vdisk']**2 + df_clean['Vbul']**2
    df_clean['Vbar'] = np.sqrt(df_clean['Vbar_sq'])
    
    df_final = df_clean[df_clean['Vbar'] > 0].reset_index(drop=True)
    
    # Guardar los datos limpios y calculados
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("-" * 50)
    print(f"2. Proceso completado exitosamente.")
    print(f"   -> Datos limpios y calculados guardados en: {OUTPUT_FILE}")
    print(f"   -> Número de puntos procesados: {len(df_final)}")
    print("-" * 50)

if __name__ == "__main__":
    prepare_sparc_data()
