import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# Nombre del archivo de datos original
INPUT_FILE_NAME = 'SPARC_Lelli2016_Table2.txt'

# Directorio de resultados. Se crea si no existe.
# Este es el directorio de resultados para la limpieza y el análisis BTFR.
RESULTS_DIR = Path("btfr_analysis_data")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Nombre del archivo de salida limpio que contendrá Vbar (será el INPUT para el BTFR)
OUTPUT_FILE = RESULTS_DIR / 'sparc_results_175.csv'

# --- Definición de Anchos de Columna y Nombres (Basado en Table 2) ---
# Usamos la definición exacta de la Tabla 2 de SPARC
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
    (68, 76), # SBbul (Brillo Superficial Bulbo)
]

names = [
    'ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'
]

def load_and_clean_sparc_data(input_file, output_path):
    """
    Carga los datos de ancho fijo de SPARC, maneja errores, calcula Vbar
    y guarda el resultado en formato CSV.
    """
    try:
        # 1. Cargar los datos usando read_fwf (read fixed-width file)
        df = pd.read_fwf(
            input_file,
            colspecs=colspecs,
            names=names,
            skiprows=8, # Omitir las primeras 8 líneas de metadatos del TXT
            engine='python' 
        )

        print(f"Datos cargados exitosamente desde: {input_file}")

        # --- Limpieza de Datos ---
        # 2. Reemplazar valores de texto (como '-----') por NaN (Not a Number)
        df = df.replace(to_replace='-', value=np.nan)
        df = df.replace(to_replace='------', value=np.nan)
        df = df.replace(to_replace='-------', value=np.nan)

        # 3. Convertir columnas relevantes a tipo numérico (float)
        numeric_cols = ['D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. Asumir Bulge nulo (Vbul=0) si el valor falta (NaN)
        df['Vbul'] = df['Vbul'].fillna(0.0)

        # 5. Filtrar filas con valores NaN en las columnas críticas (Vobs, Vgas, Vdisk)
        df_clean = df.dropna(subset=['Vobs', 'Vgas', 'Vdisk', 'D']).copy()

        # --- Cálculo de la Velocidad Bariónica Clave (Vbar) ---
        # 6. Calcular la Velocidad de Componentes Bariónicos (Vbar)
        # La relación es Vbar^2 = Vgas^2 + Vdisk^2 + Vbul^2
        df_clean['Vbar_sq'] = df_clean['Vgas']**2 + df_clean['Vdisk']**2 + df_clean['Vbul']**2
        df_clean['Vbar'] = np.sqrt(df_clean['Vbar_sq'])

        # 7. Filtrar puntos con velocidad bariónica válida
        df_final = df_clean[df_clean['Vbar'] > 0].reset_index(drop=True)
        
        # 8. Guardar el resultado
        df_final.to_csv(output_path, index=False)
        
        print("-" * 50)
        print("Proceso de limpieza completado.")
        print(f"Datos de las curvas de rotación listos para el análisis BTFR.")
        print(f"Número total de puntos de datos limpios: {len(df_final)}")
        print(f"Archivo de datos limpio (INPUT para BTFR) guardado en: {output_path}")
        print("-" * 50)

    except FileNotFoundError:
        print(f"ERROR: Archivo de entrada no encontrado en '{input_file}'.")
        print(f"Asegúrate de que el archivo '{INPUT_FILE_NAME}' esté disponible para el script.")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante el procesamiento: {e}")

if __name__ == "__main__":
    load_and_clean_sparc_data(INPUT_FILE_NAME, OUTPUT_FILE)
