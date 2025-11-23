import pandas as pd
import numpy as np
import os

# --- Configuración y Rutas ---
# El archivo subido que contiene los datos de las curvas de rotación es SPARC_Lelli2016_Table2.txt.
# Aunque el usuario mencionó una estructura de carpetas, en este entorno, el archivo cargado 
# se accede directamente por su nombre. Usamos este nombre.
INPUT_FILE_NAME = 'SPARC_Lelli2016_Table2.txt'

# Definir la carpeta para guardar resultados intermedios y finales
RESULTS_DIR = 'btfr_results'
# Definir la ruta del archivo de salida limpio (CSV) dentro de la carpeta de resultados
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'sparc_data_cleaned.csv')

# Crear la carpeta de resultados si no existe
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Directorio creado: {RESULTS_DIR}")

# --- Definición de Anchos de Columna y Nombres (Basado en Table 2) ---
# Los datos están en formato de ancho fijo, definidos en la documentación de Lelli et al. (2016).
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

def load_and_clean_sparc_data(input_file, output_file):
    """
    Carga los datos de ancho fijo de SPARC, maneja errores y guarda en formato CSV.
    """
    try:
        # 1. Cargar los datos usando read_fwf (read fixed-width file)
        # Se asume que el archivo cargado está disponible por su nombre.
        df = pd.read_fwf(
            input_file,
            colspecs=colspecs,
            names=names,
            skiprows=8, # Omitir las primeras 8 líneas que son encabezados y descripción
            engine='python' # Usar motor Python para mejor manejo de colspecs
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

        # 4. Eliminar filas con valores NaN en las columnas críticas para el ajuste BTFR.
        df_clean = df.dropna(subset=['Vobs', 'Vgas', 'Vdisk', 'Vbul', 'D']).copy()

        # --- Cálculo de Variables Clave ---
        # 5. Calcular la Velocidad de Componentes Bariónicos (Vbar)
        # La relación es Vbar^2 = Vgas^2 + Vdisk^2 + Vbul^2
        df_clean['Vbar_sq'] = df_clean['Vgas']**2 + df_clean['Vdisk']**2 + df_clean['Vbul']**2
        df_clean['Vbar'] = np.sqrt(df_clean['Vbar_sq'])

        # 6. Eliminar filas con Vbar cero o nulo después del cálculo
        df_final = df_clean[df_clean['Vbar'] > 0].reset_index(drop=True)

        # 7. Guardar el DataFrame limpio en un archivo CSV
        df_final.to_csv(output_file, index=False)
        print("-" * 50)
        print(f"Proceso de limpieza completado.")
        print(f"Número total de puntos (filas) de datos limpios: {len(df_final)}")
        print(f"Archivo de datos limpio guardado en: {output_file}")
        print("-" * 50)

    except FileNotFoundError:
        print(f"ERROR: Archivo de entrada no encontrado en '{input_file}'.")
        print(f"Asegúrate de que el archivo '{INPUT_FILE_NAME}' esté disponible para el script.")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante el procesamiento: {e}")

if __name__ == "__main__":
    # Usamos el nombre del archivo subido.
    load_and_clean_sparc_data(INPUT_FILE_NAME, OUTPUT_FILE)
