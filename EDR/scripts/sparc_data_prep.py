import pandas as pd
import numpy as np
import os

# --- Configuración y Rutas ---
# Definir la ruta del archivo de datos de entrada
INPUT_FILE = 'SPARC_Lelli2016_Table2.txt'
# Definir la ruta del archivo de salida limpio (CSV)
OUTPUT_FILE = 'sparc_data_cleaned.csv'
# Definir la carpeta para guardar resultados intermedios y finales
RESULTS_DIR = 'btfr_results'

# Crear la carpeta de resultados si no existe
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Directorio creado: {RESULTS_DIR}")

# --- Definición de Anchos de Columna y Nombres ---
# Los datos están en formato de ancho fijo, definidos en la documentación de Lelli et al. (2016).
# Usamos 'colspecs' para definir el rango de columnas (índices 0-base) y 'names' para las etiquetas.
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
        # Cargar los datos usando read_fwf (read fixed-width file)
        df = pd.read_fwf(
            input_file,
            colspecs=colspecs,
            names=names,
            skiprows=8, # Omitir las primeras 8 líneas que son encabezados y descripción
            engine='python' # Usar motor Python para mejor manejo de colspecs
        )

        print(f"Datos cargados exitosamente desde: {input_file}")

        # --- Limpieza de Datos ---
        # 1. Reemplazar valores de texto (como '-----') por NaN (Not a Number)
        # Esto es crucial para poder convertir las columnas a tipos numéricos.
        df = df.replace(to_replace='-', value=np.nan)
        df = df.replace(to_replace='------', value=np.nan)
        df = df.replace(to_replace='-------', value=np.nan)

        # 2. Convertir columnas relevantes a tipo numérico (float)
        # Las velocidades y las distancias son clave para los cálculos posteriores.
        numeric_cols = ['D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
        for col in numeric_cols:
            # Forzar la conversión. Si hay un error, el valor será NaN.
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Eliminar filas con valores NaN en las columnas críticas para el ajuste.
        # Las filas deben tener Vobs y los componentes Vgas, Vdisk, Vbul definidos.
        df_clean = df.dropna(subset=['Vobs', 'Vgas', 'Vdisk', 'Vbul', 'D']).copy()

        # --- Cálculo de Variables Clave ---
        # 4. Calcular la Velocidad de Componentes Bariónicos (Vbar)
        # La relación es Vbar^2 = Vgas^2 + Vdisk^2 + Vbul^2
        df_clean['Vbar_sq'] = df_clean['Vgas']**2 + df_clean['Vdisk']**2 + df_clean['Vbul']**2
        df_clean['Vbar'] = np.sqrt(df_clean['Vbar_sq'])

        # 5. Calcular la Masa Bariónica (Mbar) a partir de Vbar (para referencia)
        # Usamos la aproximación Mbar ~ (Vbar^2 * R / G)
        # G = 4.3009 x 10^-6 (km/s)^2 kpc M_solar^-1
        G_CONST = 4.3009e-6 # G en unidades (km/s)^2 kpc M_solar^-1

        # Mbar (log) se calcula después del ajuste BTFR, pero Vbar es esencial aquí.

        # 6. Eliminar filas con Vbar cero o nulo después del cálculo (seguridad)
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
        print("Asegúrate de que 'SPARC_Lelli2016_Table2.txt' esté en la misma carpeta.")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante el procesamiento: {e}")

if __name__ == "__main__":
    load_and_clean_sparc_data(INPUT_FILE, OUTPUT_FILE)
