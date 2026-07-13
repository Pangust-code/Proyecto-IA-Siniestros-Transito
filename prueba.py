from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent
file_path = base_dir / 'Dataset_Accidentes_Cuenca_2025.csv'

# 1. Cargar el dataset con el delimitador correcto.
# El archivo parece estar en codificación cp1252/latin1, así que usamos esa
# codificación para evitar errores al leer tildes y caracteres especiales.
df = pd.read_csv(file_path, sep=';', encoding='cp1252', decimal=',')

# 2. Comprobar la carga
print("--- PRIMERAS 5 FILAS DEL DATASET ---")
print(df.head())

# 3. Comprobar la estructura y tipos de datos
print("\n--- INFORMACIÓN DEL DATASET (TIPOS DE DATOS) ---")
print(df.info())

# 4. Comprobar valores nulos
print("\n--- CANTIDAD DE VALORES NULOS ---")
print(df.isnull().sum())

# 5. Comprobar que solo sea Cuenca
print("\n--- VALORES ÚNICOS EN LA COLUMNA 'CANTON' ---")
print(df['CANTON'].unique())

# 6. Limpiar nombres de columnas (quitar espacios en blanco accidentales)
df.columns = df.columns.str.strip()