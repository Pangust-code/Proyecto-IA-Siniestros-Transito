import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 1. CARGAR LOS ARTEFACTOS GENERADOS EN LA FASE 2
print("Cargando modelo de Red Neuronal y diccionarios...")
modelo = load_model('modelo_alerta_cuenca_RNA.keras')

with open('encoders_fase2.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler_fase2.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2. FUNCIÓN DE INFERENCIA PARA NUEVOS EJEMPLOS
def predecir_nivel_riesgo(fecha_str, hora_str, parroquia, zona):
    """
    Toma datos crudos de entrada, los procesa usando los encoders/scaler
    y retorna la predicción del nivel de riesgo con su probabilidad.
    """
    # Extraer variables temporales igual que en el entrenamiento
    fecha = pd.to_datetime(fecha_str)
    mes = fecha.month
    dia_semana = fecha.dayofweek # 0=Lunes, 6=Domingo
    hora_num = int(hora_str.split(':')[0])
    
    # Validar que la parroquia exista en los datos de entrenamiento
    if parroquia.upper() not in label_encoders['PARROQUIA'].classes_:
        return f"Error: La parroquia '{parroquia}' no está registrada en el sistema."
    
    # Traducir los textos a números usando el archivo .pkl cargado
    parroquia_enc = label_encoders['PARROQUIA'].transform([parroquia.upper()])[0]
    zona_enc = label_encoders['ZONA'].transform([zona.upper()])[0]
    
    # Crear el vector de características como DataFrame para evitar warnings
    df_entrada = pd.DataFrame(
        [[parroquia_enc, zona_enc, mes, dia_semana, hora_num]], 
        columns=['PARROQUIA', 'ZONA', 'MES', 'DIA_SEMANA', 'HORA_NUM']
    )
    
    # Escalar numéricamente el vector
    vector_escalado = scaler.transform(df_entrada)
    
    # Realizar la predicción con la Red Neuronal
    predicciones = modelo.predict(vector_escalado, verbose=0)
    
    # Obtener la clase con mayor probabilidad (0, 1 o 2)
    clase_ganadora = np.argmax(predicciones[0])
    probabilidad_ganadora = np.max(predicciones[0]) * 100
    
    # Destraducir el número a la etiqueta de texto (BAJO, MEDIO, ALTO)
    nivel_riesgo = label_encoders['NIVEL_RIESGO'].inverse_transform([clase_ganadora])[0]
    
    return {
        "Fecha Evaluada": fecha_str,
        "Hora Evaluada": hora_str,
        "Parroquia": parroquia,
        "Zona": zona,
        "Nivel de Riesgo Predicho": nivel_riesgo,
        "Confianza del Modelo": f"{probabilidad_ganadora:.2f}%"
    }

# 3. PRUEBA DEL PROTOTIPO CON UN NUEVO CASO (Simulación de consulta)
# Imagina que un usuario consulta qué pasaría un viernes en la noche en Tarqui
print("\\n--- Ejecutando consulta de prueba de alerta temprana ---")

resultado_consulta = predecir_nivel_riesgo(
    fecha_str="2026-01-5",  # Fecha futura (un viernes)
    hora_str="23:30:00",     # Hora pico en la noche
    parroquia="CUENCA",  # Parroquia conocida
    zona="URBANA"  # Zona conocida
)

# Mostrar el JSON final en pantalla de forma ordenada
import json
print(json.dumps(resultado_consulta, indent=4, ensure_ascii=False))

# --- 4. PRUEBA MASIVA DE PREDICCIONES ---
print("\n--- Ejecutando Prueba Masiva (100 escenarios aleatorios) ---")

# 1. Cargar el dataset para sacar ejemplos reales
df_pruebas = pd.read_csv('dataset_riesgo_cuenca.csv')

# Tomar 100 ejemplos al azar (puedes cambiar este número)
ejemplos_aleatorios = df_pruebas.sample(n=100, random_state=42).copy()

# 2. Preparar los datos tal como lo hicimos individualmente
X_pruebas = ejemplos_aleatorios[['PARROQUIA', 'ZONA', 'MES', 'DIA_SEMANA', 'HORA_NUM']].copy()

# Traducir textos a números
X_pruebas['PARROQUIA'] = label_encoders['PARROQUIA'].transform(X_pruebas['PARROQUIA'])
X_pruebas['ZONA'] = label_encoders['ZONA'].transform(X_pruebas['ZONA'])

# Escalar los datos
X_pruebas_scaled = scaler.transform(X_pruebas)

# 3. Predecir todos los ejemplos de golpe
predicciones_masivas = modelo.predict(X_pruebas_scaled, verbose=0)

# Obtener la clase ganadora para cada uno
clases_predichas = np.argmax(predicciones_masivas, axis=1)

# Destraducir los números a texto (BAJO, MEDIO, ALTO)
etiquetas_predichas = label_encoders['NIVEL_RIESGO'].inverse_transform(clases_predichas)

# 4. Contar y mostrar los resultados
ejemplos_aleatorios['PREDICCION_IA'] = etiquetas_predichas

# Contar cuántos salieron de cada nivel
conteo_resultados = ejemplos_aleatorios['PREDICCION_IA'].value_counts()

print("Distribución de las 100 predicciones:")
print(conteo_resultados)

# (Opcional) Mostrar los casos que salieron con riesgo ALTO para analizarlos
casos_altos = ejemplos_aleatorios[ejemplos_aleatorios['PREDICCION_IA'] == 'ALTO']

if not casos_altos.empty:
    print(f"\nSe encontraron {len(casos_altos)} casos con riesgo ALTO. Aquí tienes algunos:")
    # Mostramos las columnas clave para ver dónde y cuándo la IA detecta peligro
    print(casos_altos[['PARROQUIA', 'ZONA', 'MES', 'DIA_SEMANA', 'HORA_NUM']].head())
else:
    print("\nEn esta muestra aleatoria de 100, la IA no detectó ningún riesgo ALTO.")