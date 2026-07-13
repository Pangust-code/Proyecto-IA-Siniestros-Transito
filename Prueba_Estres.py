import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 1. CARGAR EL MODELO Y ARTEFACTOS
print("Cargando modelo de Red Neuronal y diccionarios...")
modelo = load_model('modelo_alerta_cuenca_RNA.keras')

with open('encoders_fase2.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler_fase2.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2. GENERAR LA SIMULACIÓN (Toda una semana, hora por hora)
print("\nGenerando escenarios: 7 días de la semana, 24 horas al día...")
escenarios = []
parroquia = "CUENCA"
zona = "URBANA"
mes = 1  # Enero

for dia_semana in range(7): # 0=Lunes a 6=Domingo
    for hora_num in range(24): # 0 a 23 horas
        escenarios.append([parroquia, zona, mes, dia_semana, hora_num])

# Convertir la lista a DataFrame (168 filas en total)
df_simulacion = pd.DataFrame(escenarios, columns=['PARROQUIA', 'ZONA', 'MES', 'DIA_SEMANA', 'HORA_NUM'])

# 3. PREPARAR LOS DATOS PARA LA IA
df_procesado = df_simulacion.copy()
df_procesado['PARROQUIA'] = label_encoders['PARROQUIA'].transform(df_procesado['PARROQUIA'])
df_procesado['ZONA'] = label_encoders['ZONA'].transform(df_procesado['ZONA'])

# Escalar (sin que salga el mensaje de warning)
X_scaled = scaler.transform(df_procesado)

# 4. EJECUTAR LA PREDICCIÓN MASIVA
predicciones = modelo.predict(X_scaled, verbose=0)
clases_predichas = np.argmax(predicciones, axis=1)

# Destraducir los números de vuelta a texto
df_simulacion['NIVEL_RIESGO'] = label_encoders['NIVEL_RIESGO'].inverse_transform(clases_predichas)

# 5. MOSTRAR RESULTADOS
print("\n--- RESULTADOS DE LA SEMANA COMPLETA EN CUENCA (URBANA) ---")
print(f"Total de horas simuladas: {len(df_simulacion)}")
print(df_simulacion['NIVEL_RIESGO'].value_counts())

# 6. REVISAR CUÁNDO OCURRE EL RIESGO ALTO
casos_altos = df_simulacion[df_simulacion['NIVEL_RIESGO'] == 'ALTO']

if not casos_altos.empty:
    print("\n¡Atención! La IA detectó riesgo ALTO en los siguientes momentos:")
    
    # Diccionario para que los días salgan con nombre y no con número
    dias_nombres = {0:'Lunes', 1:'Martes', 2:'Miércoles', 3:'Jueves', 4:'Viernes', 5:'Sábado', 6:'Domingo'}
    casos_altos_visual = casos_altos.copy()
    casos_altos_visual['DIA_SEMANA'] = casos_altos_visual['DIA_SEMANA'].map(dias_nombres)
    
    # Formatear la hora para que se vea bonita
    casos_altos_visual['HORA'] = casos_altos_visual['HORA_NUM'].apply(lambda x: f"{x:02d}:00")
    
    print(casos_altos_visual[['DIA_SEMANA', 'HORA']].to_string(index=False))
else:
    print("\nNo se detectó riesgo ALTO en esta simulación.")