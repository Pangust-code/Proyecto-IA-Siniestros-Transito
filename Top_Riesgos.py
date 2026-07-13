import pandas as pd
import numpy as np
import pickle
import itertools
from tensorflow.keras.models import load_model

# 1. CARGAR ARTEFACTOS
print("Cargando cerebro de la IA...")
modelo = load_model('modelo_alerta_cuenca_RNA.keras')

with open('encoders_fase2.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler_fase2.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2. OBTENER TODAS LAS VARIABLES REGISTRADAS
# Extraemos dinámicamente las parroquias y zonas que el modelo conoce
parroquias = label_encoders['PARROQUIA'].classes_
zonas = label_encoders['ZONA'].classes_
meses = range(1, 13)      # Meses del 1 al 12
dias = range(7)           # 0=Lunes a 6=Domingo
horas = range(24)         # 0 a 23 horas

# 3. CONSTRUIR EL UNIVERSO DE ESCENARIOS
print("Generando el universo de combinaciones posibles...")
# itertools.product crea todas las combinaciones posibles entre las listas
combinaciones = list(itertools.product(parroquias, zonas, meses, dias, horas))

df_total = pd.DataFrame(combinaciones, columns=['PARROQUIA', 'ZONA', 'MES', 'DIA_SEMANA', 'HORA_NUM'])
print(f"Total de escenarios a evaluar: {len(df_total)} simulaciones.")

# 4. TRADUCIR Y ESCALAR PARA LA IA
df_procesado = df_total.copy()
df_procesado['PARROQUIA'] = label_encoders['PARROQUIA'].transform(df_procesado['PARROQUIA'])
df_procesado['ZONA'] = label_encoders['ZONA'].transform(df_procesado['ZONA'])

X_scaled = scaler.transform(df_procesado)

# 5. PREDECIR Y EXTRAER LA PROBABILIDAD DE 'ALTO'
print("Calculando riesgos con la Red Neuronal...")
predicciones = modelo.predict(X_scaled, verbose=0)

# Buscamos qué columna de salida corresponde a la palabra 'ALTO'
indice_alto = np.where(label_encoders['NIVEL_RIESGO'].classes_ == 'ALTO')[0][0]

# Extraemos solo la probabilidad de que sea ALTO y la guardamos
df_total['PROB_ALTO'] = predicciones[:, indice_alto] * 100

# También guardamos cuál fue la predicción ganadora final
clases_ganadoras = np.argmax(predicciones, axis=1)
df_total['RESULTADO_FINAL'] = label_encoders['NIVEL_RIESGO'].inverse_transform(clases_ganadoras)

# 6. ORDENAR PARA SACAR EL TOP DE PELIGRO
# Ordenamos de mayor a menor basándonos puramente en la probabilidad de ALTO
top_peligro = df_total.sort_values(by='PROB_ALTO', ascending=False).head(15).copy()

# --- Darle un formato bonito para imprimir ---
dias_nombres = {0:'Lunes', 1:'Martes', 2:'Miércoles', 3:'Jueves', 4:'Viernes', 5:'Sábado', 6:'Domingo'}
top_peligro['DIA_SEMANA'] = top_peligro['DIA_SEMANA'].map(dias_nombres)
top_peligro['HORA_NUM'] = top_peligro['HORA_NUM'].apply(lambda x: f"{x:02d}:00")
top_peligro['PROB_ALTO'] = top_peligro['PROB_ALTO'].apply(lambda x: f"{x:.2f}%")

print("\n🚨 EL TOP 15 DE ESCENARIOS CON MAYOR RIESGO DE ACCIDENTES EN CUENCA 🚨")
print(top_peligro.to_string(index=False))