import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# 1. Cargar el dataset que generaste en la Fase 1
# Nota: Asegúrate de guardar el 'dataset_riesgo' de la Fase 1 en un CSV antes de correr esto.
df = pd.read_csv('dataset_riesgo_cuenca.csv') 

# 2. REAJUSTE INTELIGENTE DE UMBRALES (Evitando el desbalanceo extremo)
def clasificar_riesgo_realista(accidentes):
    if accidentes <= 1:
        return 'BAJO'
    elif accidentes == 2:
        return 'MEDIO'
    else:
        return 'ALTO' # 3, 4 o 5 accidentes

df['NIVEL_RIESGO'] = df['ACCIDENTES'].apply(clasificar_riesgo_realista)
print("Nueva distribución de riesgo:\\n", df['NIVEL_RIESGO'].value_counts())

# 3. SELECCIÓN DE VARIABLES (Features y Target)
# Usamos solo las variables que influyen en la predicción
X = df[['PARROQUIA', 'ZONA', 'MES', 'DIA_SEMANA', 'HORA_NUM']]
y = df['NIVEL_RIESGO']

# 4. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
label_encoders = {}
for col in ['PARROQUIA', 'ZONA']:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Codificar el target (BAJO=0, MEDIO=1, ALTO=2)
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)
label_encoders['NIVEL_RIESGO'] = le_y

# Guardar los encoders para la interfaz final (Power BI o Web)
with open('encoders_fase2.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# 5. ESCALADO DE DATOS Y SEPARACIÓN (Train / Test)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open('scaler_fase2.pkl', 'wb') as f:
    pickle.dump(scaler, f)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 6. CÁLCULO DE PESOS DE CLASE (Para penalizar errores en la clase ALTO)
pesos = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(pesos))
print("Pesos asignados a las clases (0:ALTO, 1:BAJO, 2:MEDIO):", class_weights)

# 7. CONSTRUCCIÓN DE LA RED NEURONAL ARTIFICIAL (RNA)
modelo = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3), # Previene el sobreajuste (overfitting)
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax') # 3 neuronas de salida para Clasificación Multiclase
])

modelo.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 8. ENTRENAMIENTO DEL MODELO
print("Iniciando entrenamiento de la RNA...")
historial = modelo.fit(X_train, y_train, 
                       epochs=80, 
                       batch_size=16, 
                       validation_split=0.2, 
                       class_weight=class_weights, # Aquí aplicamos el balanceo
                       verbose=1)

# 9. EVALUACIÓN Y GUARDADO
loss, accuracy = modelo.evaluate(X_test, y_test)
print(f"Precisión final en datos desconocidos: {accuracy*100:.2f}%")

modelo.save('modelo_alerta_cuenca_RNA.keras')
print("¡Modelo exportado exitosamente!")