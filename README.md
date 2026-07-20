# 🚦 Sistema de Alerta Temprana de Accidentes de Tránsito en Cuenca

Proyecto de Machine Learning con una red neuronal artificial para estimar el riesgo relativo de accidentes de tránsito en Cuenca, Ecuador. El sistema se organiza en tres fases metodológicas: preparación de datos, entrenamiento del modelo y generación de predicciones en una aplicación web interactiva[cite: 7].

## Resumen

El proyecto toma el historial de accidentes, lo limpia y lo transforma en variables útiles para el aprendizaje automático[cite: 7]. A partir de esa base, construye un modelo de regresión que devuelve un porcentaje de riesgo entre 0 y 100[cite: 7]. Luego, la fase final combina la predicción con datos históricos de contexto para producir una visualización dinámica en una página web.

La idea general es responder esta pregunta: dada una fecha, una hora y una ubicación concreta en Cuenca, ¿qué tan alto es el riesgo histórico relativo de accidentes en ese contexto?[cite: 7]. El resultado no es una probabilidad absoluta, sino un índice aprendido a partir de los patrones observados en los datos[cite: 7].

---

## 🌐 ¿Cómo funciona la Página Web? (Arquitectura del Sistema)

Nuestra plataforma web está diseñada para ser rápida, predictiva y tolerante a fallos, dividida en dos partes principales:

1. **Frontend (Interfaz de Usuario en React):** 
   - El usuario selecciona únicamente la fecha, la hora, la parroquia y si es feriado.
   - La interfaz muestra de forma dinámica un **Dashboard Analítico** con el Nivel de Riesgo predicho, un indicador de severidad (total de víctimas históricas), top de calles, siniestros y vehículos.
2. **Backend (API en FastAPI + Python):**
   - Recibe la petición del frontend, transforma los datos al vuelo y **consulta a la Red Neuronal** para obtener el riesgo exacto.
   - **Lógica Tolerante a Fallos:** Si el usuario elige una combinación que no tiene datos suficientes en una zona, el sistema busca los datos generales de la parroquia automáticamente, evitando pantallas en blanco y devolviendo un aviso inteligente en la interfaz.
   - Genera dinámicamente un **Mapa de Folium (HTML)** que filtra y muestra *solo* los puntos de calor del nivel de riesgo correspondiente.

---

## 🧠 Fases del Modelo de Machine Learning

### 🗂️ Fase 1: Preparación de Datos

*   Limpia y normaliza columnas, texto y coordenadas para evitar duplicados por escritura distinta o valores mal capturados[cite: 7].
*   Convierte la fecha y la hora a formatos numéricos y también crea variables derivadas como mes, día de la semana y rango horario[cite: 7].
*   Aplica transformaciones cíclicas con seno y coseno para que el modelo entienda que las horas y los días no “terminan” de forma brusca[cite: 7].
*   Genera el porcentaje de riesgo relativo y clasifica cada caso como bajo, medio o alto[cite: 7].
*   Exporta los datos procesados y el preprocesador en formato `.pkl`[cite: 7].

### 🤖 Fase 2: Entrenamiento del Modelo

*   Carga los datos preparados en la fase anterior y recupera el preprocesador ya ajustado[cite: 7].
*   Entrena una red neuronal con capas densas y `Dropout` para aprender relaciones no lineales entre ubicación, horario y riesgo[cite: 7].
*   Usa `EarlyStopping` y ajuste de tasa de aprendizaje para detener el entrenamiento cuando el modelo deja de mejorar[cite: 7].
*   Evalúa el modelo con MAE, MSE, RMSE y R² para medir el error y la capacidad explicativa[cite: 7].
*   Guarda el modelo final en formato `.keras`[cite: 7].

### 🚀 Fase 3: Predicción y Salida

*   Recibe nuevos datos del usuario y deriva automáticamente las variables temporales necesarias para el modelo[cite: 7].
*   Aplica el mismo preprocesador usado en el entrenamiento[cite: 7].
*   Genera una predicción de riesgo y la traduce a nivel bajo, medio o alto[cite: 7].
*   Busca información histórica relacionada para completar la salida con contexto útil, como la causa más frecuente o el vehículo más común[cite: 7].

---

## 🛠️ Tecnologías Utilizadas

* **Machine Learning & Datos:** `TensorFlow`, `Keras`, `Pandas`, `Scikit-Learn`, `Numpy`.
* **Backend & APIs:** `FastAPI`, `Uvicorn`.
* **Visualización Geográfica:** `Folium`, `Leaflet`.
* **Frontend:** `React`, `Tailwind CSS`.

---

## 📌 Ejecución del Proyecto (Desarrollo)

1. Instalar las dependencias de Python:
   ```bash
   pip install -r requirements.txt

## 🎥 Video Demostrativo

[![Demostración del Sistema de Alerta Temprana](https://img.youtube.com/vi/-Bo8oi-aF8o/maxresdefault.jpg)](https://youtu.be/-Bo8oi-aF8o)