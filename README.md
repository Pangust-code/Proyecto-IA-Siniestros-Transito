# 🚦 Sistema de Alerta Temprana de Accidentes de Tránsito en Cuenca

Este repositorio contiene la arquitectura y el desarrollo completo de un sistema inteligente de Machine Learning y Deep Learning orientado a la predicción, análisis espacial y gestión temprana de riesgos viales en Cuenca, Ecuador. 

> **Propósito del Proyecto:** Desarrollar una solución basada en Redes Neuronales Artificiales para estimar el riesgo histórico relativo de siniestros de tránsito, integrando un backend robusto en Python, una interfaz web interactiva y análisis geoespacial.

---

## 👨‍💻 Contexto Académico / Profesional

**Proyecto de Inteligencia Artificial y Aprendizaje Automático**
- **Institución:** Universidad Politécnica Salesiana
- **Equipo de Desarrollo:** Diana Avila, Sebastian Cabrera, Valeria Mantilla, Daniel Guanga

**Competencias demostradas:**
- 🏗️ Pipeline de ingeniería de datos y transformación espacial (Geolocalización).
- 🤖 Modelado predictivo con Redes Neuronales Artificiales en TensorFlow / Keras.
- 📊 Despliegue de lógica predictiva y analítica orientada a la toma de decisiones.
- 🔌 Integración de modelos analíticos con interfaces web interactivas y mapas de calor dinámicos.

---

## 🎥 Video Demo

**Demostración Técnica del Sistema:**

[![Video de Demostración del Sistema de Alerta Temprana](https://img.youtube.com/vi/-Bo8oi-aF8o/maxresdefault.jpg)](https://youtu.be/-Bo8oi-aF8o)

### Estructura de la Demostración:
- **El Problema:** Exposición de la alta incidencia de accidentes en Cuenca y la limitación de reportes estáticos tradicionales.
- **La Solución Técnica:** Explicación del funcionamiento del modelo de red neuronal y el procesamiento en el backend.
- **Demostración de la Web:** Visualización en vivo de la consulta de riesgo por fecha, hora, parroquia y zona, junto con el despliegue del mapa de puntos y gráficos analíticos.

---

## 🎯 El Problema y la Solución

### El Desafío
Las ciudades modernas enfrentan el reto constante de reducir la siniestralidad vial. Las herramientas tradicionales suelen ofrecer reportes históricos estáticos que dificultan:
- ❌ Anticipar zonas críticas en franjas horarias específicas.
- ❌ Visualizar la severidad y las causas principales de forma dinámica.
- ❌ Brindar acceso rápido a ciudadanos o autoridades mediante una interfaz ágil.

### La Solución: Sistema Inteligente de Alerta Temprana
Una plataforma adaptativa que:
- **Procesa datos históricos complejos** aplicando transformaciones cíclicas temporales y validación geoespacial.
- **Estima un índice de riesgo relativo** (0% a 100%) mediante una red neuronal entrenada con patrones reales de Cuenca.
- **Despliega un mapa interactivo inteligente** que filtra puntos críticos de acuerdo con el nivel de riesgo predicho por la IA.
- **Genera un panel analítico completo** con el top de causas, tipos de siniestros, vehículos involucrados e indicador de severidad (víctimas).

---

## 📊 Rendimiento y Evaluación del Modelo

### Métricas de Validación (Conjunto de Prueba)

| Métrica | Valor Obtenido | Interpretación Técnica |
| :--- | :--- | :--- |
| **MAE (Error Absoluto Medio)** | 9.84 | Desviación promedio porcentual baja en las estimaciones. |
| **RMSE (Raíz del Error Cuadrático)** | 13.28 | Control efectivo de errores atípicos en el modelo. |
| **R² (Coeficiente de Determinación)** | 0.6126 | El modelo explica de manera sólida la variabilidad del riesgo histórico. |
| **Exactitud de Clasificación** | 68.42% | Precisión al clasificar los niveles de alerta (Bajo, Medio, Alto). |

---

## 📁 Estructura del Proyecto & Flujo de Trabajo

El sistema está dividido en tres fases secuenciales implementadas en Jupyter Notebooks, complementadas con scripts de producción:

```text
Proyecto-Siniestros-Transito/
├── Fase_1_Riesgo_Ubicacion_Accidentes_Cuenca.ipynb   # Limpieza, ingeniería de variables y exportación
├── Fase_2_Entrenamiento_Riesgo_Ubicacion_Cuenca.ipynb # Entrenamiento de la Red Neuronal y métricas
├── Fase_3_Completa_Riesgo_Accidentes_Cuenca.ipynb      # Inferencia, contexto histórico y exportación Power BI
├── dataset_modelo_riesgo.csv                         # Base agregada para el entrenamiento
├── resumen_historico_powerbi.csv                     # Indicadores estadísticos para reportes
├── accidentes_limpios_powerbi.csv                    # Registros individuales limpios para mapas
├── modelo_riesgo_ubicacion_cuenca.keras              # Red neuronal entrenada
└── datos_fase1_riesgo_ubicacion.pkl                  # Pipeline de preprocesamiento guardado
```
---

## 🛠️ Tecnologías Utilizadas

### Backend & Machine Learning
- **Python 3.12+**
- **TensorFlow / Keras:** Diseño y entrenamiento de la red neuronal artificial de regresión.
- **Scikit-Learn:** Canalizaciones de preprocesamiento (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`).
- **Pandas & NumPy:** Manipulación y transformación de matrices de datos e ingeniería de características cíclicas.

### Visualización & Web (Prototipo POC)
- **FastAPI:** Servidor backend de alto rendimiento para procesar las peticiones de la IA.
- **Folium & Leaflet:** Generación dinámica de mapas interactivos basados en coordenadas geográficas.
- **React & Tailwind CSS:** Interfaz de usuario moderna en formato Dashboard analítico.

---

## 🏗️ Metodología Detallada (Las 3 Fases)

### Fase 1: Preparación y Limpieza de Datos
1. **Limpieza Geoespacial y Textual:** Estandarización de nombres de parroquias, eliminación de espacios y corrección estricta de coordenadas atípicas utilizando medianas por zona.
2. **Ingeniería de Variables Temporales:** Extracción de mes, día de la semana y horas. Aplicación de transformaciones trigonométricas cíclicas (`seno` y `coseno`) para las horas (24h) y días (7 días), permitiendo que la red comprenda la naturaleza circular del tiempo.
3. **Cálculo de Riesgo Relativo:** Agrupación por combinaciones de tiempo y ubicación para calcular la frecuencia de accidentes, normalizándola frente al pico máximo histórico (11 accidentes) para obtener un porcentaje de `0%` a `100%`.

### Fase 2: Entrenamiento de la Red Neuronal
1. **Arquitectura del Modelo:** Red secuencial multicapa densas (`Dense 64` $\rightarrow$ `32` $\rightarrow$ `16` $\rightarrow$ `1`) con activación `ReLU` en las capas ocultas y `Sigmoid` en la salida, escalada posteriormente a porcentaje.
2. **Estrategia de Regularización:** Uso de capas `Dropout` (0.3 y 0.2) para mitigar el sobreajuste.
3. **Callbacks de Optimización:** Implementación de `EarlyStopping` (paciencia de 12 épocas) y `ReduceLROnPlateau` para garantizar una convergencia suave y estable.

### Fase 3: Predicción, Contexto y Salida
1. **Inferencia Dinámica:** Carga de los artefactos (`.keras` y `.pkl`) para recibir variables en tiempo real desde la web.
2. **Cruce de Contexto Histórico:** Adición automática de información de soporte (causa principal, tipo de vehículo más común, promedios de lesionados y fallecidos).
3. **Exportación de Datos:** Generación de estructuras limpias en CSV optimizados con separador `;` y coma decimal para tableros analíticos en Power BI.

---

## 📌 Guía de Instalación y Ejecución local

1. Clona el repositorio y sitúate en la carpeta del proyecto.
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt

---

## 📌 Repositorios del Backend y Frontend de nuestro proyecto

- **Frontend**: https://github.com/Pangust-code/ProyectoIA-Siniestros-Transito-Frontend

- **Backend**: https://github.com/Pangust-code/ProyectoIA-Siniestros-Transito-Backend
