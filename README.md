# 🚦 Sistema de Alerta Temprana de Accidentes de Tránsito en Cuenca

Este proyecto implementa un modelo de Machine Learning (Red Neuronal Artificial) diseñado para predecir el riesgo de accidentes de tránsito en la ciudad de Cuenca, Ecuador. El sistema se divide en tres fases principales que abarcan desde el procesamiento de datos históricos hasta la generación de predicciones para su visualización en Power BI.

## 🗂️ Fase 1: Preparación y Limpieza de Datos
En esta etapa se estructura el conocimiento histórico para que la Inteligencia Artificial pueda aprender los patrones de accidentalidad.

*   **Variables clave:** Se preparó el dataset extrayendo y limpiando características espaciotemporales: Fecha, Hora, Parroquia, Zona (Urbana/Rural) y Feriado.
*   **Ingeniería de características:** Se aplicaron transformaciones matemáticas cíclicas (seno y coseno) a las horas y días de la semana para que el modelo entienda la continuidad del tiempo.
*   **Cálculo de Riesgo Relativo:** Se generó un porcentaje de riesgo (0% a 100%) dividiendo los accidentes de cada combinación específica entre el máximo histórico de accidentes observado. 
*   **Clasificación de Niveles:** Se establecieron umbrales para categorizar el riesgo en Bajo (0-39%), Medio (40-69%) y Alto (70-100%).
*   **Exportación:** Se dividieron los datos en conjuntos de entrenamiento y prueba (80/20) y se guardó el preprocesador en formato `.pkl`.

## 🧠 Fase 2: Entrenamiento del Modelo de IA
Esta fase se centra en la construcción, entrenamiento y evaluación de la Red Neuronal de regresión.

*   **Arquitectura:** Se diseñó un modelo Secuencial con capas ocultas Densas (64, 32 y 16 neuronas) usando activación `ReLU` y capas de `Dropout` para evitar el sobreajuste.
*   **Capa de salida:** Se utilizó una única neurona con activación `sigmoid`, cuyo resultado se escala para obtener el porcentaje final de riesgo.
*   **Optimización:** El entrenamiento se realizó con el optimizador `Adam`, utilizando `EarlyStopping` (para detener el entrenamiento si no hay mejora) y `ReduceLROnPlateau` (para ajustar la tasa de aprendizaje).
*   **Evaluación:** El modelo se validó utilizando métricas estadísticas como MAE, MSE, RMSE y R², y se guardó el modelo final en formato `.keras`.

## 🚀 Fase 3: Predicción y Salida a Producción
La última fase integra los artefactos creados para interactuar con el usuario y alimentar paneles de inteligencia de negocios.

*   **Ingreso de datos:** El sistema recibe parámetros manuales del usuario (Fecha, Hora, Parroquia, Zona y Feriado) y los transforma automáticamente usando el preprocesador de la Fase 1.
*   **Predicción y Contexto:** La IA calcula el nivel de riesgo y busca en el historial para adjuntar datos estadísticos de la zona, como: causa probable, siniestro más común, vehículo frecuente y promedios de víctimas.
*   **Integración con Power BI:** Todos los resultados, incluyendo las coordenadas geográficas (Latitud y Longitud) y la clasificación del mapa, se exportan automáticamente a archivos CSV optimizados (separados por punto y coma y con coma decimal) para su consumo directo en herramientas como Power BI.

link del video:
https://youtu.be/-Bo8oi-aF8o?feature=shared
