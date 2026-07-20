# Sistema de Alerta Temprana de Accidentes de Tránsito en Cuenca

Proyecto de Machine Learning con una red neuronal artificial para estimar el riesgo relativo de accidentes de tránsito en Cuenca, Ecuador. El sistema se organiza en tres notebooks: preparación de datos, entrenamiento del modelo y generación de predicciones listas para Power BI.

## Resumen

El proyecto toma el historial de accidentes, lo limpia y lo transforma en variables útiles para aprendizaje automático. A partir de esa base, construye un modelo de regresión que devuelve un porcentaje de riesgo entre 0 y 100. Luego, la fase final combina la predicción con datos históricos de contexto para producir salidas útiles en análisis y visualización.

La idea general es responder esta pregunta: dada una fecha, una hora y una ubicación concreta en Cuenca, ¿qué tan alto es el riesgo histórico relativo de accidentes en ese contexto? El resultado no es una probabilidad absoluta, sino un índice aprendido a partir de los patrones observados en los datos.

## Cómo funciona

### Fase 1: preparación de datos

- Limpia y normaliza columnas, texto y coordenadas para evitar duplicados por escritura distinta o valores mal capturados.
- Convierte la fecha y la hora a formatos numéricos y también crea variables derivadas como mes, día de la semana y rango horario.
- Aplica transformaciones cíclicas con seno y coseno para que el modelo entienda que las horas y los días no “terminan” de forma brusca.
- Agrupa los accidentes por combinación de tiempo y ubicación para calcular frecuencia histórica.
- Genera el porcentaje de riesgo relativo y clasifica cada caso como bajo, medio o alto.
- Exporta los datos procesados, el preprocesador y archivos auxiliares para las fases siguientes y para Power BI.

En esta fase también se generan tablas útiles para análisis externo, por ejemplo un resumen histórico con métricas como la causa más frecuente, el tipo de siniestro más común, el vehículo más repetido y los promedios de lesionados y fallecidos.

### Fase 2: entrenamiento del modelo

- Carga los datos preparados en la fase anterior y recupera el preprocesador ya ajustado.
- Entrena una red neuronal con capas densas y `Dropout` para aprender relaciones no lineales entre ubicación, horario y riesgo.
- Usa `EarlyStopping` y ajuste de tasa de aprendizaje para detener el entrenamiento cuando el modelo deja de mejorar.
- Convierte la salida a un valor entre 0 y 1 durante el entrenamiento y luego la reescala a porcentaje.
- Evalúa el modelo con MAE, MSE, RMSE y R² para medir el error y la capacidad explicativa.
- Guarda el modelo final en formato `.keras`, junto con métricas, historial y resultados de prueba.

Además de la evaluación numérica, esta fase permite comparar visualmente los valores reales frente a los predichos y revisar cómo se comporta la clasificación final en niveles bajo, medio y alto.

### Fase 3: predicción y salida

- Recibe nuevos datos de fecha, hora, parroquia, zona y feriado desde una entrada manual o desde una interfaz.
- Deriva automáticamente las variables temporales necesarias para el modelo, como el mes, el día de la semana y el rango horario.
- Aplica el mismo preprocesador usado en el entrenamiento para mantener consistencia entre lo que vio el modelo y lo que predice.
- Genera una predicción de riesgo y la traduce a nivel bajo, medio o alto.
- Busca información histórica relacionada para completar la salida con contexto útil, como la causa más frecuente o el vehículo más común.
- Exporta archivos CSV listos para Power BI, análisis geográfico o consumo en otras aplicaciones.

Esta fase está pensada para ser la puerta de salida del proyecto: combina el modelo, el histórico y los archivos de exportación en un solo flujo para que la predicción se pueda reutilizar en reportes o en una aplicación.

## Estructura del proyecto

- [Fase 1](Fase_1_Riesgo_Ubicacion_Accidentes_Cuenca.ipynb): limpieza, ingeniería de variables y exportación de datos.
- [Fase 2](Fase_2_Entrenamiento_Riesgo_Ubicacion_Cuenca.ipynb): entrenamiento y evaluación del modelo.
- [Fase 3](Fase_3_Completa_Riesgo_Accidentes_Cuenca.ipynb): predicción, contexto histórico y exportación final.

## Archivos principales

- `dataset_modelo_riesgo.csv`: base agregada usada para entrenar y analizar el riesgo. Contiene la combinación de variables de tiempo y ubicación con su porcentaje de riesgo relativo.
- `resumen_historico_powerbi.csv`: tabla con indicadores históricos para reportes. Sirve para enriquecer la visualización con contexto estadístico.
- `accidentes_limpios_powerbi.csv`: registros limpios para análisis y mapa. Conserva los campos necesarios para explorar la información geográfica.
- `modelo_riesgo_ubicacion_cuenca.keras`: modelo entrenado y listo para inferencia. Es el archivo que usa la Fase 3 para predecir.
- `datos_fase1_riesgo_ubicacion.pkl`: preprocesador, variables y particiones de entrenamiento/prueba. Permite reproducir el mismo procesamiento en otras sesiones.
- `predicciones_riesgo_powerbi.csv`, `mapa_riesgo_powerbi.csv`, `ultima_prediccion_riesgo.csv`: salidas generadas en la Fase 3 para consumo directo en Power BI o para revisar una predicción puntual.

## Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Uso recomendado

1. Ejecuta primero la Fase 1 para generar los datos procesados.
2. Ejecuta la Fase 2 para entrenar y guardar el modelo.
3. Ejecuta la Fase 3 para probar predicciones y exportar archivos para Power BI.

## Nota

El resultado del modelo es un índice de riesgo histórico relativo, no una probabilidad absoluta de accidente.

## Video

https://youtu.be/-Bo8oi-aF8o?feature=shared
