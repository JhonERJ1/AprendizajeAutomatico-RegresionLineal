# AprendizajeAutomatico-RegresionLineal
Proyecto de Machine Learning enfocado en la implementación de modelos de regresión lineal, exploración de datos y evaluación de resultados utilizando herramientas de análisis.
# AprendizajeAutomatico-RegresionLineal
Proyecto de Machine Learning enfocado en la implementación de modelos de regresión lineal, exploración de datos y evaluación de resultados utilizando herramientas de análisis.

-Documentación del Proyecto

1. Introducción
El presente proyecto tiene como objetivo desarrollar un sistema de predicción del precio futuro de un activo financiero (en este caso, Intel Corporation - INTC) utilizando técnicas de aprendizaje automático supervisado, con énfasis en modelos de regresión.

A partir de datos históricos del mercado bursátil, se busca identificar patrones entre distintas variables financieras (precios, volumen, indicadores técnicos) para estimar el comportamiento del precio del activo. Se implementaron múltiples modelos de regresión (lineales y no lineales), se evaluaron mediante métricas estadísticas y se seleccionó el modelo con mejor desempeño: Lasso Regression.

El sistema final permite, a partir de un archivo CSV con nuevos datos, generar predicciones del precio de cierre del día siguiente, manteniendo la consistencia del preprocesamiento utilizado durante el entrenamiento.

2. Dataset Utilizado
El dataset consta de datos históricos diarios de Intel (INTC) desde el 1 de enero de 2019 hasta el 31 de diciembre de 2024, obtenidos a través de la API de Alpaca Markets. Contiene 1094 registros después del preprocesamiento.

Las variables principales del dataset son:

Variable	Descripción
open	Precio de apertura del día
high	Precio máximo del día
low	Precio mínimo del día
close	Precio de cierre del día
volume	Volumen de acciones negociadas
trade_count	Número de transacciones realizadas
vwap	Precio promedio ponderado por volumen
A partir de estas variables, se generaron nuevas características derivadas para enriquecer el modelo:

Característica	Descripción
daily_return	Cambio porcentual del precio de cierre respecto al día anterior
ma_7	Media móvil simple de 7 días del precio de cierre
ma_21	Media móvil simple de 21 días del precio de cierre
volatility_7	Desviación estándar de los retornos diarios en una ventana de 7 días
daily_range	Diferencia entre el precio máximo y mínimo del día (high - low)
La variable objetivo (target) es el precio de cierre del día siguiente, lo que convierte al problema en una predicción de serie temporal con horizonte de un paso adelante.

3. Análisis Exploratorio de Datos (EDA)
Se realizó un análisis exploratorio exhaustivo para comprender la naturaleza de los datos y validar supuestos. Las principales actividades incluyeron:

Visualización temporal: gráficos del precio de cierre, volumen y volatilidad a lo largo del tiempo.

Estadísticas descriptivas: media, desviación estándar, valores mínimos y máximos de cada variable.

Distribución de retornos: histograma y análisis de asimetría (skewness) y curtosis, revelando colas pesadas típicas de activos financieros.

Correlaciones: matriz de correlación para identificar relaciones entre variables y el target, destacando altas correlaciones lineales entre close, vwap, ma_7 y ma_21.

Estacionalidad: análisis de rangos diarios promedio por mes, mostrando mayor actividad en meses de reportes trimestrales.

Este análisis permitió:

Confirmar la ausencia de valores nulos tras el preprocesamiento.

Identificar las variables más relevantes para la predicción.

Detectar la linealidad de las relaciones, justificando el uso de modelos lineales.

4. Procesamiento de Datos
Para preparar los datos para el entrenamiento y la predicción, se aplicaron los siguientes pasos:

4.1 Limpieza de datos
Eliminación de la columna symbol (redundante).

Conversión de la columna timestamp a índice de fechas.

Verificación de valores nulos (ninguno en los datos crudos).

Eliminación de filas con valores nulos generados por los cálculos de medias móviles y volatilidad (al inicio del período).

4.2 Ingeniería de características
Se generaron las 5 variables derivadas descritas en la sección anterior, utilizando ventanas deslizantes (rolling) con ventanas de 7 y 21 días.

4.3 Escalado de datos
Todas las características de entrada se escalaron utilizando StandardScaler para que tengan media 0 y desviación estándar 1. Esto es crucial para modelos sensibles a la escala (como SVM, KNN, y regularización L1/L2). El scaler se ajustó únicamente con los datos de entrenamiento para evitar fugas de información.

4.4 División de datos
Se utilizó una división temporal, respetando el orden cronológico:

Entrenamiento: 80% de los datos (primeros 875 registros)

Prueba: 20% de los datos (últimos 219 registros)

No se utilizó validación cruzada aleatoria para preservar la estructura temporal de la serie.

5. Modelos Implementados
Se entrenaron 8 modelos de regresión, cubriendo tanto enfoques lineales como no lineales:

Modelo	Descripción
OLS	Regresión lineal ordinaria, base para comparación.
Ridge	Regresión lineal con regularización L2 (penaliza coeficientes grandes).
Bayesian Ridge	Enfoque probabilístico con distribuciones previas sobre los coeficientes.
Lasso	Regularización L1 que puede reducir coeficientes a cero (selección de características).
KNN	K‑Nearest Neighbors, promedio de los k vecinos más cercanos.
Random Forest	Conjunto de árboles de decisión, captura relaciones no lineales.
SVM	Support Vector Machine con kernel RBF.
MLP	Red neuronal multicapa con tres capas ocultas (128, 64, 32).
Hiperparámetros utilizados (seleccionados por validación empírica y valores por defecto ajustados):

OLS: fit_intercept=True

Ridge: alpha=1.0, max_iter=1000, random_state=42

Bayesian Ridge: max_iter=300, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6

Lasso: alpha=0.001, max_iter=10000, random_state=42

KNN: n_neighbors=5, weights="distance"

Random Forest: n_estimators=100, max_depth=10, min_samples_split=5, random_state=42

SVM: kernel="rbf", C=100, epsilon=0.1, gamma="scale"

MLP: hidden_layer_sizes=(128,64,32), activation="relu", max_iter=500, early_stopping=True

Todos los modelos se entrenaron con los datos de entrenamiento escalados.

6. Evaluación y Resultados
Se evaluaron los modelos con los datos de prueba utilizando las siguientes métricas:

MAE (Mean Absolute Error): error absoluto promedio.

MSE (Mean Squared Error): error cuadrático promedio.

RMSE (Root Mean Squared Error): raíz del MSE, en las mismas unidades que la variable objetivo.

R² (Coeficiente de determinación): proporción de la varianza explicada por el modelo.

Modelo	MAE	MSE	RMSE	R²
Lasso	0.6518	0.9687	0.9842	0.9845
OLS	0.6566	0.9828	0.9914	0.9843
Bayesian Ridge	0.6595	0.9875	0.9937	0.9842
Ridge	0.6732	1.0093	1.0046	0.9839
Random Forest	2.0990	8.5016	2.9157	0.8642
MLP	1.6350	8.9926	2.9988	0.8563
KNN	3.2526	20.0904	4.4822	0.6791
SVM	3.2029	33.4580	5.7843	0.4655
Análisis:

Los modelos lineales con regularización (Lasso, Ridge, Bayesian Ridge) superan ampliamente a los modelos no lineales en este conjunto de datos.

Lasso es el mejor modelo: obtiene el R² más alto (0.9845) y los errores más bajos. Además, su regularización L1 ayuda a seleccionar características relevantes.

La alta linealidad de las relaciones (evidenciada en las gráficas de correlación) explica por qué los modelos lineales funcionan tan bien.

Los modelos no lineales (Random Forest, MLP, SVM) tienen un rendimiento inferior, posiblemente debido al sobreajuste o a la naturaleza predominantemente lineal de los datos.

7. Sistema de Predicción Final
Se desarrolló un script independiente (Prediction-System.py) que permite:

Cargar un archivo CSV con nuevos datos históricos de Intel.

Aplicar el mismo preprocesamiento y feature engineering que se usó durante el entrenamiento.

Escalar los datos con el StandardScaler entrenado.

Utilizar el modelo Lasso (el mejor) para generar predicciones del precio de cierre del día siguiente.

Guardar los resultados en un archivo CSV.

7.1 Estructura del sistema
El sistema consta de las siguientes funciones modulares:

cargar_modelo(ruta_scaler, ruta_modelo): carga los archivos pickle del scaler y del modelo.

aplicar_feature_engineering(df): calcula las mismas variables derivadas (daily_return, ma_7, ma_21, volatility_7, daily_range).

procesar_csv(ruta_csv, scaler): lee el CSV, detecta si ya tiene las 12 columnas completas o solo las 7 básicas, aplica feature engineering si es necesario, elimina NaN y escala.

predecir_y_guardar(ruta_csv, scaler, modelo, ruta_salida): orquesta el proceso y guarda el resultado.

7.2 Uso
Requisitos previos:

Python 3.8 o superior.

Librerías: numpy, pandas, scikit-learn, joblib.

Archivos necesarios: models/scaler.pkl y models/lasso.pkl.

Ejecución básica (usa archivos por defecto):

bash
python Prediction-System.py
Ejecución con argumentos personalizados:

bash
python Prediction-System.py --csv mis_datos.csv --salida mis_predicciones.csv
Argumentos disponibles:

--csv: ruta del archivo CSV de entrada (default: data/intc_processed.csv)

--scaler: ruta del scaler entrenado (default: models/scaler.pkl)

--modelo: ruta del modelo entrenado (default: models/lasso.pkl)

--salida: archivo de salida para las predicciones (default: predicciones.csv)

7.3 Formato del CSV de entrada
El sistema acepta dos formatos:

Formato básico (el que se obtiene directamente de la API):

Columnas: fecha, open, high, low, close, volume, trade_count, vwap

Formato pre‑procesado (el que se genera en el notebook después del feature engineering):

Columnas: las 7 básicas más daily_return, ma_7, ma_21, volatility_7, daily_range

El script detecta automáticamente qué columnas contiene y actúa en consecuencia.

7.4 Salida
Genera un archivo CSV con dos columnas:

fecha: la fecha original de cada registro.

prediccion: el precio estimado para el día siguiente (target = close del día siguiente).

8. Organización del Notebook
El notebook intel_prediction.ipynb está estructurado en secciones claras:

Instalación de librerías y carga de datos (conexión a Alpaca).

Preprocesamiento y feature engineering (generación de variables derivadas, normalización).

Análisis Exploratorio de Datos (EDA) con 8 gráficas analíticas.

División de datos (entrenamiento/prueba temporal).

Entrenamiento de los 8 modelos con sus hiperparámetros.

Evaluación y comparación (métricas, ranking).

Visualización de resultados (gráficos de errores, comparación real vs predicho, importancia de características).

Guardado de modelos y resultados.

Se utilizaron comentarios y separadores en Markdown para mejorar la legibilidad.

9. Conclusiones
Se logró desarrollar un sistema completo de predicción de precios basado en datos históricos de Intel, cumpliendo con todos los requisitos del proyecto.

El modelo Lasso demostró ser el más efectivo, con un R² de 0.9845 y errores muy bajos, superando a otros modelos lineales y no lineales.

La ingeniería de características (medias móviles, volatilidad, rango diario) fue clave para capturar la dinámica del activo.

El sistema de predicción es robusto, flexible y está listo para ser utilizado con nuevos datos, manteniendo consistencia en el preprocesamiento.

10. Trabajo Futuro
Explorar la inclusión de variables macroeconómicas o de sentimiento de mercado para mejorar la predicción.

Implementar validación cruzada con ventanas deslizantes para evaluar la estabilidad temporal del modelo.

Desplegar el sistema como un servicio web (API) para consumo en tiempo real.
