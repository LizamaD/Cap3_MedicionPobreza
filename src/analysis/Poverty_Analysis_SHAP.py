# -*- coding: utf-8 -*-
"""
Este script es una conversión del notebook 'Analisis_Pobreza_Autoencoder_SHAP.ipynb'.
Cada sección está marcada con comentarios para que puedas identificar las celdas originales.
"""

# =============================================================================
# CELL 1: MARKDOWN
# =============================================================================
# # Análisis de Dimensiones de Pobreza con Autoencoders y SHAP
#
# **Objetivo:** Utilizar un autoencoder entrenado en hogares no pobres para identificar y explicar las características que más contribuyen a que un hogar sea clasificado como pobre. Usaremos el error de reconstrucción y los valores SHAP para descubrir dimensiones de pobreza no capturadas por el índice oficial del CONEVAL.
#
# **Metodología:**
# 1.  **Cargar Artefactos:** Importar el modelo autoencoder y el escalador de datos.
# 2.  **Procesar Datos:** Cargar los datasets de hogares pobres y no pobres, aplicando el escalado correspondiente.
# 3.  **Inferencia y Error:** Calcular el error de reconstrucción para el set de hogares pobres.
# 4.  **Explicabilidad con SHAP:** Usar `GradientExplainer` para calcular la contribución de cada variable al error de reconstrucción.
# 5.  **Análisis Ponderado:** Calcular la importancia global de las variables utilizando el factor de expansión de la encuesta para obtener resultados representativos a nivel nacional.
# 6.  **Visualización e Interpretación:** Graficar los resultados y analizar las variables más relevantes que no forman parte del índice de pobreza multidimensional tradicional.

# =============================================================================
# CELL 2: MARKDOWN
# =============================================================================
# ## 1. Configuración del Entorno
#
# Primero, instalamos y cargamos las librerías necesarias.

# =============================================================================
# CELL 3: CODE
# =============================================================================
# Descomenta la siguiente línea si no tienes 'shap' instalado
# !pip install shap

# =============================================================================
# CELL 4: CODE
# =============================================================================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow version:", tf.__version__)
print("SHAP version:", shap.__version__)

# =============================================================================
# CELL 5: MARKDOWN
# =============================================================================
# ## 2. Carga de Datos y Modelos
#
# Cargamos el autoencoder, el escalador y los datasets. Asegúrate de que las rutas a los archivos sean correctas. Para este ejemplo, asumimos que los archivos están en un directorio `data/processed` y el modelo y scaler en `results`.

# =============================================================================
# CELL 6: CODE
# =============================================================================
# Define las rutas a tus archivos
DATA_DIR = 'data/processed/'
MODEL_DIR = 'results/'

# Cargar el modelo y el escalador
try:
    autoencoder = load_model(f'{MODEL_DIR}autoencoder_final.keras')
    scaler = joblib.load(f'{MODEL_DIR}scaler_final.joblib')
    print("Modelo y escalador cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar los artefactos: {e}")
    print("Asegúrate de haber entrenado y guardado el modelo y el scaler previamente.")

# Cargar los datos
try:
    df_no_pobres = pd.read_csv(f'{DATA_DIR}no_pobres.csv')
    df_pobres = pd.read_csv(f'{DATA_DIR}pobres.csv')
    print(f"Datos de no pobres cargados: {df_no_pobres.shape}")
    print(f"Datos de pobres cargados: {df_pobres.shape}")
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo de datos: {e}")
    print("Por favor, verifica que los archivos 'no_pobres.csv' y 'pobres.csv' estén en el directorio correcto.")

# =============================================================================
# CELL 7: MARKDOWN
# =============================================================================
# ## 3. Preprocesamiento de Datos
#
# Preparamos los datos para la inferencia. Esto incluye:
# 1.  Separar la columna `factor` para el análisis ponderado.
# 2.  Eliminar las columnas de identificación (`folioviv`, `foliohog`, `numren`) y las etiquetas de pobreza (`pobreza`, `pobreza_e`).
# 3.  Aplicar el escalador `MinMaxScaler` que fue ajustado con los datos de **no pobres**.

# =============================================================================
# CELL 8: CODE
# =============================================================================
def preprocess_data_for_inference(df, scaler):
    """Prepara el DataFrame para la inferencia del autoencoder."""
    # Columnas a eliminar
    id_cols = ['folioviv', 'foliohog', 'numren', 'pobreza', 'pobreza_e']
    
    # Extraer el factor de expansión y guardarlo
    factor = df['factor'].copy()
    
    # Seleccionar solo las columnas de features que existen en el scaler
    # El scaler fue ajustado con el df de no pobres, así que sus features son la referencia
    feature_cols = scaler.feature_names_in_
    df_features = df[feature_cols]
    
    # Escalar los datos
    X_scaled = scaler.transform(df_features)
    
    return X_scaled, factor, feature_cols

# Procesar ambos datasets
X_no_pobres_scaled, factor_no_pobres, feature_names = preprocess_data_for_inference(df_no_pobres, scaler)
X_pobres_scaled, factor_pobres, _ = preprocess_data_for_inference(df_pobres, scaler)

print("Dimensiones del set de 'no pobres' escalado:", X_no_pobres_scaled.shape)
print("Dimensiones del set de 'pobres' escalado:", X_pobres_scaled.shape)

# =============================================================================
# CELL 9: MARKDOWN
# =============================================================================
# ## 4. Explicabilidad con SHAP para Error de Reconstrucción
#
# Para que SHAP pueda explicar el **error** del modelo, no su salida directa, necesitamos un enfoque especial. Definimos una nueva función que SHAP interpretará como "la salida del modelo". Esta función calculará la diferencia cuadrática media (MSE) entre la entrada y la salida del autoencoder.

# =============================================================================
# CELL 10: CODE
# =============================================================================
# El explainer de gradiente necesita un "background dataset" para calcular las expectativas.
# Usar una muestra de los datos de no pobres es lo correcto, ya que el modelo aprendió de ellos.
background_sample = shap.sample(X_no_pobres_scaled, 100) # 100 muestras suelen ser suficientes

# 1. Definir la función de error que SHAP explicará
def model_reconstruction_error(x):
    """Calcula el error de reconstrucción (MSE) para una entrada x."""
    reconstruction = autoencoder.predict(x)
    # Calculamos el error cuadrático por cada feature y luego lo promediamos.
    # SHAP necesita una única salida por cada entrada, por eso el promedio.
    return np.mean((x - reconstruction)**2, axis=1)

# 2. Inicializar el explainer de SHAP
# Usamos GradientExplainer, que es eficiente para modelos de redes neuronales.
explainer = shap.GradientExplainer(autoencoder, background_sample)

# 3. Calcular los valores SHAP sobre el dataset de pobres
# Esto puede tomar tiempo dependiendo del tamaño del dataset
print("Calculando valores SHAP para el dataset de pobres. Esto puede tardar...")
shap_values = explainer.shap_values(X_pobres_scaled)

print("Cálculo de SHAP finalizado.")
# SHAP devuelve una lista, en este caso con un solo elemento. Lo extraemos.
shap_values_pobres = shap_values[0]

# =============================================================================
# CELL 11: MARKDOWN
# =============================================================================
# ## 5. Análisis del Error y Visualización
#
# Ahora que tenemos los valores SHAP, podemos investigar qué variables y qué hogares tienen el mayor impacto en el error de reconstrucción.

# =============================================================================
# CELL 12: CODE
# =============================================================================
# Calcular el error de reconstrucción para cada hogar pobre
reconstruction_error_pobres = np.mean((X_pobres_scaled - autoencoder.predict(X_pobres_scaled))**2, axis=1)
df_pobres['reconstruction_error'] = reconstruction_error_pobres

# Identificar los hogares con mayor error
top_error_indices = df_pobres['reconstruction_error'].nlargest(200).index

print("Visualizando el SHAP summary plot para los 200 hogares con mayor error de reconstrucción:")

# Crear un objeto de explicación de SHAP para facilitar los gráficos
shap_explanation = shap.Explanation(
    values=shap_values_pobres[top_error_indices],
    base_values=None, # No aplica para GradientExplainer
    data=X_pobres_scaled[top_error_indices],
    feature_names=feature_names
)

shap.summary_plot(shap_explanation, max_display=15)

# =============================================================================
# CELL 13: MARKDOWN
# =============================================================================
# ### Análisis de Impacto Global Ponderado
#
# Para que nuestros hallazgos sean representativos de toda la población de México, debemos ponderar la importancia de cada variable (su valor SHAP absoluto) por el `factor` de expansión de cada hogar. Esto nos dará una medida del **impacto agregado a nivel nacional**.

# =============================================================================
# CELL 14: CODE
# =============================================================================
# Calcular el valor SHAP absoluto
abs_shap_values = np.abs(shap_values_pobres)

# Ponderar cada valor SHAP por el factor de expansión
# Usamos np.newaxis para alinear las dimensiones para la multiplicación
weighted_shap_values = abs_shap_values * factor_pobres.to_numpy()[:, np.newaxis]

# Calcular la importancia global ponderada sumando los valores SHAP ponderados por cada variable
global_weighted_impact = pd.Series(np.sum(weighted_shap_values, axis=0), index=feature_names)

# Normalizar para que sea más fácil de interpretar (opcional, pero recomendado)
total_factor_sum = factor_pobres.sum()
mean_global_weighted_impact = global_weighted_impact / total_factor_sum

mean_global_weighted_impact.sort_values(ascending=False, inplace=True)

# =============================================================================
# CELL 15: MARKDOWN
# =============================================================================
# ### Exclusión de Variables Tradicionales (CONEVAL)
#
# El objetivo es encontrar **nuevas** dimensiones de la pobreza. Por lo tanto, filtraremos las variables que ya son parte explícita de la medición multidimensional de la pobreza en México para evitar redundancia en el análisis.
#
# Excluimos variables relacionadas directamente con:
# - Ingreso (`ing_`, `gasto_`)
# - Carencias sociales explícitas (`carencia_`, `rezago_educ`)
# - Acceso a servicios básicos (`serv_`, `acceso_`)
# - Características de la vivienda (`pared`, `techo`, `piso`)

# =============================================================================
# CELL 16: CODE
# =============================================================================
def filter_coneval_vars(impact_series):
    """Filtra las variables que ya son parte del índice de pobreza de CONEVAL."""
    coneval_keywords = [
        'ing_', 'gasto_', 'carencia', 'rezago_educ', 'acceso_salud', 
        'seguridad_social', 'alim', 'serv_agua', 'serv_dren', 'serv_luz',
        'pared', 'techo', 'piso', 'combustible'
    ]
    
    # Crear una máscara booleana
    mask = impact_series.index.to_series().apply(
        lambda x: not any(keyword in x for keyword in coneval_keywords)
    )
    
    return impact_series[mask]

non_coneval_impact = filter_coneval_vars(mean_global_weighted_impact)

top_15_new_dimensions = non_coneval_impact.head(15)

# Graficar los resultados
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 10))
top_15_new_dimensions.sort_values(ascending=True).plot(kind='barh', ax=ax, color='skyblue')
ax.set_title('Top 15 Nuevas Dimensiones de Pobreza (Impacto Global Ponderado)', fontsize=16)
ax.set_xlabel('Impacto SHAP Absoluto Promedio Ponderado', fontsize=12)
ax.set_ylabel('Variables', fontsize=12)
plt.tight_layout()
plt.show()

# =============================================================================
# CELL 17: MARKDOWN
# =============================================================================
# ## 6. Interpretación de Resultados
#
# **(Esta sección es para ser completada por el analista después de ejecutar el script)**
#
# ### Análisis del Gráfico de Barras
# El gráfico anterior muestra las 15 variables que, según el modelo, más contribuyen al error de reconstrucción de los hogares pobres, después de excluir las dimensiones tradicionales de la pobreza. Un alto impacto significa que estas características en los hogares pobres son muy diferentes a las de los hogares no pobres.
#
# **Posibles Hallazgos a Buscar:**
#
# 1.  **Brecha Digital:** Si aparecen variables como `tiene_internet`, `tiene_computadora`, o gastos en `comunicacion`, podría ser un fuerte indicador de que la falta de acceso digital es una dimensión moderna y crucial de la exclusión social, impactando el acceso a la información, educación y oportunidades laborales.
#
# 2.  **Uso del Tiempo y Carga de Cuidados:** Variables como `htrab` (horas trabajadas), `trabajo_domestico_hrs` o la presencia de personas que no estudian ni trabajan (`nini`) podrían sugerir que la **pobreza de tiempo** es un factor limitante. Un número excesivo de horas trabajadas por un bajo ingreso o una gran carga de trabajo no remunerado pueden ser barreras significativas para el bienestar.
#
# 3.  **Salud y Bienestar más allá del Acceso:** En lugar de solo el acceso a servicios de salud, quizás el modelo resalta gastos específicos como `gasto_medicamentos_sin_receta` o `gasto_atencion_ambulatoria`. Esto podría indicar que, aunque tengan acceso formal, las familias están haciendo frente a problemas de salud con gastos de bolsillo que merman su capacidad económica.
#
# 4.  **Activos del Hogar y Resiliencia:** La ausencia de ciertos bienes (`tiene_refri`, `tiene_lavadora`) o, por el contrario, la presencia de deudas (`pago_deudas`) pueden ser indicativos de una falta de resiliencia económica. El modelo podría estar detectando que la incapacidad de construir un patrimonio básico o el sobreendeudamiento son características clave de la pobreza persistente.
#
# 5.  **Dinámicas Laborales Precarias:** Variables como `contrato_temporal`, `sin_prestaciones`, o `tam_emp_micro` (trabajo en microempresas) pueden ser más determinantes que simplemente estar empleado. El modelo podría estar señalando que la **calidad del empleo**, y no solo su existencia, es una dimensión fundamental de la pobreza.
#
# ### Conclusión Preliminar
#
# El análisis de SHAP sobre el error de reconstrucción del autoencoder nos permite ir más allá de las carencias oficiales. Las variables destacadas en el gráfico son candidatas a ser consideradas **dimensiones emergentes de la pobreza** que reflejan las complejidades de la vida moderna. Estos hallazgos podrían servir como base para diseñar políticas públicas más integrales que no solo se enfoquen en el ingreso o las carencias básicas, sino también en la inclusión digital, la calidad del empleo y la reducción de la carga de cuidados.

