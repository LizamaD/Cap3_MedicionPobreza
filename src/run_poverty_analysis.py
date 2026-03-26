# -*- coding: utf-8 -*-
"""
Análisis de Dimensiones de Pobreza con Autoencoder y SHAP

Este script carga un autoencoder pre-entrenado y un escalador para analizar
las características de hogares en situación de pobreza.

Metodología:
1.  Cargar artefactos del modelo (autoencoder y scaler).
2.  Procesar los datasets de hogares pobres y no pobres.
3.  Calcular el error de reconstrucción para el set de hogares pobres.
4.  Utilizar SHAP (GradientExplainer) para explicar el error de reconstrucción.
5.  Calcular la importancia global ponderada de las variables usando el factor
    de expansión de la encuesta (representatividad nacional).
6.  Visualizar los resultados, excluyendo variables tradicionales, para
    descubrir nuevas dimensiones de la pobreza.
"""

# =============================================================================
# 1. Configuración del Entorno
# =============================================================================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import shap
import matplotlib.pyplot as plt
import os

print("--- Iniciando Análisis de Pobreza con Autoencoder y SHAP ---")
print(f"Versión de TensorFlow: {tf.__version__}")
print(f"Versión de SHAP: {shap.__version__}")

# =============================================================================
# 2. Carga de Datos y Modelos
# =============================================================================
DATA_DIR = 'data/processed/'
MODEL_DIR = 'results/'
OUTPUT_DIR = 'results/' # Directorio para guardar los gráficos

print("\n--- Cargando artefactos del modelo y datos ---")

try:
    # Cargar el modelo y el escalador
    autoencoder = load_model(os.path.join(MODEL_DIR, 'autoencoder_final.keras'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_final.joblib'))
    print("Modelo y escalador cargados exitosamente.")

    # Cargar los datos
    df_no_pobres = pd.read_csv(os.path.join(DATA_DIR, 'no_pobres.csv'))
    df_pobres = pd.read_csv(os.path.join(DATA_DIR, 'pobres.csv'))
    print(f"Datos de no pobres cargados: {df_no_pobres.shape}")
    print(f"Datos de pobres cargados: {df_pobres.shape}")

except Exception as e:
    print(f"Error crítico al cargar archivos: {e}")
    print("Asegúrate de que 'autoencoder_final.keras', 'scaler_final.joblib', 'no_pobres.csv' y 'pobres.csv' existan en sus directorios correctos.")
    exit()

# =============================================================================
# 3. Preprocesamiento de Datos
# =============================================================================
print("\n--- Preprocesando datos para la inferencia ---")

def preprocess_data_for_inference(df, scaler):
    """Prepara el DataFrame para la inferencia del autoencoder."""
    if 'factor' not in df.columns:
        raise ValueError("La columna 'factor' no se encontró en el DataFrame.")
    
    factor = df['factor'].copy()
    feature_cols = scaler.feature_names_in_
    
    # Asegurarse de que todas las columnas de features están presentes
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan las siguientes columnas en el DataFrame: {missing_cols}")

    df_features = df[feature_cols]
    X_scaled = scaler.transform(df_features)
    
    return X_scaled, factor, feature_cols

X_no_pobres_scaled, _, feature_names = preprocess_data_for_inference(df_no_pobres, scaler)
X_pobres_scaled, factor_pobres, _ = preprocess_data_for_inference(df_pobres, scaler)

print("Datos procesados y escalados correctamente.")

# =============================================================================
# 4. Explicabilidad con SHAP para Error de Reconstrucción
# =============================================================================
print("\n--- Configurando el explainer de SHAP ---")

# Usar una muestra de los datos de no pobres como background
background_sample = shap.sample(X_no_pobres_scaled, 100)

# Inicializar el explainer de SHAP con el modelo y el background
explainer = shap.GradientExplainer(autoencoder, background_sample)

print("Calculando valores SHAP para el dataset de pobres. Esto puede tardar...")
# Calcular los valores SHAP sobre el dataset de pobres
try:
    shap_values = explainer.shap_values(X_pobres_scaled)
    shap_values_pobres = shap_values[0]
    print("Cálculo de SHAP finalizado.")
except Exception as e:
    print(f"Error durante el cálculo de SHAP: {e}")
    exit()

# =============================================================================
# 5. Análisis del Error y Visualización de SHAP Summary Plot
# =============================================================================
print("\n--- Analizando el error de reconstrucción y generando SHAP Summary Plot ---")

# Calcular el error de reconstrucción para cada hogar pobre
reconstruction_error_pobres = np.mean((X_pobres_scaled - autoencoder.predict(X_pobres_scaled))**2, axis=1)
df_pobres['reconstruction_error'] = reconstruction_error_pobres

# Identificar los hogares con mayor error
top_error_indices = df_pobres['reconstruction_error'].nlargest(200).index

# Crear un objeto de explicación de SHAP para facilitar los gráficos
shap_explanation = shap.Explanation(
    values=shap_values_pobres[top_error_indices],
    base_values=explainer.expected_value[0],
    data=X_pobres_scaled[top_error_indices],
    feature_names=feature_names
)

# Generar y guardar el SHAP summary plot
plt.figure()
shap.summary_plot(shap_explanation, max_display=15, show=False)
plt.title("SHAP Summary Plot para Hogares con Mayor Error")
summary_plot_path = os.path.join(OUTPUT_DIR, 'shap_summary_plot_top_error.png')
plt.savefig(summary_plot_path, bbox_inches='tight')
plt.close()
print(f"SHAP summary plot guardado en: {summary_plot_path}")

# =============================================================================
# 6. Análisis de Impacto Global Ponderado
# =============================================================================
print("\n--- Calculando impacto global ponderado de las variables ---")

# Ponderar cada valor SHAP absoluto por el factor de expansión
abs_shap_values = np.abs(shap_values_pobres)
weighted_shap_values = abs_shap_values * factor_pobres.to_numpy()[:, np.newaxis]

# Calcular la importancia global ponderada y normalizar
global_weighted_impact = pd.Series(np.sum(weighted_shap_values, axis=0), index=feature_names)
total_factor_sum = factor_pobres.sum()
mean_global_weighted_impact = global_weighted_impact / total_factor_sum
mean_global_weighted_impact.sort_values(ascending=False, inplace=True)

# =============================================================================
# 7. Exclusión de Variables Tradicionales y Visualización Final
# =============================================================================
print("\n--- Identificando y visualizando nuevas dimensiones de pobreza ---")

def filter_coneval_vars(impact_series):
    """Filtra variables del índice de pobreza de CONEVAL."""
    coneval_keywords = [
        'ing_', 'gasto_', 'carencia', 'rezago_educ', 'acceso_salud', 
        'seguridad_social', 'alim', 'serv_agua', 'serv_dren', 'serv_luz',
        'pared', 'techo', 'piso', 'combustible'
    ]
    mask = ~impact_series.index.str.contains('|'.join(coneval_keywords), case=False)
    return impact_series[mask]

non_coneval_impact = filter_coneval_vars(mean_global_weighted_impact)
top_15_new_dimensions = non_coneval_impact.head(15)

print("\nTop 15 nuevas dimensiones de pobreza encontradas:")
print(top_15_new_dimensions)

# Graficar los resultados
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 10))
top_15_new_dimensions.sort_values(ascending=True).plot(kind='barh', ax=ax, color='skyblue')
ax.set_title('Top 15 Nuevas Dimensiones de Pobreza (Impacto Global Ponderado)', fontsize=16)
ax.set_xlabel('Impacto SHAP Absoluto Promedio Ponderado', fontsize=12)
ax.set_ylabel('Variables', fontsize=12)
plt.tight_layout()

# Guardar el gráfico final
barplot_path = os.path.join(OUTPUT_DIR, 'top_15_nuevas_dimensiones_pobreza.png')
plt.savefig(barplot_path)
plt.close()
print(f"\nGráfico de barras con las nuevas dimensiones guardado en: {barplot_path}")

print("\n--- Análisis finalizado con éxito ---")

# =============================================================================
# 8. Interpretación de Resultados (para referencia)
# =============================================================================
# ### Análisis del Gráfico de Barras
# El gráfico 'top_15_nuevas_dimensiones_pobreza.png' muestra las 15 variables que, 
# según el modelo, más contribuyen al error de reconstrucción de los hogares pobres, 
# después de excluir las dimensiones tradicionales de la pobreza. Un alto impacto 
# significa que estas características en los hogares pobres son muy diferentes a las 
# de los hogares no pobres a nivel nacional.
#
# ### Posibles Hallazgos a Buscar en el Gráfico:
# 1.  **Brecha Digital:** Variables como `tiene_internet`, `tiene_computadora`, `gasto_com`.
# 2.  **Uso del Tiempo y Carga de Cuidados:** `htrab` (horas trabajadas), `trabajo_domestico_hrs`.
# 3.  **Salud y Bienestar más allá del Acceso:** `gasto_medicamentos_sin_receta`, `gasto_atencion_ambulatoria`.
# 4.  **Activos del Hogar y Resiliencia:** `tiene_refri`, `tiene_lavadora`, `pago_deudas`.
# 5.  **Dinámicas Laborales Precarias:** `contrato_temporal`, `sin_prestaciones`, `tam_emp_micro`.
#
# Estos hallazgos son candidatos a ser **dimensiones emergentes de la pobreza**.
# =============================================================================
