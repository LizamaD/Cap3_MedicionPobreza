import os
import sys

# --- 1. CONFIGURACIONES CRÍTICAS (Ejecutar antes de cualquier import) ---
# Evita el bloqueo del kernel en macOS por conflicto de librerías SSL/KMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Fuerza el uso de CPU para evitar errores de drivers CUDA/Metal en la carga
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- 2. IMPORTACIÓN DE LIBRERÍAS ---
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- 3. CONFIGURACIÓN DE RUTAS ---
# Rutas locales (Mac) hacia tus datos de CONEVAL y modelos
BASE_DIR_MODELOS = '/Users/davidlizama/Documents/modelos_tesis/Cap3_MedicionPobreza/results/'
BASE_DIR_DATOS = '/Users/davidlizama/Documents/modelos_tesis/Cap3_MedicionPobreza/data/'

PATH_MODEL = os.path.join(BASE_DIR_MODELOS, 'autoencoder_final.keras')
PATH_SCALER = os.path.join(BASE_DIR_MODELOS, 'scaler_final.joblib')
PATH_POBRES = os.path.join(BASE_DIR_DATOS, 'pobres.csv')
PATH_NO_POBRES = os.path.join(BASE_DIR_DATOS, 'no_pobres.csv')
TXT_OUTPUT = 'resultados_tesis.txt'

def preprocess_data_for_inference(df, scaler):
    """
    Prepara un DataFrame para la inferencia con el autoencoder asegurando
    la consistencia con las variables de entrenamiento.
    """
    feature_cols = scaler.feature_names_in_
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan las siguientes columnas en el DataFrame: {missing_cols}")
    
    factor = df['factor'].copy()
    df_features = df[feature_cols]
    
    # Aplicar la transformación (NO fit_transform)
    X_scaled = scaler.transform(df_features)
    
    return X_scaled, factor, feature_cols

def filter_coneval_vars(impact_series):
    """Filtra variables tradicionales del índice de pobreza de CONEVAL."""
    coneval_keywords = [
        'ing_', 'gasto_', 'carencia', 'rezago_educ', 'acceso_salud', 
        'seguridad_social', 'alim', 'serv_agua', 'serv_dren', 'serv_luz',
        'pared', 'techo', 'piso', 'combustible'
    ]
    mask = ~impact_series.index.str.contains('|'.join(coneval_keywords), case=False)
    return impact_series[mask]

def main():
    # Redirigir la salida estándar a un archivo de texto
    original_stdout = sys.stdout
    with open(TXT_OUTPUT, 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        print("="*50)
        print("INICIANDO PIPELINE DE INFERENCIA - MEDICIÓN DE POBREZA")
        print("="*50)

        # --- FASE 1: CARGA DE ARTEFACTOS ---
        print("\n[1/4] Cargando modelo y escalador...")
        if not os.path.exists(PATH_MODEL):
            raise FileNotFoundError(f"No se encontró el modelo en: {PATH_MODEL}")
        if not os.path.exists(PATH_SCALER):
            raise FileNotFoundError(f"No se encontró el escalador en: {PATH_SCALER}")

        autoencoder = load_model(PATH_MODEL, compile=False)
        lr_optimo = 0.000576
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_optimo), loss='mse')
        scaler = joblib.load(PATH_SCALER)
        
        print(f"Modelo cargado exitosamente. {len(autoencoder.layers)} capas detectadas.")
        print(f"Escalador cargado. Esperando {len(scaler.feature_names_in_)} variables.")

        # --- FASE 2: CARGA Y PREPROCESAMIENTO DE DATOS ---
        print("\n[2/4] Cargando y preprocesando bases de datos de CONEVAL...")
        df_pobres = pd.read_csv(PATH_POBRES)
        df_no_pobres = pd.read_csv(PATH_NO_POBRES)
        
        X_pobres_scaled, factor_pobres, feature_names = preprocess_data_for_inference(df_pobres, scaler)
        X_no_pobres_scaled, factor_no_pobres, _ = preprocess_data_for_inference(df_no_pobres, scaler)

        # --- FASE 3: ERROR DE RECONSTRUCCIÓN ---
        print("\n[3/4] Generando reconstrucciones y calculando Error Cuadrático Medio (MSE)...")
        # Aquí sí pasamos todos los datos (la predicción es rápida)
        reconstructions = autoencoder.predict(X_pobres_scaled, verbose=0)
        mse_per_observation = np.mean((X_pobres_scaled - reconstructions)**2, axis=1)
        df_pobres['reconstruction_error'] = mse_per_observation
        
        print("\nResumen estadístico del Error de Reconstrucción (Hogares Pobres):")
        print(df_pobres['reconstruction_error'].describe())

        # Guardar gráfico de distribución
        plt.figure(figsize=(10, 6))
        sns.histplot(df_pobres['reconstruction_error'], bins=50, kde=True)
        plt.title('Distribución del Error de Reconstrucción (Pobres)')
        plt.xlabel('Error Cuadrático Medio (MSE)')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.savefig('distribucion_error.png')
        plt.close()

        # --- FASE 4: ANÁLISIS SHAP OPTIMIZADO (TURBO) ---
        print("\n[4/4] Inicializando análisis SHAP (Versión Optimizada por Muestreo)...")
        
        # 1. Background reducido para mayor velocidad (50 es el estándar recomendado para GradientExplainer)
        background_sample = shap.sample(X_no_pobres_scaled, 50)
        explainer = shap.GradientExplainer(autoencoder, background_sample)
        
        # 2. Tomar una muestra representativa de 1000 hogares pobres
        np.random.seed(42)  # Semilla para reproducibilidad
        indices_muestra = np.random.choice(range(len(X_pobres_scaled)), size=1000, replace=False)
        X_inferencia_shap = X_pobres_scaled[indices_muestra]
        factor_muestra = factor_pobres.iloc[indices_muestra]
        df_pobres_muestra = df_pobres.iloc[indices_muestra]

        print(f"Calculando valores SHAP para {len(X_inferencia_shap)} registros muestreados...")
        shap_values = explainer.shap_values(X_inferencia_shap)
        shap_values_pobres = shap_values[0]

        # Ajustar el objeto de explicación para la gráfica Summary
        # Graficamos el top de anomalías DENTRO de la muestra
        top_error_indices_muestra = df_pobres_muestra['reconstruction_error'].nlargest(200).index
        # Mapear los índices originales del dataframe a los índices posicionales de la matriz de la muestra
        posiciones_top_error = [list(df_pobres_muestra.index).index(idx) for idx in top_error_indices_muestra]
        
        shap_explanation = shap.Explanation(
            values=shap_values_pobres[posiciones_top_error],
            base_values=explainer.expected_value[0],
            data=X_inferencia_shap[posiciones_top_error],
            feature_names=feature_names
        )

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_explanation, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png')
        plt.close()

        # --- FASE 5: IMPACTO GLOBAL PONDERADO (NUEVAS DIMENSIONES) ---
        print("\n--- RESULTADOS: DIMENSIONES DE POBREZA ---")
        abs_shap_values = np.abs(shap_values_pobres)
        
        # OJO: Ponderar usando el factor de expansión exclusivo de la muestra
        weighted_shap_values = abs_shap_values * factor_muestra.to_numpy()[:, np.newaxis]
        
        global_weighted_impact = pd.Series(np.sum(weighted_shap_values, axis=0), index=feature_names)
        mean_global_weighted_impact = global_weighted_impact / factor_muestra.sum()
        mean_global_weighted_impact.sort_values(ascending=False, inplace=True)
        
        print("\nTop 10 variables con mayor impacto global ponderado:")
        print(mean_global_weighted_impact.head(10))

        # Filtrar variables tradicionales de CONEVAL
        non_coneval_impact = filter_coneval_vars(mean_global_weighted_impact)
        top_15_new_dimensions = non_coneval_impact.head(15)
        
        print("\nTop 15 'Nuevas Dimensiones' descubiertas por el modelo:")
        print(top_15_new_dimensions)

        fig, ax = plt.subplots(figsize=(12, 10))
        top_15_new_dimensions.sort_values(ascending=True).plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title('Top 15 Nuevas Dimensiones de Pobreza (Muestra Representativa)', fontsize=16)
        ax.set_xlabel('Impacto SHAP Absoluto Promedio Ponderado', fontsize=12)
        ax.set_ylabel('Variables', fontsize=12)
        plt.tight_layout()
        plt.savefig('nuevas_dimensiones.png')
        plt.close()

        print("\nProceso finalizado exitosamente. Gráficos generados como PNG.")

    # Restaurar la salida a la consola al terminar
    sys.stdout = original_stdout
    print(f"¡Listo! El análisis con la muestra terminó rápido. Resultados guardados en '{TXT_OUTPUT}'.")

if __name__ == "__main__":
    main()