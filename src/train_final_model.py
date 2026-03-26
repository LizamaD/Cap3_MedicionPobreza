import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def create_final_autoencoder(params, input_shape):
    """
    Construye el modelo de autoencoder final usando los hiperparámetros ganadores.
    Esta función es una adaptación de la usada en el tuning, pero lee los
    parámetros desde un diccionario en lugar de un 'trial' de Optuna.
    """
    # --- Extraer hiperparámetros del diccionario ---
    bottleneck_dim = params['bottleneck_dim']
    n_layers = params['n_layers']
    lr = params['lr']
    dropout_rate = params['dropout']
    
    # Extraer las unidades de cada capa
    layer_units = [params[f'units_layer_{i}'] for i in range(n_layers)]
        
    # --- Construcción del Modelo ---
    input_layer = layers.Input(shape=(input_shape,), name='input_layer')
    x = input_layer

    # Encoder
    for i in range(n_layers):
        x = layers.Dense(layer_units[i], activation='relu', name=f'encoder_dense_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'encoder_dropout_{i}')(x)

    # Bottleneck (Espacio latente)
    bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='latent_space')(x)

    # Decoder (Espejo del encoder)
    x = bottleneck
    for i in reversed(range(n_layers)):
        x = layers.Dense(layer_units[i], activation='relu', name=f'decoder_dense_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'decoder_dropout_{i}')(x)

    output_layer = layers.Dense(input_shape, activation='sigmoid', name='output_layer')(x)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    return autoencoder

if __name__ == "__main__":
    # --- 1. Definir Hiperparámetros Ganadores y Rutas ---
    print("Configurando el entrenamiento final...")
    
    # Estos son los parámetros del mejor trial que encontraste
    BEST_PARAMS = {
        'bottleneck_dim': 15,
        'n_layers': 3,
        'lr': 0.000576,
        'dropout': 0.032113,
        'units_layer_0': 432,
        'units_layer_1': 387,
        'units_layer_2': 362
        # units_layer_3 y 4 no se usan porque n_layers es 3
    }

    BASE_DIR = "data/processed/"
    MODEL_SAVE_DIR = "results/"

    # --- 2. Carga y Preparación de Datos Completos ---
    print("Cargando el dataset completo de 'no pobres'...")
    df_no_pobres = pd.read_csv(os.path.join(BASE_DIR, "no_pobres.csv"))
    print(f"Datos cargados: {df_no_pobres.shape[0]} registros.")

    # Columnas que no son features
    non_features = ['folioviv', 'foliohog', 'numren', 'pobreza', 'pobreza_e']
    
    # El factor de expansión se usa como peso en el entrenamiento
    weights = df_no_pobres['factor']
    
    # Seleccionar solo las columnas de features
    features_df = df_no_pobres.drop(columns=non_features + ['factor'], errors='ignore')
    
    # --- 3. Escalado de Datos ---
    # Se escala el conjunto de datos completo de 'no pobres'
    print("Escalando todos los datos de entrenamiento...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    # --- 4. Creación y Entrenamiento del Modelo Final ---
    print("Creando el modelo final con los mejores hiperparámetros...")
    final_autoencoder = create_final_autoencoder(BEST_PARAMS, X_scaled.shape[1])
    
    print("Resumen del modelo final:")
    final_autoencoder.summary()

    print("\nIniciando entrenamiento del modelo final...")
    # Entrenamos con todos los datos, sin set de validación.
    # Usamos un número fijo de epochs. 100 es un buen punto de partida.
    final_autoencoder.fit(
        X_scaled, X_scaled,
        sample_weight=weights.to_numpy(),
        epochs=100,
        batch_size=512,
        verbose=2,
        shuffle=True
    )

    # --- 5. Guardar el Modelo y el Scaler ---
    print("\nEntrenamiento completado. Guardando artefactos del modelo...")

    # Crear el directorio si no existe
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Guardar el modelo de Keras
    model_path = os.path.join(MODEL_SAVE_DIR, "autoencoder_final.keras")
    final_autoencoder.save(model_path)
    print(f"  - Modelo guardado en: {model_path}")

    # Guardar el scaler de Scikit-learn
    # Es CRUCIAL para aplicar la misma transformación a los datos de 'pobres'
    scaler_path = os.path.join(MODEL_SAVE_DIR, "scaler_final.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  - Scaler guardado en: {scaler_path}")

    print("\n¡Proceso finalizado con éxito!")