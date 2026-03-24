import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

def create_autoencoder(trial, input_shape):
    """
    Construye dinámicamente un modelo de autoencoder basado en los hiperparámetros de Optuna.
    """
    # --- Espacio de búsqueda de hiperparámetros ---
    bottleneck_dim = trial.suggest_int('bottleneck_dim', 8, 64)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout', 0.0, 0.4)
    
    # --- Sugerir parámetros para un número máximo de capas ---
    # Esto estabiliza el espacio de búsqueda de Optuna. Al definir siempre
    # los parámetros para el máximo de capas posibles, evitamos errores cuando
    # `n_layers` cambia entre "trials", lo que podía generar un espacio de búsqueda inconsistente.
    layer_units = []
    max_layers = 3 # Debe coincidir con el máximo de 'n_layers'
    for i in range(max_layers):
        layer_units.append(trial.suggest_int(f'units_layer_{i}', 64, 256))
        
    # --- Construcción Dinámica del Modelo ---
    input_layer = layers.Input(shape=(input_shape,), name='input_layer')
    x = input_layer

    # Encoder dinámico
    # Usamos los parámetros pre-sugeridos, construyendo solo `n_layers`
    for i in range(n_layers):
        x = layers.Dense(layer_units[i], activation='relu', name=f'encoder_dense_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'encoder_dropout_{i}')(x)

    # Bottleneck (Espacio latente)
    bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='latent_space')(x)

    # Decoder dinámico (Espejo del encoder)
    x = bottleneck
    for i in reversed(range(n_layers)):
        x = layers.Dense(layer_units[i], activation='relu', name=f'decoder_dense_{i}')(x)
        # Mejora: Añadir dropout también en el decoder para tener simetría y mejor regularización.
        x = layers.Dropout(dropout_rate, name=f'decoder_dropout_{i}')(x)

    output_layer = layers.Dense(input_shape, activation='sigmoid', name='output_layer')(x)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    return autoencoder

def objective(trial, X_train, w_train, X_val, w_val):
    """
    Función objetivo que Optuna intentará minimizar.
    Entrena un autoencoder y devuelve la pérdida de validación.
    """
    # Limpiar la sesión de Keras para evitar fugas de memoria
    tf.keras.backend.clear_session()

    # Crear el modelo para este "trial"
    autoencoder = create_autoencoder(trial, X_train.shape[1])

    # Callback para detener el entrenamiento si no mejora
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # --- Entrenamiento con Pesos (Factor de Expansión) ---
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val, w_val), # Usar pesos en validación también
        sample_weight=w_train,
        epochs=100,  # Un número alto, EarlyStopping se encargará de parar
        batch_size=512,
        verbose=0,
        callbacks=[early_stopping]
    )

    # El valor que queremos MINIMIZAR es la pérdida en validación
    val_loss = np.min(history.history['val_loss'])
    return val_loss


if __name__ == "__main__":
    # --- 1. Carga de Datos ---
    print("Cargando datos pre-procesados...")
    BASE_DIR = "/content/drive/MyDrive/Doctorado_DavidLizama/datos_tesis/coneval/2024/Bases de datos/"
    
    # Entrenaremos solo con los "no pobres"
    df_no_pobres = pd.read_csv(os.path.join(BASE_DIR, "no_pobres.csv"))
    
    print(f"Datos de 'no pobres' cargados: {df_no_pobres.shape[0]} registros.")

    # --- 2. Preparación de Datos para el Modelo ---
    print("Preparando datos para el autoencoder...")
    
    # Columnas que no son features
    non_features = ['folioviv', 'foliohog', 'numren', 'pobreza', 'pobreza_e']
    
    # El factor de expansión se trata por separado
    weights = df_no_pobres['factor']
    
    # Seleccionar solo las columnas de features
    features_df = df_no_pobres.drop(columns=non_features + ['factor'], errors='ignore')
    
    # --- 3. Split de Entrenamiento y Validación ---
    # Dividimos el set de "no pobres" para entrenar y validar el modelo
    X_train, X_val, w_train, w_val = train_test_split(
        features_df,
        weights,
        test_size=0.2,
        random_state=42
    )
    
    print(f"  - Set de entrenamiento: {X_train.shape[0]} registros")
    print(f"  - Set de validación: {X_val.shape[0]} registros")

    # --- 4. Escalado de Datos ---
    # Es CRUCIAL escalar los datos para redes neuronales.
    # Usamos MinMaxScaler porque mantiene los ceros y es ideal para autoencoders.
    scaler = MinMaxScaler()
    
    # Ajustar el scaler SOLO con los datos de entrenamiento para evitar data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Aplicar la misma transformación a los datos de validación
    X_val_scaled = scaler.transform(X_val)
    
    # Convertir de nuevo a DataFrame para mantener nombres de columnas (opcional pero bueno)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    
    print("Datos escalados con MinMaxScaler.")

    # --- 5. Búsqueda de Hiperparámetros con Optuna ---
    print("\nIniciando estudio con Optuna para encontrar la mejor arquitectura...")
    
    # Creamos un "estudio" que buscará minimizar el resultado de la función 'objective'
    study = optuna.create_study(direction='minimize')
    
    # Optuna llamará a la función 'objective' 50 veces, probando diferentes combinaciones
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, w_train, X_val_scaled, w_val),
        n_trials=100,  # Aumenta este número para una búsqueda más exhaustiva (ej. 100)
        n_jobs=-1      # Usar -1 para paralelizar si tu CPU lo permite, pero en Colab con GPU es mejor 1
    )

    # --- 6. Resultados ---
    print("\n¡Estudio de Optuna completado!")
    print(f"Mejor valor de 'loss' en validación: {study.best_value}")
    print("Mejor combinación de hiperparámetros encontrada:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    # Guardar los resultados del estudio
    results_df = study.trials_dataframe()
    results_path = os.path.join(BASE_DIR, "optuna_study_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResultados completos del estudio guardados en: {results_path}")

    # Visualización (requiere plotly)
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
    except ImportError:
        print("\nPara visualizar los resultados, instala plotly: pip install plotly")
