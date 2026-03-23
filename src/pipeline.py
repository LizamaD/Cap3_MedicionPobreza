import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Importar todas las funciones de procesamiento de los otros módulos
from src.poblacion import process_poblacion
from src.viviendas import process_viviendas
from src.hogares import process_hogares
from src.trabajos import process_trabajos
from src.gastospersona import procesar_gastos_persona_enigh
from src.gastoshogar import procesar_gastos_enigh
from src.ingresos import generar_ingreso_deflactado_ago2024

def create_master_table(
    pob_keys: pd.DataFrame,
    pob_df: pd.DataFrame,
    viv_df: pd.DataFrame,
    hog_df: pd.DataFrame,
    trab_df: pd.DataFrame,
    gasper_df: pd.DataFrame,
    gashog_df: pd.DataFrame,
    ing_df: pd.DataFrame,
    output_path: str = "master_table.csv"
) -> pd.DataFrame:
    """
    Orquesta el procesamiento de todas las tablas de la ENIGH, las une en una
    sola tabla maestra a nivel persona y la guarda en un archivo CSV.

    Args:
        pob_keys: DataFrame con las llaves y la variable objetivo de pobreza.
        pob_df: DataFrame raw de población.
        viv_df: DataFrame raw de viviendas.
        hog_df: DataFrame raw de hogares.
        trab_df: DataFrame raw de trabajos.
        gasper_df: DataFrame raw de gastos de persona.
        gashog_df: DataFrame raw de gastos de hogar.
        ing_df: DataFrame raw de ingresos.
        output_path: Ruta donde se guardará el archivo CSV final.

    Returns:
        Un DataFrame maestro con toda la información unificada a nivel persona.
    """
    print("Procesando tablas individuales...")
    pob_keys = pob_keys[['folioviv','foliohog','numren','pobreza','pobreza_e']]
    pob_keys = pob_keys[~pob_keys['pobreza'].isna()]
    pob_proc = process_poblacion(pob_df)
    viv_proc = process_viviendas(viv_df)
    hog_proc = process_hogares(hog_df)
    trab_proc = process_trabajos(trab_df)
    gasper_proc = procesar_gastos_persona_enigh(gasper_df)
    gashog_proc = procesar_gastos_enigh(gashog_df)
    # Ingresos necesita los DFs raw de trabajo e ingresos
    ing_proc = generar_ingreso_deflactado_ago2024(trab_df, ing_df)

    print("Uniendo tablas en una tabla maestra...")
    # La tabla de población es nuestra base (nivel persona)
    # Las llaves (folioviv, foliohog, numren) ya vienen procesadas como strings
    # desde cada función. Es más seguro y consistente hacer el merge con strings
    # para evitar problemas con ceros a la izquierda.
    master_df = pob_keys.merge(pob_proc, on=['folioviv', 'foliohog', 'numren'], how='left') 

    # Merge con viviendas (nivel vivienda)
    master_df = master_df.merge(viv_proc, on='folioviv', how='left')

    # Merge con hogares, gastos de hogar e ingresos (nivel hogar)
    keys_hogar = ['folioviv', 'foliohog']
    master_df = master_df.merge(hog_proc, on=keys_hogar, how='left')
    master_df = master_df.merge(gashog_proc, on=keys_hogar, how='left')
    master_df = master_df.merge(ing_proc, on=keys_hogar, how='left')

    # Merge con trabajos y gastos de persona (nivel persona)
    keys_persona = ['folioviv', 'foliohog', 'numren']
    master_df = master_df.merge(trab_proc, on=keys_persona, how='left')
    master_df = master_df.merge(gasper_proc, on=keys_persona, how='left')

    print(f"Tabla maestra creada con {master_df.shape[0]} filas y {master_df.shape[1]} columnas.")
    
    # Guardar en CSV
    print(f"Guardando tabla maestra en '{output_path}'...")
    master_df.to_csv(output_path, index=False)

    return master_df


def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes en un DataFrame.
    - Para variables categóricas, usa la categoría '__MISSING__'.
    - Para variables numéricas, usa la mediana y crea una columna flag.

    Args:
        df: El DataFrame que se va a limpiar.

    Returns:
        Un DataFrame sin valores nulos.
    """
    print("Iniciando proceso de imputación de datos...")
    df_clean = df.copy()

    # 1. Imputar variables categóricas
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            print(f"  - Categórica '{col}': Imputando NaNs con '__MISSING__'.")
            df_clean[col] = df_clean[col].fillna('__MISSING__')

    # 2. Imputar variables numéricas
    numerical_cols = df_clean.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if df_clean[col].isnull().any():
            # Crear la columna flag que indica imputación
            flag_col_name = f'{col}_imputed'
            df_clean[flag_col_name] = df_clean[col].isnull().astype(int)

            # Calcular la mediana (ignorando NaNs)
            median_val = df_clean[col].median()
            
            print(f"  - Numérica '{col}': Imputando NaNs con mediana ({median_val:.2f}) y creando flag '{flag_col_name}'.")

            # Imputar con la mediana
            df_clean[col] = df_clean[col].fillna(median_val)
    
    print("Imputación completada. El DataFrame ya no tiene valores faltantes.")
    return df_clean


def prepare_and_split_for_autoencoder(df_imputed: pd.DataFrame, output_dir: str) -> tuple:
    """
    Prepara los datos para el autoencoder: convierte categóricas a dummies,
    y luego divide el dataset en hogares pobres y no pobres.

    Args:
        df_imputed: El DataFrame maestro ya imputado.
        output_dir: El directorio donde se guardarán los archivos CSV.

    Returns:
        Una tupla con (df_processed, df_pobres, df_no_pobres).
    """
    print("\nIniciando preparación y split para el autoencoder...")
    df_processed = df_imputed.copy()

    # 1. Convertir categóricas a numéricas (One-Hot Encoding)
    categorical_cols = [
        'parentesco', 'sexo', 'tipo_viv', 'tenencia', 'class_emp', 'tam_emp', 'tipoact', 
        'socios', 'soc_nr1', 'soc_nr2', 'soc_resp', 'otra_act', 'tipoact2', 
        'tipoact3', 'tipoact4', 'lugar', 'conf_pers', 'inst'
    ]
    
    # Nos aseguramos de que solo procesamos las columnas que existen en el DF
    cols_to_process = [col for col in categorical_cols if col in df_processed.columns]
    
    print(f"Convirtiendo {len(cols_to_process)} columnas categóricas a formato dummy...")
    df_processed = pd.get_dummies(df_processed, columns=cols_to_process, dummy_na=False, dtype=int)
    
    # La categoría '__MISSING__' creada por la imputación también se convertirá en una columna dummy,
    # lo cual es perfecto porque le dice al modelo que la ausencia de dato es información.
    
    print(f"El DataFrame ahora tiene {df_processed.shape[1]} columnas después del one-hot encoding.")
    
    # 2. Eliminar columnas con baja varianza
    print("\nAplicando filtro de baja varianza...")
    
    # Separar las columnas que no son features para no eliminarlas
    non_features = ['folioviv', 'foliohog', 'numren', 'pobreza', 'pobreza_e', 'factor']
    feature_cols = [col for col in df_processed.columns if col not in non_features]
    features_df = df_processed[feature_cols]

    # Este umbral elimina variables que tienen el mismo valor en más del 99% de los casos
    selector = VarianceThreshold(threshold=(.99 * (1 - .99)))
    selector.fit(features_df)

    # Obtener las columnas que se quedan y las que se van
    kept_features = features_df.columns[selector.get_support()].tolist()
    dropped_features = features_df.columns[~selector.get_support()].tolist()

    print(f"Dimensión original de features: {features_df.shape[1]}")
    print(f"Variables eliminadas por baja varianza: {len(dropped_features)}")
    print(f"Variables que quedaron: {len(kept_features)}")

    if dropped_features:
        print("\nColumnas eliminadas:")
        # Imprimir en un formato más legible si son muchas
        for i in range(0, len(dropped_features), 5):
             print("  ", dropped_features[i:i+5])

    # Reconstruir el DataFrame con solo las columnas de alta varianza + las no-features
    df_processed = df_processed[non_features + kept_features]
    print(f"\nDimensión final del DataFrame procesado: {df_processed.shape[1]} columnas.")

    # 3. Guardar el DataFrame procesado completo
    full_path = f"{output_dir}/full_processed.csv"
    print(f"Guardando el DataFrame procesado completo en '{full_path}'...")
    df_processed.to_csv(full_path, index=False)

    # 4. Hacer el split entre pobres y no pobres
    # Usamos la variable 'pobreza' que viene desde el inicio (1=Pobre, 0=No Pobre)
    print("Realizando el split entre hogares pobres y no pobres...")
    df_pobres = df_processed[df_processed['pobreza'] == 1].copy()
    df_no_pobres = df_processed[df_processed['pobreza'] == 0].copy()

    print(f"  - Encontrados {len(df_pobres)} registros de personas en pobreza.")
    print(f"  - Encontrados {len(df_no_pobres)} registros de personas no pobres.")

    # 5. Guardar los DataFrames del split
    pobres_path = f"{output_dir}/pobres.csv"
    no_pobres_path = f"{output_dir}/no_pobres.csv"
    
    print(f"Guardando el set de pobres en '{pobres_path}'...")
    df_pobres.to_csv(pobres_path, index=False)
    
    print(f"Guardando el set de no pobres en '{no_pobres_path}'...")
    df_no_pobres.to_csv(no_pobres_path, index=False)

    print("Proceso completado. ¡Archivos listos para el modelado!")
    return df_processed, df_pobres, df_no_pobres