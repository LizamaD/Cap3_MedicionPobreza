import pandas as pd
import numpy as np

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
        pobreza_df: DataFrame raw de pobreza.
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
    pob_proc[['folioviv', 'foliohog', 'numren']] = pob_proc[['folioviv', 'foliohog', 'numren']].astype(int)
    master_df = pob_keys.merge(pob_proc, on=['folioviv', 'foliohog', 'numren'], how='left') 

    # Merge con viviendas (nivel vivienda)
    viv_proc['folioviv'] = viv_proc['folioviv'].astype(int)
    master_df = master_df.merge(viv_proc, on='folioviv', how='left')

    # Merge con hogares, gastos de hogar e ingresos (nivel hogar)
    keys_hogar = ['folioviv', 'foliohog']
    hog_proc[keys_hogar] = hog_proc[keys_hogar].astype(int)
    gashog_proc[keys_hogar] = gashog_proc[keys_hogar].astype(int)
    ing_proc[keys_hogar] = ing_proc[keys_hogar].astype(int)
    master_df = master_df.merge(hog_proc, on=keys_hogar, how='left')
    master_df = master_df.merge(gashog_proc, on=keys_hogar, how='left')
    master_df = master_df.merge(ing_proc, on=keys_hogar, how='left')

    # Merge con trabajos y gastos de persona (nivel persona)
    keys_persona = ['folioviv', 'foliohog', 'numren']
    trab_proc[keys_persona] = trab_proc[keys_persona].astype(int)
    gasper_proc[keys_persona] = gasper_proc[keys_persona].astype(int)
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