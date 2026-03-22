import pandas as pd
import numpy as np

def process_viviendas(df):
    df = df.copy()
    
    # 1. BLINDAJE Y LIMPIEZA
    #df['folioviv'] = df['folioviv'].astype(str).str.zfill(10)
    
    # Reemplazar códigos de "No especificado" específicos de la ENIGH a NaN
    # antes de convertir a numérico. '&' en categóricas, -1 en estim_pago, etc.
    df = df.replace({'&': np.nan, '& ': np.nan})
    
    # Columnas categóricas clave (asegurarse que sean string/object)
    cols_cat = ['tipo_viv', 'tenencia']
    for col in cols_cat:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.0', '', regex=False).str.strip()

    # Columnas numéricas clave
    cols_num = ['tot_resid', 'cuart_dorm', 'num_cuarto', 'estim_pago']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. FEATURE ENGINEERING (Las variables que "venden")

    # A) HACINAMIENTO (Ratio de personas por dormitorio)
    # Evitamos división por cero con np.where. Si cuart_dorm es 0, el hacinamiento es el total de residentes.
    df['ratio_hacinamiento'] = np.where(df['cuart_dorm'] > 0, 
                                        df['tot_resid'] / df['cuart_dorm'], 
                                        df['tot_resid']) # Si no hay cuarto, la persona misma es el ratio
    df['ind_hacinamiento'] = (df['ratio_hacinamiento'] > 2.5).astype(int)

    # B) CALIDAD DE MATERIALES (Dummies de carencia)
    # Se ajustan las condiciones para que coincidan con el tipo de dato (numérico o string).
    df['carencia_piso'] = np.where(df['mat_pisos'].astype(str) == '1', 1, 0) # 1 es Tierra
    # Paredes de material de desecho, lámina o embarro (metodología CONEVAL)
    df['carencia_pared'] = np.where(df['mat_pared'].isin([1, 2, 3]), 1, 0)
    # Techos de material de desecho o lámina de cartón
    df['carencia_techo'] = np.where(df['mat_techos'].isin([1, 2]), 1, 0)

    # C) SERVICIOS BÁSICOS
    # Carencia de Agua (No entubada dentro de la vivienda o terreno)
    df['carencia_agua'] = np.where(~df['agua_ent'].isin([1, 2]), 1, 0)
    # Carencia de Drenaje (Si no tiene o es directo a tubería que da a la calle/río)
    df['carencia_drenaje'] = np.where(df['drenaje'].isin([3, 4, 5]), 1, 0)
    # Carencia de Energía Eléctrica
    df['carencia_luz'] = np.where(df['disp_elect'] == 2, 1, 0)
    # Combustible para cocinar (Leña o carbón sin chimenea)
    df['carencia_combustible'] = np.where(df['combus'].astype(str) == '1', 1, 0)

    # D) SCORE DE INFRAESTRUCTURA (Suma de activos de la vivienda)
    activos = ['cisterna', 'tinaco_azo', 'calent_sol', 'calent_gas', 'aire_acond']
    for col in activos:
        df[f'{col}_bin'] = np.where(df[col] == 1, 1, 0)
    
    activos_bin = [c for c in df.columns if '_bin' in c]
    df['score_vivienda_alta'] = df[activos_bin].sum(axis=1)

    # 3. SELECCIÓN FINAL
    final_columns = [
        'folioviv', 'tipo_viv', 'tenencia', 'estim_pago',
        'ratio_hacinamiento', 'ind_hacinamiento',
        'carencia_piso', 'carencia_pared', 'carencia_techo',
        'carencia_agua', 'carencia_drenaje', 'carencia_luz', 'carencia_combustible',
        'score_vivienda_alta', 'tot_resid', 'tot_hog'
    ]
    
    return df[final_columns]