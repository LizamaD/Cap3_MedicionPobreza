import pandas as pd
import numpy as np

def process_hogares(df):
    # 1. Identificadores y Factores (Obligatorios)
    identificadores = ['folioviv', 'foliohog', 'tot_integ', 'clase_hog']
    
    # 2. Seguridad Alimentaria (Para carencia de CONEVAL)
    # Seleccionamos acc_alim 1 al 16
    alim_cols = [f'acc_alim{i}' for i in range(1, 17)] + [f'alim17_{i}' for i in range(1, 13)] + ['acc_alim18']
    
    # 3. Equipamiento (Solo el "Número de", quitamos los "años")
    equipamiento = [
        'num_auto', 'num_van', 'num_pick', 'num_moto', 'num_compu', 'num_lap', 
        'num_table', 'num_refri', 'num_estuf', 'num_lavad', 'num_micro', 'num_licua'
    ]
    
    # 4. Servicios y Conectividad (Fundamentales hoy en día)
    servicios = ['telefono', 'celular', 'conex_inte', 'tv_paga']
    
    # 5. Variables de flujo (Autoconsumo y transferencias)
    flujos = ['autocons', 'regalos', 'transferen', 'remunera']

    all_cols = identificadores + alim_cols + equipamiento + servicios + flujos
    
    # Filtramos solo las columnas que existan en el DF por si las moscas
    df_filtered = df[df.columns.intersection(all_cols)].copy()
    
    # --- TRANSFORMACIONES RÁPIDAS ---
    
    # Limpieza de folios
    #df_filtered['folioviv'] = df_filtered['folioviv'].astype(str).str.zfill(10)
    #df_filtered['foliohog'] = df_filtered['foliohog'].astype(str)
    
    # Crear un "Índice de Conectividad" (0 a 3)
    # Convertimos a binario (1 si tiene, 0 si no)
    for col in ['conex_inte', 'celular', 'tv_paga']:
        df_filtered[col] = df_filtered[col].replace({'1': 1, '2': 0}).fillna(0).astype(int)
    
    df_filtered['indice_conectividad'] = df_filtered['conex_inte'] + df_filtered['celular']
    
    # IAAS Score (Suma de carencias alimentarias donde 1 es "Sí")
    # Nota: En la base original 1=Sí, 2=No.
    for col in alim_cols:
        if col in df_filtered.columns:
            df_filtered[f'{col}_bin'] = np.where(df_filtered[col] == '1', 1, 0)
            
    # Creamos el score total de inseguridad alimentaria
    bin_cols = [c for c in df_filtered.columns if '_bin' in c]
    df_filtered['score_iaas'] = df_filtered[bin_cols].sum(axis=1)
    
    return df_filtered
