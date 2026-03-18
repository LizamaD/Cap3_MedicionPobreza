import pandas as pd
import numpy as np

def process_poblacion(df):
    # 1. BLINDAJE Y LIMPIEZA
    df = df.copy()
    
    # Llaves como strings
    for col in ['folioviv', 'foliohog', 'numren', 'parentesco', 'nivelaprob', 'gradoaprob']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.0', '', regex=False).str.strip().str.zfill(2)
    
    df['folioviv'] = df['folioviv'].str.zfill(10) # Folio viv siempre a 10
    
    # Números como números
    cols_num = ['edad', 'hijos_viv', 'hijos_sob', 'factor']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. TRANSFORMACIONES ESTRATÉGICAS
    
    # A) REZAGO EDUCATIVO (Simplificado)
    # Creamos una variable de 'años_escolaridad' aproximada
    # (Esto es una joya para modelos de regresión)
    dict_nivel = {
        '00': 0, '01': 0, '02': 6, '03': 9, '04': 12, 
        '05': 12, '06': 12, '07': 16, '08': 18, '09': 22
    }
    df['anios_esc'] = df['nivelaprob'].map(dict_nivel).fillna(0) + pd.to_numeric(df['gradoaprob'], errors='coerce').fillna(0)

    # B) ACCESO A SALUD (Dummy 1/0)
    # Si tiene cualquier institución de la 1 a la 9 (IMSS, ISSSTE, Bienestar, etc.)
    inst_cols = [f'inst_{i}' for i in range(1, 10)]
    df['tiene_salud'] = df[inst_cols].apply(lambda x: 1 if '1' in x.values else 0, axis=1)

    # C) DISCAPACIDAD (Dummy 1/0)
    # Si tiene dificultad mucha o no puede en cualquiera de las dimensiones
    disc_cols = [c for c in df.columns if 'disc_' in c]
    # En ENIGH, 1 y 2 es "poca/ninguna", 3 y 4 es "mucha/no puede"
    df['pob_discapacidad'] = df[disc_cols].apply(lambda x: 1 if any(v in ['3', '4'] for v in x.values) else 0, axis=1)

    # D) TRABAJO DOMÉSTICO (Consolidar Uso del Tiempo)
    # Sumamos horas de las columnas de usotiempo (ej. usotiempo1 es cocinar, 2 limpiar, etc.)
    # Ojo: aquí deberías decidir si sumas las horas o solo el score
    uso_cols = [f'usotiempo{i}' for i in range(1, 8)]
    df['score_trabajo_domestico'] = df[uso_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

    # 3. SELECCIÓN FINAL (Las que de verdad mueven la aguja)
    final_columns = [
        'folioviv', 'foliohog', 'numren', 'parentesco', 'edad', 'sexo',
        'anios_esc', 'alfabetism', 'asis_esc', 
        'tiene_salud', 'segsoc', 'pob_discapacidad',
        'etnia', 'hablaind', 'afrod', 
        'trabajo_mp', 'num_trabaj', 'score_trabajo_domestico',
        'factor'
    ]
    
    return df[final_columns]