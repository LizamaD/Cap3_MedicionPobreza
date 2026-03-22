import pandas as pd
import numpy as np

def process_poblacion(df):
    # 1. BLINDAJE Y LIMPIEZA
    df = df.copy()
    
    # Llaves como strings
    for col in ['parentesco', 'nivelaprob', 'gradoaprob']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.0', '', regex=False).str.strip().str.zfill(2)
    
    #df['folioviv'] = df['folioviv'].str.zfill(10) # Folio viv siempre a 10
    
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
    df['tiene_salud'] = df[inst_cols].apply(lambda x: 1 if 1 in x.values else 0, axis=1)

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

    # E) CONVERSIÓN A VARIABLES DICOTÓMICAS
    # Se convierten varias columnas a formato binario (1=Sí, 0=No/No especificado)
    # para facilitar su uso en modelos.

    # Columnas donde el código '1' representa una respuesta afirmativa.
    affirmative_is_1 = [
        'afrod', 'hablaind', 'etnia', 'alfabetism',
        'asis_esc', 'segsoc', 'trabajo_mp'
    ]
    for col in affirmative_is_1:
        if col in df.columns:
            # Se convierte a numérico antes de comparar para manejar de forma robusta
            # los tipos mixtos (int, float, object). Así, 1, 1.0 y '1' se tratan igual.
            # Los valores no numéricos (errores) o nulos se evaluarán como Falso, resultando en 0.
            condition = pd.to_numeric(df[col], errors='coerce') == 1
            df[col] = np.where(condition, 1, 0)

    # Caso especial para num_trabaj: 1 si tiene al menos un trabajo, 0 si no.
    if 'num_trabaj' in df.columns:
        # Se convierte a numérico, los errores/NaN se vuelven 0.
        # Luego, cualquier valor > 0 se convierte en 1.
        df['num_trabaj'] = (pd.to_numeric(df['num_trabaj'], errors='coerce').fillna(0) > 0).astype(int)
    # 3. SELECCIÓN FINAL (Las que de verdad mueven la aguja)
    final_columns = [
        'folioviv', 'foliohog', 'numren', 'parentesco', 'edad', 'sexo',
        'anios_esc', 'alfabetism', 'asis_esc', 
        'tiene_salud', 'segsoc', 'pob_discapacidad',
        'etnia', 'hablaind', 'afrod', 
        'trabajo_mp', 'num_trabaj', 'score_trabajo_domestico',
        'factor'
    ]
    
    # Usar intersection para evitar errores si alguna columna no existe
    return df[df.columns.intersection(final_columns)]