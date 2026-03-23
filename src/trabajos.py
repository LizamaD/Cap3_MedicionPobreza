import pandas as pd
import numpy as np

def process_trabajos(df):
    df = df.copy()
    # Reemplazar strings en blanco o con espacios por NaN para unificar valores faltantes.
    # Esto soluciona el problema de tener la categoría ' ' en la tabla final.
    df = df.replace(r'^\s*$', np.nan, regex=True)

    keys = ['folioviv', 'foliohog', 'numren']

    trabajo_cols = ['trapais', 'subor', 'indep', 'personal', 'pago', 'contrato', 'tipocontr']
    prestaciones_cols = [
        'pres_1', 'pres_2', 'pres_3', 'pres_4', 'pres_5',
        'pres_6', 'pres_7', 'pres_8', 'pres_9', 'pres_10',
        'pres_11', 'pres_12', 'pres_13', 'pres_14', 'pres_15',
        'pres_16', 'pres_17', 'pres_18', 'pres_19', 'pres_20',
        'medtrab_1', 'medtrab_2', 'medtrab_3', 'medtrab_4', 'medtrab_5', 'medtrab_6', 'medtrab_7'
    ]
    empleo_cols = [
        'clas_emp', 'tam_emp', 'tipoact', 'no_ing', 'tiene_suel',
        'socios', 'soc_nr1', 'soc_nr2', 'soc_resp',
        'otra_act', 'tipoact2', 'tipoact3', 'tipoact4', 'lugar', 'conf_pers'
    ]

    # 1) PREPARACIÓN VECTORIZADA (Evita usar lambda en groupby)
    cols_base = keys + trabajo_cols + prestaciones_cols + ['htrab']
    df_aggs = df[[c for c in cols_base if c in df.columns]].copy()
    
    # Convertir prestaciones a 1 (no nulo) y 0 (nulo) antes de agrupar
    df_aggs[prestaciones_cols] = df_aggs[prestaciones_cols].notna().astype(int)
    
    # 2) UN SOLO GROUPBY (En lugar de 3 separados)
    agg_dict = {col: 'max' for col in trabajo_cols + prestaciones_cols}
    agg_dict['htrab'] = 'sum'
    
    agrupado = df_aggs.groupby(keys, as_index=False).agg(agg_dict)
    
    for col in trabajo_cols:
        agrupado[col] = (agrupado[col] == 1).astype(int)
        
    agrupado = agrupado.rename(columns={c: f'flag_{c}' for c in prestaciones_cols})

    # 3) EMPLEO PRINCIPAL (drop_duplicates es C-level y rapidísimo)
    empleo_principal = df[df['id_trabajo'] == 1].drop_duplicates(subset=keys, keep='first')[keys + empleo_cols]

    # 4) MERGE ÚNICO Y LIMPIEZA
    df_final = agrupado.merge(empleo_principal, on=keys, how='left')
    df_final['htrab'] = df_final['htrab'].fillna(0).astype(int)

    # Convertir variables categóricas a string para que los modelos no las traten como numéricas
    for col in ['class_emp', 'tam_emp', 'tipoact']:
        if col in df_final.columns:
            # Rellenar NaNs (de personas sin trabajo principal) y convertir a string
            df_final[col] = df_final[col].fillna(0).astype(int).astype(str)

    # Convertir 'no_ing' a una variable dicotómica
    if 'no_ing' in df_final.columns:
        # 1 si no recibe ingresos, 0 en cualquier otro caso (sí recibe, no aplica, no especificado)
        condition = pd.to_numeric(df_final['no_ing'], errors='coerce') == 1
        df_final['no_ing'] = np.where(condition, 1, 0)

    # Convertir 'tiene_suel' a una variable dicotómica
    if 'tiene_suel' in df_final.columns:
        # 1 si tiene sueldo, 0 en otro caso (no tiene, no aplica, no especificado)
        condition = pd.to_numeric(df_final['tiene_suel'], errors='coerce') == 1
        df_final['tiene_suel'] = np.where(condition, 1, 0)

    return df_final
