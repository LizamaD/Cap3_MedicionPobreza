import pandas as pd
import numpy as np

def process_trabajos(df):
    keys = ['folioviv', 'foliohog', 'numren']

    # 1) flags de trabajo (máximo por grupo -> luego binarizamos)
    trabajo_cols = ['trapais', 'subor', 'indep', 'personal', 'pago', 'contrato', 'tipocontr']
    flags_trabajo = df.groupby(keys, as_index=False)[trabajo_cols].max()
    flags_trabajo[trabajo_cols] = (flags_trabajo[trabajo_cols] == 1).astype(int)

    # 2) flags prestaciones (por grupo, cualquiera no-nulo -> 1, else 0)
    prestaciones_cols = [
        'pres_1', 'pres_2', 'pres_3', 'pres_4', 'pres_5',
        'pres_6', 'pres_7', 'pres_8', 'pres_9', 'pres_10',
        'pres_11', 'pres_12', 'pres_13', 'pres_14', 'pres_15',
        'pres_16', 'pres_17', 'pres_18', 'pres_19', 'pres_20',
        'medtrab_1', 'medtrab_2', 'medtrab_3', 'medtrab_4', 'medtrab_5', 'medtrab_6', 'medtrab_7'
    ]
    flags_prestaciones = (
        df.groupby(keys, as_index=False)[prestaciones_cols]
          .agg(lambda x: x.notna().any())   # True/False por columna
    )
    # convertir a 0/1 y renombrar columnas con prefijo flag_
    flags_prestaciones[prestaciones_cols] = flags_prestaciones[prestaciones_cols].astype(int)
    flags_prestaciones = flags_prestaciones.rename(columns={c: f'flag_{c}' for c in prestaciones_cols})

    # 3) horas trabajadas (suma por grupo)
    horas_trabajadas = df.groupby(keys, as_index=False)['htrab'].sum()

    # 4) empleo principal: filas con id_trabajo==1, tomar la primera por grupo (evita duplicados)
    empleo_cols = [
        'clas_emp', 'tam_emp', 'tipoact', 'no_ing', 'tiene_suel',
        'socios', 'soc_nr1', 'soc_nr2', 'soc_resp',
        'otra_act', 'tipoact2', 'tipoact3', 'tipoact4', 'lugar', 'conf_pers'
    ]
    empleo_principal = (
        df[df['id_trabajo'] == 1]
        .groupby(keys, as_index=False)[empleo_cols]
        .first()
    )

    # 5) merge de todo
    df_final = flags_trabajo.merge(flags_prestaciones, on=keys, how='left') \
                            .merge(horas_trabajadas, on=keys, how='left') \
                            .merge(empleo_principal, on=keys, how='left')

    # 6) Rellenar NaNs en columnas de flags y horas con 0 (las demás columnas se mantienen NaN si faltan)
    flag_cols = [c for c in df_final.columns if c.startswith('flag_')] + ['htrab']
    for c in flag_cols:
        if c in df_final.columns:
            df_final[c] = df_final[c].fillna(0).astype(int)

    return df_final
