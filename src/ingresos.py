import pandas as pd
import numpy as np

def generar_ingreso_deflactado_ago2024(trabajo_df: pd.DataFrame,
                                       ingresos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera el ingreso monetario deflactado a precios de agosto 2024
    a partir de las tablas de trabajo e ingresos.

    Parameters
    ----------
    trabajo_df : pd.DataFrame
        Tabla de trabajo (aguinaldo).
    ingresos_df : pd.DataFrame
        Tabla de ingresos.

    Returns
    -------
    pd.DataFrame
        Tabla a nivel hogar con:
        ['folioviv', 'foliohog', 'ing_mon', 'ing_lab', 'ing_ren', 'ing_tra']
    """

    # -----------------------------
    # 1. Base de aguinaldo
    # -----------------------------
    trabajo_df = trabajo_df.copy()

    trabajo_df[['numren', 'id_trabajo', 'pres_2']] = (
        trabajo_df[['numren', 'id_trabajo', 'pres_2']]
        .apply(pd.to_numeric, errors='coerce')
    )

    aguinaldo = pd.pivot_table(
        trabajo_df,
        index=['folioviv', 'foliohog', 'numren'],
        columns='id_trabajo',
        values='pres_2',
        aggfunc=np.sum,
        fill_value=0
    )

    aguinaldo.columns = [f'pres_2{c}' for c in aguinaldo.columns]
    aguinaldo = aguinaldo.reset_index()

    aguinaldo['trab'] = 1
    aguinaldo['aguinaldo1'] = np.where(aguinaldo.get('pres_21', 0) == 2, 1, 0)
    aguinaldo['aguinaldo2'] = np.where(aguinaldo.get('pres_22', 0) == 2, 1, 0)

    aguinaldo = aguinaldo[
        ['folioviv', 'foliohog', 'numren', 'aguinaldo1', 'aguinaldo2', 'trab']
    ]

    # -----------------------------
    # 2. Merge con ingresos
    # -----------------------------
    ing = ingresos_df.copy()

    df = pd.merge(
        ing,
        aguinaldo,
        on=['folioviv', 'foliohog', 'numren'],
        how='outer'
    )

    df = df[
        ~(
            ((df['clave'] == 'P009') & (df['aguinaldo1'] != 1)) |
            ((df['clave'] == 'P016') & (df['aguinaldo2'] != 1))
        )
    ]

    # -----------------------------
    # 3. Deflactores
    # -----------------------------
    dic23 = 0.9732378523
    ene24 = 0.981928198
    feb24 = 0.9828545801
    mar24 = 0.9856778396
    abr24 = 0.9876702962
    may24 = 0.9858395889
    jun24 = 0.9895671737
    jul24 = 0.9999264776
    ago24 = 1
    sep24 = 1.0004926
    oct24 = 1.0059920743
    nov24 = 1.0103740084
    dic24 = 1.0142339335

    num_cols = [
        'mes_1', 'mes_2', 'mes_3', 'mes_4', 'mes_5', 'mes_6',
        'ing_1', 'ing_2', 'ing_3', 'ing_4', 'ing_5', 'ing_6'
    ]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # -----------------------------
    # 4. Deflactación mensual
    # -----------------------------
    df['ing_6'] = np.where(df['mes_6'].isna(), df['ing_6'],
                    np.where(df['mes_6'] == 2, df['ing_6'] / feb24,
                    np.where(df['mes_6'] == 3, df['ing_6'] / mar24,
                    np.where(df['mes_6'] == 4, df['ing_6'] / abr24,
                             df['ing_6'] / may24))))

    df['ing_5'] = np.where(df['mes_5'].isna(), df['ing_5'],
                    np.where(df['mes_5'] == 3, df['ing_5'] / mar24,
                    np.where(df['mes_5'] == 4, df['ing_5'] / abr24,
                    np.where(df['mes_5'] == 5, df['ing_5'] / may24,
                             df['ing_5'] / jun24))))

    df['ing_4'] = np.where(df['mes_4'].isna(), df['ing_4'],
                    np.where(df['mes_4'] == 4, df['ing_4'] / abr24,
                    np.where(df['mes_4'] == 5, df['ing_4'] / may24,
                    np.where(df['mes_4'] == 6, df['ing_4'] / jun24,
                             df['ing_4'] / jul24))))

    df['ing_3'] = np.where(df['mes_3'].isna(), df['ing_3'],
                    np.where(df['mes_3'] == 5, df['ing_3'] / may24,
                    np.where(df['mes_3'] == 6, df['ing_3'] / jun24,
                    np.where(df['mes_3'] == 7, df['ing_3'] / jul24,
                             df['ing_3'] / ago24))))

    df['ing_2'] = np.where(df['mes_2'].isna(), df['ing_2'],
                    np.where(df['mes_2'] == 6, df['ing_2'] / jun24,
                    np.where(df['mes_2'] == 7, df['ing_2'] / jul24,
                    np.where(df['mes_2'] == 8, df['ing_2'] / ago24,
                             df['ing_2'] / sep24))))

    df['ing_1'] = np.where(df['mes_1'].isna(), df['ing_1'],
                    np.where(df['mes_1'] == 7, df['ing_1'] / jul24,
                    np.where(df['mes_1'] == 8, df['ing_1'] / ago24,
                    np.where(df['mes_1'] == 9, df['ing_1'] / sep24,
                             df['ing_1'] / oct24))))

    # -----------------------------
    # 5. Ajustes especiales
    # -----------------------------
    df.loc[df['clave'].isin(['P008', 'P015']), 'ing_1'] = df['ing_1'] / may24 / 12
    df.loc[df['clave'].isin(['P009', 'P016']), 'ing_1'] = df['ing_1'] / dic24 / 12

    df['ing_mens'] = df[['ing_1', 'ing_2', 'ing_3', 'ing_4', 'ing_5', 'ing_6']].mean(axis=1, skipna=True)

    # -----------------------------
    # 6. Clasificación de ingresos
    # -----------------------------
    df.loc[
        ((df['clave'].between('P001', 'P009')) |
         (df['clave'].between('P011', 'P016')) |
         (df['clave'].between('P018', 'P048')) |
         (df['clave'].between('P067', 'P081')) |
         (df['clave'].between('P101', 'P108'))),
        'ing_mon'
    ] = df['ing_mens']

    df.loc[
        ((df['clave'].between('P001', 'P009')) |
         (df['clave'].between('P011', 'P016')) |
         (df['clave'].between('P018', 'P022')) |
         (df['clave'].between('P067', 'P081'))),
        'ing_lab'
    ] = df['ing_mens']

    df.loc[df['clave'].between('P023', 'P031'), 'ing_ren'] = df['ing_mens']

    df.loc[
        ((df['clave'].between('P032', 'P048')) |
         (df['clave'].between('P101', 'P108'))),
        'ing_tra'
    ] = df['ing_mens']

    # -----------------------------
    # 7. Agregación final a hogar
    # -----------------------------
    df = (
        df.groupby(['folioviv', 'foliohog'])[['ing_mon', 'ing_lab', 'ing_ren', 'ing_tra']]
        .sum(numeric_only=True)
        .reset_index()
    )

    return df