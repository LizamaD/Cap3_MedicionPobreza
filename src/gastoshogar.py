import pandas as pd
import numpy as np

def procesar_gastos_enigh(gastoshogar):
    """
    Versión blindada: Convierte tipos de datos antes de procesar 
    para evitar el TypeError de concatenación.
    """
    df = gastoshogar.copy()

    # 1. LIMPIEZA DE TIPOS (El "Blindaje")
    # Columnas que DEBEN ser strings (llaves y códigos)
    cols_str = ['clave', 'lugar_comp', 'forma_pag1', 'tipo_gasto']
    for col in cols_str:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.0', '', regex=False).str.strip()

    # Columnas que DEBEN ser números (gastos y factores)
    # errors='coerce' convertirá cualquier cosa rara (como '&' o espacios) en NaN
    cols_num = ['gasto_tri', 'gasto_nm', 'gas_nm_tri', 'imujer_tri', 'gasto', 'factor']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. PROCESAMIENTO DE LLAVES
    #df['folioviv'] = df['folioviv'].str.zfill(10)
    
    # 3. CREAR CATEGORÍAS (Usando las claves ya limpias como string)
    df['division_gasto'] = df['clave'].str.zfill(6).str[:2]
    
    # Gasto en Alimentos (División 01), Salud (06), Educación (10)
    df['gasto_alimentos'] = np.where(df['division_gasto'] == '01', df['gasto_tri'] + df['gas_nm_tri'], 0)
    df['gasto_salud'] = np.where(df['division_gasto'] == '06', df['gasto_tri'] + df['gas_nm_tri'], 0)
    df['gasto_educacion'] = np.where(df['division_gasto'] == '10', df['gasto_tri'] + df['gas_nm_tri'], 0)
    
    # Canales de compra y formas de pago
    df['gasto_formal'] = np.where(df['lugar_comp'].isin(['06', '07', '08']), df['gasto_tri'], 0)
    df['gasto_tarjeta'] = np.where(df['forma_pag1'].isin(['02', '03']), df['gasto_tri'], 0)

    # 4. AGREGACIÓN (Ahora sí, sin errores de tipo)
    hogar_gastos = df.groupby(['folioviv', 'foliohog']).agg({
        'gasto_tri': 'sum',
        'gas_nm_tri': 'sum',
        'imujer_tri': 'sum',
        'gasto_alimentos': 'sum',
        'gasto_salud': 'sum',
        'gasto_educacion': 'sum',
        'gasto_formal': 'sum',
        'gasto_tarjeta': 'sum'
    }).reset_index()

    # 5. INDICADORES FINALES
    hogar_gastos['gasto_total'] = hogar_gastos['gasto_tri'] + hogar_gastos['gas_nm_tri']
    hogar_gastos['pct_gasto_alimentos'] = (hogar_gastos['gasto_alimentos'] / hogar_gastos['gasto_total']).fillna(0)
    
    return hogar_gastos