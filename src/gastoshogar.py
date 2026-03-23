import pandas as pd
import numpy as np

def procesar_gastos_enigh(gastoshogar):
    """
    Versión blindada: Convierte tipos de datos antes de procesar 
    para evitar el TypeError de concatenación.
    """
    df = gastoshogar.copy()

    # 1. LIMPIEZA DE TIPOS (El "Blindaje")
    # Columnas de código que deben ser strings
    # 'clave' se rellena a 6 dígitos, 'tipo_gasto' es un string como 'G1'
    if 'clave' in df.columns:
        df['clave'] = df['clave'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.zfill(6)
    if 'tipo_gasto' in df.columns:
        df['tipo_gasto'] = df['tipo_gasto'].astype(str).str.strip()

    # Columnas que DEBEN ser números (gastos y factores)
    # errors='coerce' convertirá cualquier cosa rara (como '&' o espacios) en NaN
    # Incluimos lugar_comp y forma_pag1 ya que son códigos numéricos.
    cols_num = ['gasto_tri', 'gasto_nm', 'gas_nm_tri', 'imujer_tri', 'gasto', 'factor',
                'lugar_comp', 'forma_pag1']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. PROCESAMIENTO DE LLAVES
    #df['folioviv'] = df['folioviv'].str.zfill(10)
    
    # 3. CREAR CATEGORÍAS (Usando las claves ya limpias como string)
    df['division_gasto'] = df['clave'].str[:2]
    
    # Gasto en Alimentos (División 01), Salud (06), Educación (10)
    df['gasto_alimentos'] = np.where(df['division_gasto'] == '01', df['gasto_tri'] + df['gas_nm_tri'], 0)
    df['gasto_salud'] = np.where(df['division_gasto'] == '06', df['gasto_tri'] + df['gas_nm_tri'], 0)
    df['gasto_educacion'] = np.where(df['division_gasto'] == '10', df['gasto_tri'] + df['gas_nm_tri'], 0)
    
    # Canales de compra y formas de pago
    # Se usan condiciones numéricas, que son más robustas al tipo de dato de origen.
    # 6: Supermercado, 7: Tienda departamental, 8: Tienda especializada
    df['gasto_formal'] = np.where(df['lugar_comp'].isin([6, 7, 8]), df['gasto_tri'], 0)
    df['gasto_tarjeta'] = np.where(df['forma_pag1'].isin([2, 3]), df['gasto_tri'], 0)

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

    # Corrección de negativos: Asegurar que ingresos por transferencias no sean negativos.
    # Un valor negativo aquí no tiene sentido para la medición de bienestar.
    hogar_gastos['imujer_tri'] = hogar_gastos['imujer_tri'].clip(lower=0)

    # 5. RENOMBRAR PARA CLARIDAD EN MERGE
    hogar_gastos = hogar_gastos.rename(columns={
        'gasto_tri': 'gasto_hog_tri',
        'gas_nm_tri': 'gas_hog_nm_tri',
        'gasto_salud': 'gasto_hog_salud',
        'gasto_educacion': 'gasto_hog_educ'
    })

    # 6. INDICADORES FINALES
    hogar_gastos['gasto_hog_total'] = hogar_gastos['gasto_hog_tri'] + hogar_gastos['gas_hog_nm_tri']
    # Evitar división por cero si el gasto total es 0
    hogar_gastos['pct_gasto_alimentos'] = np.where(hogar_gastos['gasto_hog_total'] > 0, hogar_gastos['gasto_alimentos'] / hogar_gastos['gasto_hog_total'], 0)
    
    return hogar_gastos