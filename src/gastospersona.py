import pandas as pd
import numpy as np

def procesar_gastos_persona_enigh(gastospersona):
    """
    Procesa la tabla GASTOSPERSONA a nivel individual.
    Ideal para modelos de pobreza multidimensional (Educación y Salud).
    """
    df = gastospersona.copy()

    # 1. BLINDAJE DE TIPOS (Evitamos el error de concatenación)
    cols_str = ['folioviv', 'foliohog', 'numren', 'clave', 'inst', 'forma_pag1']
    for col in cols_str:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.0', '', regex=False).str.strip()

    cols_num = ['gasto_tri', 'gas_nm_tri', 'inscrip', 'colegia', 'gasto', 'factor']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Normalizamos llaves
    df['folioviv'] = df['folioviv'].str.zfill(10)
    df['numren'] = df['numren'].str.zfill(2)

    # 2. FEATURE ENGINEERING: Identificación de rubros clave
    # Educación: Usamos los campos específicos 'inscrip' y 'colegia' 
    # y también la clave (División 10 en COICOP)
    df['div_gasto'] = df['clave'].str.zfill(6).str[:2]
    
    # Gasto total en educación para esa persona (trimestralizado)
    # Nota: inscrip y colegia ya suelen estar incluidos en gasto_tri, 
    # pero aquí los aislamos para ver el peso del costo educativo.
    df['gasto_educ_total'] = np.where(df['div_gasto'] == '10', df['gasto_tri'] + df['gas_nm_tri'], 0)
    
    # Gasto en Salud individual (División 06)
    df['gasto_salud_ind'] = np.where(df['div_gasto'] == '06', df['gasto_tri'] + df['gas_nm_tri'], 0)

    # 3. AGREGACIÓN A NIVEL PERSONA (folioviv + foliohog + numren)
    persona_gastos = df.groupby(['folioviv', 'foliohog', 'numren']).agg({
        # Totales trimestrales
        'gasto_tri': 'sum',      # Gasto monetario de la persona
        'gas_nm_tri': 'sum',     # Gasto no monetario (becas en especie, regalos)
        
        # Educación y Salud (Crucial para pobreza)
        'gasto_educ_total': 'sum',
        'gasto_salud_ind': 'sum',
        'inscrip': 'sum',        # Monto específico de inscripciones
        'colegia': 'sum',        # Monto específico de colegiaturas
        
        # Contexto de salud (¿Fue en institución pública o privada?)
        # 01: IMSS, 02: ISSSTE, 07: Privado, etc.
        'inst': lambda x: x.mode()[0] if not x.mode().empty else '00',
        
        # Metadatos
        'factor': 'first'
    }).reset_index()

    # 4. INDICADORES DERIVADOS
    persona_gastos['gasto_persona_total'] = persona_gastos['gasto_tri'] + persona_gastos['gas_nm_tri']
    
    # ¿La persona tiene gasto educativo? (Booleano para el modelo)
    persona_gastos['tiene_gasto_educ'] = (persona_gastos['gasto_educ_total'] > 0).astype(int)
    
    return persona_gastos