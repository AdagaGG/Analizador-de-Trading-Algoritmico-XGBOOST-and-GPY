"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🏭 THE QUANT REFINERY v3.2 🏭                              ║
║                  SISTEMA ADAPTATIVO DE TRADING OPTIMIZADO                    ║
║                                                                              ║
║  HOTFIX v3.2:                                                               ║
║  🔧 Fix crítico de dtype en columna 'Equity' (int64 → float64)             ║
║  🔧 Fix columnas 'Position' y 'Prediction' (explicit float/int)            ║
║                                                                              ║
║  MEJORAS v3.1:                                                              ║
║  ✅ Eliminación de redundancia (función única de refinado)                 ║
║  ✅ Type Safety (separación número/texto antes de float32)                 ║
║  ✅ Optimización GPU (max_bin=256 para RTX 3050 4GB)                       ║
║  ✅ Pandas 2.0 syntax (.ffill() / .bfill() sin warnings)                   ║
║  ✅ Parche reforzado de yfinance (limpieza de columnas duplicadas)         ║
║                                                                              ║
║  ANALOGÍA INGENIERIL:                                                       ║
║  - Precios = Materia Prima Cruda                                            ║
║  - Benchmarks = Señales de Control del Proceso                              ║
║  - ADX = Número de Reynolds (Laminar vs Turbulento)                        ║
║  - XGBoost = Controlador Adaptativo Multivariable                           ║
║  - Fase_Mercado = Sensor de Régimen de Flujo                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# 🎨 CONFIGURACIÓN DE LA INTERFAZ
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🏭 The Quant Refinery v3.2",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 The Quant Refinery v3.2")
st.markdown("### *Sistema Adaptativo Optimizado - Fix de Tipos de Datos*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# 🛡️ FUNCIÓN AUXILIAR: PARCHE REFORZADO DE YFINANCE
# ═══════════════════════════════════════════════════════════════════════════

def fix_yfinance_columns(df):
    """
    🛡️ PARCHE REFORZADO v3.1: Limpieza Total de Columnas
    
    PROBLEMAS QUE RESUELVE:
    1. MultiIndex (columnas tipo tupla)
    2. Columnas duplicadas
    3. Nombres vacíos o con espacios
    4. Asegurar que existan: Close, High, Low, Open, Volume
    """
    
    # Paso 1: Aplanar MultiIndex si existe
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Paso 2: Limpiar nombres de columnas
    df.columns = df.columns.str.strip()
    
    # Paso 3: Mapeo a nombres estándar (case-insensitive)
    column_mapping = {
        'close': 'Close',
        'high': 'High',
        'low': 'Low',
        'open': 'Open',
        'volume': 'Volume',
        'adj close': 'Adj Close',
        'adjclose': 'Adj Close'
    }
    
    # Aplicar mapeo
    df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
    
    # Paso 4: Eliminar columnas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Paso 5: Verificar columnas críticas
    required_cols = ['Close', 'High', 'Low', 'Open']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"⚠️ Columnas faltantes: {missing_cols}")
        return None
    
    return df

# ═══════════════════════════════════════════════════════════════════════════
# 📡 FUNCIÓN 1: DESCARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def descargar_datos(ticker, periodo_dias=365):
    """
    📡 SISTEMA DE ADQUISICIÓN DE DATOS
    """
    try:
        st.info(f"📡 Descargando {ticker}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=periodo_dias)
        
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError("DataFrame vacío")
        
        # 🛡️ Aplicar parche reforzado
        df = fix_yfinance_columns(df)
        
        if df is None:
            raise ValueError("Estructura de datos inválida")
        
        # ⚡ Optimización de memoria
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].astype('float32')
        
        st.success(f"✅ {len(df)} barras descargadas")
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Error: {e}. Generando datos sintéticos...")
        
        # MODO DEMO
        days = periodo_dias
        dates = pd.date_range(end=datetime.now(), periods=days)
        
        trend = np.linspace(100, 150, days)
        seasonal = 20 * np.sin(np.linspace(0, 4*np.pi, days))
        noise = np.random.randn(days) * 5
        prices = trend + seasonal + noise
        
        df = pd.DataFrame({
            'Open': prices * 0.98,
            'High': prices * 1.02,
            'Low': prices * 0.97,
            'Close': prices,
            'Volume': np.random.randint(1e6, 1e7, days)
        }, index=dates)
        
        # Convertir a float32
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].astype('float32')
        
        st.success("✅ Datos sintéticos generados (MODO DEMO)")
        return df

# ═══════════════════════════════════════════════════════════════════════════
# 🔍 FUNCIÓN 2: ESCÁNER DE RESONANCIA MACRO
# ═══════════════════════════════════════════════════════════════════════════

def escanear_benchmarks(df_activo, periodo_dias=365):
    """
    🔍 MÓDULO DE ESCÁNER DE RESONANCIA MACRO
    """
    
    benchmarks = {
        '^GSPC': 'S&P 500',
        '^IXIC': 'Nasdaq',
        '^VIX': 'VIX (Volatilidad)',
        'DX-Y.NYB': 'Dólar Index',
        '^TNX': 'Treasury 10Y',
        'GC=F': 'Oro',
        'CL=F': 'Petróleo WTI',
        'BTC-USD': 'Bitcoin'
    }
    
    st.info("🔍 Escaneando benchmarks macro...")
    
    correlaciones = {}
    
    for ticker, nombre in benchmarks.items():
        try:
            df_bench = descargar_datos(ticker, periodo_dias)
            
            # Alinear temporalmente
            df_merged = pd.merge(
                df_activo[['Close']], 
                df_bench[['Close']], 
                left_index=True, 
                right_index=True, 
                how='inner',
                suffixes=('_activo', '_bench')
            )
            
            # Calcular correlación
            if len(df_merged) > 20:
                corr = df_merged['Close_activo'].corr(df_merged['Close_bench'])
                correlaciones[ticker] = {
                    'nombre': nombre,
                    'correlacion': corr,
                    'data': df_bench
                }
                
        except Exception:
            continue
    
    if not correlaciones:
        st.warning("⚠️ No se pudo descargar benchmarks. Modo autónomo.")
        return df_activo, None, None
    
    # Selección del benchmark dominante
    benchmark_dominante = max(
        correlaciones.items(), 
        key=lambda x: abs(x[1]['correlacion'])
    )
    
    ticker_dom = benchmark_dominante[0]
    info_dom = benchmark_dominante[1]
    
    st.success(f"""
    ✅ **Benchmark Dominante:**
    - {info_dom['nombre']} ({ticker_dom})
    - Correlación: {info_dom['correlacion']:.3f}
    """)
    
    # Fusión de datos
    df_bench = info_dom['data']
    
    df_fusionado = pd.merge(
        df_activo,
        df_bench[['Close']].rename(columns={'Close': 'Benchmark_Close'}),
        left_index=True,
        right_index=True,
        how='left'
    )
    
    # ✅ FIX PANDAS 2.0
    df_fusionado['Benchmark_Close'] = df_fusionado['Benchmark_Close'].ffill()
    df_fusionado['Benchmark_Close'] = df_fusionado['Benchmark_Close'].bfill()
    
    # Features exógenas
    df_fusionado['Ret_Activo'] = df_fusionado['Close'].pct_change()
    df_fusionado['Ret_Benchmark'] = df_fusionado['Benchmark_Close'].pct_change()
    df_fusionado['Relative_Strength_Macro'] = (
        df_fusionado['Ret_Activo'] - df_fusionado['Ret_Benchmark']
    )
    
    df_fusionado['Rolling_Corr'] = (
        df_fusionado['Close']
        .rolling(20)
        .corr(df_fusionado['Benchmark_Close'])
    )
    
    return df_fusionado, info_dom['nombre'], info_dom['correlacion']

# ═══════════════════════════════════════════════════════════════════════════
# 🌊 FUNCIÓN 3: DETECTOR DE FASES DE MERCADO
# ═══════════════════════════════════════════════════════════════════════════

def calcular_fase_mercado(df):
    """
    🌊 DETECTOR DE NÚMERO DE REYNOLDS FINANCIERO
    """
    
    data = df.copy()
    
    # Calcular ADX
    try:
        adx_data = ta.adx(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            length=14
        )
        
        if adx_data is not None and 'ADX_14' in adx_data.columns:
            data['ADX'] = adx_data['ADX_14']
        else:
            raise ValueError("ADX no calculado")
            
    except Exception:
        st.warning("⚠️ ADX no disponible, usando valor neutro")
        data['ADX'] = 25.0
    
    # Clasificación de fase
    def clasificar_fase(adx_val):
        if pd.isna(adx_val):
            return "DESCONOCIDO"
        elif adx_val > 25:
            return "LAMINAR"
        elif adx_val < 20:
            return "TURBULENTO"
        else:
            return "TRANSICIÓN"
    
    data['Fase_Mercado'] = data['ADX'].apply(clasificar_fase)
    
    return data

# ═══════════════════════════════════════════════════════════════════════════
# ⚙️ FUNCIÓN 4: REFINADO COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def aplicar_refinado_completo(df):
    """
    🏭 PLANTA DE REFINADO COMPLETA
    """
    
    data = df.copy()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Promedios Móviles
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Bandas de Bollinger
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (std * 2)
    
    # Volatilidad
    data['Volatility'] = (data['High'] - data['Low']) / data['Close']
    
    # 🛡️ PARCHE FOREX
    if 'Volume' in data.columns:
        vol_sum = data['Volume'].sum()
        
        if vol_sum > 0:
            vol_mean = data['Volume'].rolling(20).mean()
            data['Volume_Norm'] = data['Volume'] / vol_mean.replace(0, 1)
        else:
            data['Volume_Norm'] = 0.0
            st.info("ℹ️ Activo sin volumen (Forex). Volume_Norm = 0")
    else:
        data['Volume_Norm'] = 0.0
    
    # Retornos
    data['Returns'] = data['Close'].pct_change()
    
    # ✅ TYPE SAFETY: Solo columnas numéricas
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].astype('float32')
    
    return data

# ═══════════════════════════════════════════════════════════════════════════
# 🎯 FUNCIÓN 5: CREACIÓN DEL TARGET
# ═══════════════════════════════════════════════════════════════════════════

def crear_target(df):
    """
    🎯 VECTOR DE FUERZA PREDICTIVO
    """
    
    data = df.copy()
    
    # Retorno futuro
    data['Future_Return'] = data['Close'].pct_change().shift(-1)
    
    # Target binario
    data['Target'] = (data['Future_Return'] > 0).astype(int)
    
    # Eliminar última fila
    data = data[:-1]
    
    return data

# ═══════════════════════════════════════════════════════════════════════════
# 🤖 FUNCIÓN 6: ENTRENAMIENTO DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════

def entrenar_modelo(X_train, y_train, params, device='cuda'):
    """
    🤖 CONTROLADOR INTELIGENTE (XGBoost Optimizado)
    """
    
    # Device guard
    if device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                st.warning("⚠️ GPU no disponible. Cambiando a CPU.")
                device = 'cpu'
        except ImportError:
            st.warning("⚠️ PyTorch no instalado. Usando CPU.")
            device = 'cpu'
    
    # Configuración optimizada
    model = xgb.XGBClassifier(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        tree_method='hist',
        device=device,
        max_bin=256,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    return model

# ═══════════════════════════════════════════════════════════════════════════
# 💰 FUNCIÓN 7: BACKTESTING (FIX CRÍTICO DE DTYPE)
# ═══════════════════════════════════════════════════════════════════════════

def simular_trading_adaptativo(df, predictions, bloquear_turbulencia=False, capital_inicial=10000):
    """
    💰 SIMULADOR CON GESTIÓN ADAPTATIVA DE RIESGO
    
    🔧 FIX v3.2: Declaración explícita de tipos de datos para evitar
    el error "Invalid value for dtype 'int64'"
    
    PROBLEMA:
    - Pandas infiere dtype al crear columnas con valores iniciales
    - Si usamos int (ej: Position=0), la columna se crea como int64
    - Luego al asignar float (ej: Equity=10092.13), crash
    
    SOLUCIÓN:
    - Declarar explícitamente dtype=float64 para columnas numéricas
    - Usar .astype() para forzar el tipo correcto
    """
    
    data = df.copy()
    
    # ────────────────────────────────────────────────────────────────────────
    # 🔧 FIX CRÍTICO: Declaración explícita de tipos
    # ────────────────────────────────────────────────────────────────────────
    
    # Predicciones (hacer copia para evitar SettingWithCopyWarning)
    data['Prediction_Raw'] = predictions.copy()
    data['Prediction'] = predictions.copy()
    
    # Filtro de turbulencia
    if bloquear_turbulencia:
        mask_turbulento = data['Fase_Mercado'] == 'TURBULENTO'
        data.loc[mask_turbulento, 'Prediction'] = 0
        
        n_bloqueados = mask_turbulento.sum()
        if n_bloqueados > 0:
            st.warning(f"🚫 {n_bloqueados} señales bloqueadas por turbulencia")
    
    # ✅ CREAR COLUMNAS CON DTYPE EXPLÍCITO (float64)
    # Esto previene el error de asignación de float a columna int64
    data['Position'] = 0.0  # float64 en lugar de int
    data['Equity'] = float(capital_inicial)  # float64 explícito
    
    # Variables de simulación
    capital = float(capital_inicial)
    shares = 0.0  # float en lugar de int
    
    # ────────────────────────────────────────────────────────────────────────
    # 💵 LOOP DE SIMULACIÓN
    # ────────────────────────────────────────────────────────────────────────
    
    for i in range(1, len(data)):
        # COMPRAR
        if data['Prediction'].iloc[i] == 1 and shares == 0:
            shares = capital / float(data['Close'].iloc[i])
            data.loc[data.index[i], 'Position'] = 1.0
            
        # VENDER
        elif data['Prediction'].iloc[i] == 0 and shares > 0:
            capital = shares * float(data['Close'].iloc[i])
            shares = 0.0
            data.loc[data.index[i], 'Position'] = 0.0
        
        # Actualizar equity (ahora no hay conflicto de tipos)
        if shares > 0:
            equity_value = shares * float(data['Close'].iloc[i])
            data.loc[data.index[i], 'Equity'] = equity_value
        else:
            data.loc[data.index[i], 'Equity'] = capital
    
    return data

# ═══════════════════════════════════════════════════════════════════════════
# 🎛️ SIDEBAR: PANEL DE CONTROL
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🎛️ Centro de Control v3.2")
    
    st.subheader("📡 Configuración de Datos")
    ticker = st.text_input("Ticker", value="AAPL", help="Ej: AAPL, TSLA, EURUSD=X")
    periodo_dias = st.slider("Período (días)", 180, 1095, 365, 30)
    
    st.subheader("⚙️ Hiperparámetros")
    max_depth = st.slider("Max Depth", 3, 10, 5)
    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
    n_estimators = st.slider("N° Árboles", 50, 500, 100, 50)
    
    st.subheader("🖥️ Dispositivo")
    device_option = st.radio(
        "Procesador",
        ["GPU (CUDA)", "CPU"],
        index=1,
        help="RTX 3050: Usar GPU | CPU: Modo seguro"
    )
    device = 'cuda' if device_option == "GPU (CUDA)" else 'cpu'
    
    st.subheader("🛡️ Gestión de Riesgo")
    bloquear_turbulencia = st.checkbox(
        "🚫 Bloquear operaciones en Turbulencia",
        value=True,
        help="No operar cuando ADX < 20"
    )
    
    st.subheader("📊 Validación")
    test_size = st.slider("Test Size (%)", 10, 40, 20)
    
    st.markdown("---")
    ejecutar = st.button("🚀 INICIAR REFINERÍA", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 🏭 PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

if ejecutar:
    
    # ETAPA 1: Descarga
    df_activo = descargar_datos(ticker, periodo_dias)
    
    # ETAPA 2: Escáner macro
    df_macro, nombre_bench, corr_bench = escanear_benchmarks(df_activo, periodo_dias)
    
    # ETAPA 3: Fase de mercado
    df_fase = calcular_fase_mercado(df_macro)
    
    # ETAPA 4: Refinado
    df_refined = aplicar_refinado_completo(df_fase)
    
    # ETAPA 5: Target
    df_target = crear_target(df_refined)
    df_clean = df_target.dropna()
    
    # ETAPA 6: Features
    feature_cols = [
        'RSI', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower',
        'Volatility', 'Volume_Norm', 'Returns', 'ADX'
    ]
    
    if 'Relative_Strength_Macro' in df_clean.columns:
        feature_cols.extend(['Relative_Strength_Macro', 'Rolling_Corr'])
    
    X = df_clean[feature_cols].values
    y = df_clean['Target'].values
    
    # División temporal
    split_idx = int(len(X) * (1 - test_size/100))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype('float32')
    X_test_scaled = scaler.transform(X_test).astype('float32')
    
    st.success(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")
    
    # ETAPA 7: Entrenamiento
    params = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators
    }
    
    with st.spinner(f"🤖 Entrenando en {device.upper()}..."):
        model = entrenar_modelo(X_train_scaled, y_train, params, device)
    
    st.success("✅ Modelo entrenado")
    
    # ETAPA 8: Predicciones
    y_pred_test = model.predict(X_test_scaled)
    
    df_backtest = df_clean.iloc[split_idx:].copy()
    df_results = simular_trading_adaptativo(
        df_backtest, 
        y_pred_test, 
        bloquear_turbulencia
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # 📊 INTERFAZ MULTI-PESTAÑA
    # ═══════════════════════════════════════════════════════════════════════
    
    tab1, tab2, tab3 = st.tabs([
        "📊 TABLERO DE MANDO",
        "🔬 CAJA NEGRA (XAI)",
        "📁 DATOS CRUDOS"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # PESTAÑA 1: TABLERO DE MANDO
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.header("📊 Panel Principal de Control")
        
        # KPIs Superiores
        precio_actual = df_results['Close'].iloc[-1]
        adx_actual = df_results['ADX'].iloc[-1]
        fase_actual = df_results['Fase_Mercado'].iloc[-1]
        pred_manana = "🟢 ALCISTA" if df_results['Prediction'].iloc[-1] == 1 else "🔴 BAJISTA"
        
        if fase_actual == "LAMINAR":
            fase_emoji = "🌊"
        elif fase_actual == "TURBULENTO":
            fase_emoji = "🌀"
        else:
            fase_emoji = "⚡"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💵 Precio Actual",
                f"${precio_actual:.2f}",
                delta=f"{df_results['Returns'].iloc[-1]*100:.2f}%"
            )
        
        with col2:
            st.metric(
                f"{fase_emoji} Fase del Mercado",
                fase_actual,
                delta=f"ADX: {adx_actual:.1f}"
            )
        
        with col3:
            if nombre_bench:
                st.metric(
                    "🔗 Benchmark Dominante",
                    nombre_bench,
                    delta=f"Corr: {corr_bench:.3f}"
                )
            else:
                st.metric("🔗 Benchmark", "Modo Autónomo", delta="N/A")
        
        with col4:
            st.metric("🎯 Predicción Mañana", pred_manana)
        
        st.markdown("---")
        
        # Gráfico principal
        fig_main = go.Figure()
        
        fig_main.add_trace(
            go.Scatter(
                x=df_results.index,
                y=df_results['Close'],
                name='Precio',
                line=dict(color='white', width=2)
            )
        )
        
        fig_main.add_trace(
            go.Scatter(
                x=df_results.index,
                y=df_results['BB_Upper'],
                name='BB Superior',
                line=dict(color='gray', width=1, dash='dot')
            )
        )
        fig_main.add_trace(
            go.Scatter(
                x=df_results.index,
                y=df_results['BB_Lower'],
                name='BB Inferior',
                line=dict(color='gray', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            )
        )
        
        # Coloreado de fondo
        for i in range(len(df_results)):
            if df_results['Prediction'].iloc[i] == 1:
                fig_main.add_vrect(
                    x0=df_results.index[i],
                    x1=df_results.index[min(i+1, len(df_results)-1)],
                    fillcolor="rgba(0,255,0,0.1)",
                    layer="below",
                    line_width=0
                )
            else:
                fig_main.add_vrect(
                    x0=df_results.index[i],
                    x1=df_results.index[min(i+1, len(df_results)-1)],
                    fillcolor="rgba(255,0,0,0.1)",
                    layer="below",
                    line_width=0
                )
        
        fig_main.update_layout(
            title="📈 Precio con Señales del Modelo",
            xaxis_title="Fecha",
            yaxis_title="Precio ($)",
            template='plotly_dark',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Diagnóstico
        diagnostico = f"""
        🔍 **DIAGNÓSTICO DEL SISTEMA:**
        
        El activo **{ticker}** está en régimen **{fase_actual}** (ADX={adx_actual:.1f}).
        """
        
        if nombre_bench:
            diagnostico += f"""
        
        Acoplamiento con **{nombre_bench}** (correlación: {corr_bench:.3f}).
        """
        
        if fase_actual == "TURBULENTO":
            diagnostico += """
        
        ⚠️ **ADVERTENCIA:** Mercado turbulento. Señales menos confiables.
        """
            if bloquear_turbulencia:
                diagnostico += " Filtro de seguridad ACTIVO."
        
        if fase_actual == "LAMINAR":
            diagnostico += """
        
        ✅ **FAVORABLE:** Tendencia clara detectada.
        """
        
        st.info(diagnostico)
        
        # Curva de Equity
        fig_equity = go.Figure()
        
        fig_equity.add_trace(
            go.Scatter(
                x=df_results.index,
                y=df_results['Equity'],
                name='Equity',
                line=dict(color='cyan', width=3),
                fill='tozeroy',
                fillcolor='rgba(0,255,255,0.1)'
            )
        )
        
        fig_equity.add_hline(
            y=10000,
            line_dash="dash",
            line_color="white",
            annotation_text="Capital Inicial"
        )
        
        fig_equity.update_layout(
            title="💰 Curva de Equity",
            xaxis_title="Fecha",
            yaxis_title="Capital ($)",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Métricas
        capital_final = df_results['Equity'].iloc[-1]
        retorno_total = ((capital_final - 10000) / 10000) * 100
        accuracy = (y_pred_test == y_test).mean() * 100
        n_trades = df_results['Position'].diff().abs().sum() / 2
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💵 Capital Final", f"${capital_final:,.2f}")
        with col2:
            st.metric("📈 Retorno Total", f"{retorno_total:.2f}%")
        with col3:
            st.metric("🎯 Accuracy", f"{accuracy:.2f}%")
        with col4:
            st.metric("📊 N° Operaciones", f"{int(n_trades)}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PESTAÑA 2: CAJA NEGRA
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.header("🔬 Análisis Interno del Modelo")
        
        st.subheader("📊 Importancia de Variables")
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig_importance = go.Figure(
            go.Bar(
                x=feature_importance_df['Importance'],
                y=feature_importance_df['Feature'],
                orientation='h',
                marker=dict(
                    color=feature_importance_df['Importance'],
                    colorscale='Viridis'
                )
            )
        )
        
        fig_importance.update_layout(
            title="¿Qué variables pesan más?",
            xaxis_title="Importancia",
            yaxis_title="Variable",
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.info("""
        **Interpretación:**
        - Variables macro altas → Benchmark influyente
        - ADX alto → Régimen decisivo
        - RSI alto → Señales internas relevantes
        """)
        
        st.subheader("🎯 Matriz de Confusión")
        
        cm = confusion_matrix(y_test, y_pred_test)
        
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=['Predicción: BAJA', 'Predicción: SUBE'],
                y=['Real: BAJA', 'Real: SUBE'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20}
            )
        )
        
        fig_cm.update_layout(
            title="Matriz de Confusión",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
        
        with st.expander("📋 Reporte de Clasificación"):
            report = classification_report(
                y_test,
                y_pred_test,
                target_names=['BAJA', 'SUBE'],
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose())
    
    # ═══════════════════════════════════════════════════════════════════════
    # PESTAÑA 3: DATOS CRUDOS
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab3:
        st.header("📁 Inspección de Datos")
        
        st.subheader("🔍 DataFrame (Últimas 100 filas)")
        st.dataframe(df_results.tail(100))
        
        st.subheader("📊 Estadísticas Descriptivas")
        st.dataframe(df_results[feature_cols].describe())
        
        st.subheader("📥 Descargar Resultados")
        csv = df_results.to_csv()
        st.download_button(
            label="💾 Descargar CSV",
            data=csv,
            file_name=f"{ticker}_resultados.csv",
            mime="text/csv"
        )

# ═══════════════════════════════════════════════════════════════════════════
# 📚 DOCUMENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════

with st.expander("📚 Manual de Usuario v3.2"):
    st.markdown("""
    ## 🔧 Hotfix v3.2
    
    ### ✅ Problema Resuelto
    
    **Error:**
```
    TypeError: Invalid value '10092.138417100188' for dtype 'int64'
```
    
    **Causa:**
    - Pandas creaba la columna 'Equity' como `int64` (valor inicial = 10000)
    - Al asignar valores float (ej: 10092.138), causaba conflicto de tipos
    
    **Solución:**
    - Declarar explícitamente `dtype=float64` para columnas numéricas
    - Usar `float()` en valores iniciales: `data['Equity'] = float(10000)`
    - Convertir shares y capital a float también
    
    ### 📊 Uso Recomendado
    
    1. Empezar con valores default
    2. Verificar que no hay errores de dtype
    3. Activar GPU solo si es estable
    4. Revisar Feature Importance
    
    ### ⚠️ Advertencias
    
    - Accuracy < 52% → Modelo no supera azar
    - Sistema EDUCATIVO, no para trading real
    """)

st.markdown("---")
st.caption("🏭 The Quant Refinery v3.2 | Fix de Dtype | Optimizado para RTX 3050")