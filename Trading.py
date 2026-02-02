"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🏭 THE QUANT REFINERY v3.4 🏭                              ║
║              Sistema de Trading Algorítmico | Velas Japonesas                ║
║                                                                              ║
║  CHANGELOG:                                                                  ║
║  v3.4 → Candlestick (OHLC) + subplot RSI / Volumen                        ║
║  v3.2 → Fix dtype int64→float64 en columna Equity                          ║
║  v3.1 → Type Safety, max_bin=256 GPU, Pandas 2.0 syntax                    ║
║                                                                              ║
║  ANALOGÍA INGENIERIL:                                                        ║
║  - Precios OHLCV   = Materia Prima Cruda (sin procesar)                     ║
║  - Velas Japonesas = Diagrama de Fase del Material (T vs P)                ║
║  - Indicadores     = Procesos de Refinado (RSI, SMA, Bollinger)            ║
║  - Target          = Vector de Fuerza Predicho (Dirección)                 ║
║  - XGBoost         = Planta de Procesamiento Inteligente                    ║
║  - Gestión Riesgo  = Factor de Seguridad (Safety Factor)                   ║
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
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# 🎨 CONFIGURACIÓN DE LA INTERFAZ GRÁFICA (SCADA STYLE)
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🏭 The Quant Refinery",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 The Quant Refinery")
st.markdown("### *Sistema de Trading Algorítmico con Aprendizaje Automático*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# 📊 FUNCIÓN 1: OBTENCIÓN DE MATERIA PRIMA (DATOS DE MERCADO)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def obtener_materia_prima(ticker, periodo_dias=365):
    """
    🏭 PROCESO DE EXTRACCIÓN DE MATERIA PRIMA (CORREGIDO)
    """
    try:
        st.info(f"📡 Descargando datos de {ticker}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=periodo_dias)
        
        # Descargar datos reales
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError("Datos vacíos")
            
        # 🔧 PARCHE DE INGENIERÍA: Aplanar MultiIndex (El problema actual)
        # Si yfinance nos da columnas tipo ('Close', 'BTC-USD'), nos quedamos solo con 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # ⚡ OPTIMIZACIÓN DE MEMORIA: float64 → float32
        df = df.astype('float32')
        
        st.success(f"✅ {len(df)} días de datos obtenidos exitosamente")
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Error al descargar datos reales: {e}")
        st.info("🔧 Generando datos sintéticos para demostración...")
        
        # MODO DEMO: Generar datos sintéticos
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
        
        df = df.astype('float32')
        
        st.success("✅ Datos sintéticos generados (MODO DEMO)")
        return df
    
# ═══════════════════════════════════════════════════════════════════════════
# ⚙️ FUNCIÓN 2: PROCESOS DE REFINADO (CÁLCULO DE INDICADORES TÉCNICOS)
# ═══════════════════════════════════════════════════════════════════════════

def aplicar_refinado(df):
    """
    🏭 PLANTA DE REFINADO (VERSIÓN BLINDADA v2)
    Corrección: Manejo de activos sin volumen (Forex/Índices)
    """
    
    data = df.copy()
    
    # ────────────────────────────────────────────────────────────────────────
    # 🔥 REACTOR 1: RSI (Índice de Fuerza Relativa)
    # ────────────────────────────────────────────────────────────────────────
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # ────────────────────────────────────────────────────────────────────────
    # 📊 REACTOR 2: PROMEDIOS MÓVILES
    # ────────────────────────────────────────────────────────────────────────
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # ────────────────────────────────────────────────────────────────────────
    # 📏 REACTOR 3: BANDAS DE BOLLINGER
    # ────────────────────────────────────────────────────────────────────────
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (std * 2)
    
    # ────────────────────────────────────────────────────────────────────────
    # 📈 FEATURES ADICIONALES (CON BYPASS DE VOLUMEN)
    # ────────────────────────────────────────────────────────────────────────
    
    # Volatilidad
    data['Volatility'] = (data['High'] - data['Low']) / data['Close']
    
    # Retorno porcentual
    data['Returns'] = data['Close'].pct_change()

    # 🔧 FIX DE INGENIERÍA: BYPASS PARA FOREX/INDICES
    # Si el volumen es 0 o no existe, ponemos 0 en lugar de calcular división por cero
    if 'Volume' in data.columns and data['Volume'].sum() > 0:
        # Evitar división por cero reemplazando ceros con 1 temporalmente
        vol_mean = data['Volume'].rolling(20).mean().replace(0, 1)
        data['Volume_Norm'] = data['Volume'] / vol_mean
    else:
        # Si no hay flujo (Forex), asignamos valor neutro
        data['Volume_Norm'] = 0.0
    
    # Limpieza final de infinitos (por si acaso)
    data = data.replace([np.inf, -np.inf], np.nan)
    
    return data.astype('float32')

# ═══════════════════════════════════════════════════════════════════════════
# 🎯 FUNCIÓN 3: CREACIÓN DEL TARGET (VARIABLE OBJETIVO)
# ═══════════════════════════════════════════════════════════════════════════

def crear_target(df):
    """
    🎯 DEFINICIÓN DEL VECTOR DE FUERZA (TARGET)
    
    ⚠️⚠️⚠️ CRÍTICO: PREVENCIÓN DE LOOK-AHEAD BIAS ⚠️⚠️⚠️
    
    PRINCIPIO DE CAUSALIDAD (como en termodinámica):
    - No podemos usar información del FUTURO para predecir el PRESENTE
    - Usamos .shift(-1) para alinear correctamente las etiquetas
    
    ANALOGÍA FÍSICA:
    - Target = 1 → Vector de Fuerza POSITIVO (precio sube mañana)
    - Target = 0 → Vector de Fuerza NEGATIVO (precio baja mañana)
    
    PROCESO:
    1. Calcular retorno del DÍA SIGUIENTE: (Precio_t+1 - Precio_t) / Precio_t
    2. Si retorno > 0 → Target = 1 (COMPRAR)
    3. Si retorno ≤ 0 → Target = 0 (VENDER/NO COMPRAR)
    """
    
    data = df.copy()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 🚨 ZONA CRÍTICA: ALINEACIÓN TEMPORAL CORRECTA
    # ═══════════════════════════════════════════════════════════════════════
    # 
    # SIN .shift(-1):  [INCORRECTO - Look-ahead bias]
    # Día 1: Precio=100 → Retorno=(105-100)/100=+5% → Target=1
    # ¡Estamos usando el precio del día 2 (105) para etiquetar el día 1!
    #
    # CON .shift(-1):  [CORRECTO - Sin look-ahead bias]
    # Día 1: Precio=100 → Target=1 (porque día 2 sube)
    # Día 2: Precio=105 → Target=0 (porque día 3 baja)
    # Ahora el target del día 1 refleja lo que REALMENTE pasó después
    # ═══════════════════════════════════════════════════════════════════════
    
    # Calcular retorno del siguiente período
    data['Future_Return'] = data['Close'].pct_change().shift(-1)
    
    # Clasificación binaria: 1=Sube, 0=Baja
    data['Target'] = (data['Future_Return'] > 0).astype(int)
    
    # Eliminar última fila (no tiene target válido)
    data = data[:-1]
    
    return data

# ═══════════════════════════════════════════════════════════════════════════
# 🤖 FUNCIÓN 4: ENTRENAMIENTO DEL MODELO (PLANTA DE PROCESAMIENTO)
# ═══════════════════════════════════════════════════════════════════════════

def entrenar_modelo(X_train, y_train, params, device='cuda'):
    """
    🏭 PLANTA DE PROCESAMIENTO INTELIGENTE (XGBoost)
    
    ANALOGÍA:
    - XGBoost = Red de Reactores en Cascada (Gradient Boosting)
    - Cada árbol = Etapa de destilación que corrige errores de la anterior
    - GPU = Procesamiento paralelo masivo (como reactores en paralelo)
    
    PARÁMETROS CLAVE:
    - max_depth: Profundidad del reactor (complejidad del modelo)
    - learning_rate: Velocidad de ajuste (η en optimización)
    - n_estimators: Número de etapas de procesamiento
    - tree_method='hist': Algoritmo eficiente para GPU
    - device='cuda': Usar GPU para acelerar cálculos
    """
    
    # Configurar modelo XGBoost con soporte GPU
    model = xgb.XGBClassifier(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        tree_method='hist',  # Algoritmo optimizado para GPU
        device=device,       # 'cuda' o 'cpu'
        random_state=42,
        eval_metric='logloss'
    )
    
    # Entrenar modelo
    model.fit(X_train, y_train, verbose=False)
    
    return model

# ═══════════════════════════════════════════════════════════════════════════
# 📈 FUNCIÓN 5: BACKTESTING (SIMULACIÓN DE PLANTA)
# ═══════════════════════════════════════════════════════════════════════════

def simular_trading(df, predictions, capital_inicial=10000):
    """
    💰 SIMULADOR DE EQUITY (CURVA DE CAPITAL) — Fix v3.2
    
    🔧 BUG RESUELTO: En Pandas 2.x, si una columna se crea con un entero
    (ej: Position=0), su dtype se fija como int64. Luego al asignar un float
    en el loop, lanza: TypeError: Invalid value 'X.XX' for dtype 'int64'.
    
    SOLUCIÓN: Usar 0.0 y float() para que Pandas infiera float64 desde inicio.
    
    ANALOGÍA: Es como preparar un molde de fundición. Si el molde está hecho
    para int, no puedes verter float sin romperlo. Preparamos el molde correcto.
    """
    
    data = df.copy()
    data['Prediction'] = predictions

    # ✅ FIX v3.2: Declarar columnas con dtype float64 desde el principio
    data['Position'] = 0.0                  # float64, NO int64
    data['Equity']   = float(capital_inicial)  # float64 explícito
    
    capital = float(capital_inicial)  # float
    shares  = 0.0                     # float
    
    for i in range(1, len(data)):
        # COMPRAR: predicción = 1 y sin posición abierta
        if data['Prediction'].iloc[i] == 1 and shares == 0.0:
            shares = capital / float(data['Close'].iloc[i])
            data.loc[data.index[i], 'Position'] = 1.0
            
        # VENDER: predicción = 0 y con posición abierta
        elif data['Prediction'].iloc[i] == 0 and shares > 0.0:
            capital = shares * float(data['Close'].iloc[i])
            shares  = 0.0
            data.loc[data.index[i], 'Position'] = 0.0
        
        # Actualizar equity (valor de cartera en ese instante)
        if shares > 0.0:
            data.loc[data.index[i], 'Equity'] = shares * float(data['Close'].iloc[i])
        else:
            data.loc[data.index[i], 'Equity'] = capital
    
    return data

# ═══════════════════════════════════════════════════════════════════════════
# 🎛️ SIDEBAR: PANEL DE CONTROL (SCADA STYLE)
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🎛️ Panel de Control")
    
    # ───────────────────────────────────────────────────────────────────────
    # 📡 SECCIÓN 1: CONFIGURACIÓN DE DATOS
    # ───────────────────────────────────────────────────────────────────────
    st.subheader("📡 Fuente de Datos")
    ticker = st.text_input("Ticker (símbolo)", value="AAPL", 
                           help="Ejemplo: AAPL, TSLA, MSFT, BTC-USD")
    periodo_dias = st.slider("Período histórico (días)", 
                             min_value=180, max_value=1095, value=365, step=30)
    
    # ───────────────────────────────────────────────────────────────────────
    # ⚙️ SECCIÓN 2: HIPERPARÁMETROS DEL MODELO
    # ───────────────────────────────────────────────────────────────────────
    st.subheader("⚙️ Hiperparámetros XGBoost")
    
    max_depth = st.slider("Max Depth (profundidad)", 
                          min_value=3, max_value=10, value=5,
                          help="Profundidad máxima de cada árbol")
    
    learning_rate = st.slider("Learning Rate (η)", 
                              min_value=0.01, max_value=0.3, value=0.1, step=0.01,
                              help="Velocidad de aprendizaje")
    
    n_estimators = st.slider("N° de Árboles", 
                             min_value=50, max_value=500, value=100, step=50,
                             help="Número de árboles en el ensamble")
    
    # ───────────────────────────────────────────────────────────────────────
    # 🖥️ SECCIÓN 3: SELECTOR DE DISPOSITIVO (GPU/CPU)
    # ───────────────────────────────────────────────────────────────────────
    st.subheader("🖥️ Dispositivo de Cómputo")
    device_option = st.radio(
        "Procesador",
        options=["GPU (CUDA)", "CPU"],
        index=0,
        help="GPU: RTX 3060/2070 | CPU: Procesador convencional"
    )
    device = 'cuda' if device_option == "GPU (CUDA)" else 'cpu'
    
    # ───────────────────────────────────────────────────────────────────────
    # 📊 SECCIÓN 4: DIVISIÓN TRAIN/TEST
    # ───────────────────────────────────────────────────────────────────────
    st.subheader("📊 Validación Temporal")
    test_size = st.slider("Test Size (%)", 
                          min_value=10, max_value=40, value=20,
                          help="Porcentaje de datos para prueba (temporalmente posteriores)")
    
    # ───────────────────────────────────────────────────────────────────────
    # 🚀 BOTÓN DE EJECUCIÓN
    # ───────────────────────────────────────────────────────────────────────
    st.markdown("---")
    ejecutar = st.button("🚀 EJECUTAR REFINERÍA", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 🏭 PIPELINE PRINCIPAL DE EJECUCIÓN
# ═══════════════════════════════════════════════════════════════════════════

if ejecutar:
    
    # ═══════════════════════════════════════════════════════════════════════
    # ETAPA 1: OBTENCIÓN DE MATERIA PRIMA
    # ═══════════════════════════════════════════════════════════════════════
    with st.spinner("🏭 Extrayendo materia prima..."):
        df_raw = obtener_materia_prima(ticker, periodo_dias)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ETAPA 2: REFINADO (CÁLCULO DE INDICADORES)
    # ═══════════════════════════════════════════════════════════════════════
    with st.spinner("⚙️ Procesando en planta de refinado..."):
        df_refined = aplicar_refinado(df_raw)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ETAPA 3: CREACIÓN DEL TARGET
    # ═══════════════════════════════════════════════════════════════════════
    with st.spinner("🎯 Generando vector objetivo..."):
        df_target = crear_target(df_refined)
    
    # Eliminar valores nulos generados por rolling windows
    df_clean = df_target.dropna()
    
    # ═══════════════════════════════════════════════════════════════════════
    # ETAPA 4: PREPARACIÓN DE DATOS (DIVISIÓN TEMPORAL ESTRICTA)
    # ═══════════════════════════════════════════════════════════════════════
    # 
    # ⚠️ CRÍTICO: NO USAR shuffle=True (violaría causalidad temporal)
    # Usar corte cronológico: [Pasado → Train] | [Futuro → Test]
    # 
    # ═══════════════════════════════════════════════════════════════════════
    
    # Definir features (variables de entrada)
    feature_cols = ['RSI', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower', 
                    'Volatility', 'Volume_Norm', 'Returns']
    
    X = df_clean[feature_cols].values
    y = df_clean['Target'].values
    
    # División temporal (sin shuffle)
    split_idx = int(len(X) * (1 - test_size/100))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalización (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype('float32')
    X_test_scaled = scaler.transform(X_test).astype('float32')
    
    st.success(f"✅ Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ETAPA 5: ENTRENAMIENTO DEL MODELO
    # ═══════════════════════════════════════════════════════════════════════
    
    params = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators
    }
    
    with st.spinner(f"🤖 Entrenando modelo en {device.upper()}..."):
        model = entrenar_modelo(X_train_scaled, y_train, params, device)
    
    st.success("✅ Modelo entrenado exitosamente")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ETAPA 6: PREDICCIONES Y BACKTESTING
    # ═══════════════════════════════════════════════════════════════════════
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Reconstruir DataFrame con predicciones
    df_backtest = df_clean.iloc[split_idx:].copy()
    
    # Simular trading
    df_results = simular_trading(df_backtest, y_pred_test)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 📊 VISUALIZACIÓN DE RESULTADOS
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.header("📊 Resultados del Backtesting")
    
    # ───────────────────────────────────────────────────────────────────────
    # GRÁFICO 1: PRECIOS Y SEÑALES DE TRADING
    # ───────────────────────────────────────────────────────────────────────
    
    # ───────────────────────────────────────────────────────────────────────
    # GRÁFICO 1: VELAS JAPONESAS + SEÑALES + RSI + VOLUMEN
    # ───────────────────────────────────────────────────────────────────────
    # 
    # 🕯️ ANALOGÍA — DIAGRAMA DE FASE (como en Metalurgia):
    #   Cada vela es un período de tiempo (1 día).
    #   - Cuerpo verde  → precio cerró MÁS ALTO que abrió  (solidificación estable)
    #   - Cuerpo rojo   → precio cerró MÁS BAJO que abrió  (enfriamiento brusco)
    #   - Sombra (mecha)→ el rango extremo que tocó el precio sin cerrar ahí
    #
    #   Es exactamente como un diagrama T-P: el cuerpo de la vela te dice
    #   en qué fase quedó el sistema al final del período, y las mechas
    #   te dicen hasta qué fase transitó durante ese tiempo.
    # ───────────────────────────────────────────────────────────────────────

    fig_signals = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            '🕯️ Velas Japonesas + Señales de Trading',
            '📊 RSI (Índice de Fuerza Relativa)',
            '📦 Volumen'
        ),
        row_heights=[0.55, 0.25, 0.20]
    )

    # ── ROW 1: CANDLESTICK ─────────────────────────────────────────────────
    fig_signals.add_trace(
        go.Candlestick(
            x=df_results.index,
            open =df_results['Open'],
            high =df_results['High'],
            low  =df_results['Low'],
            close=df_results['Close'],
            name='Precio (OHLC)',
            increasing=dict(line=dict(color='#26a69a'), fillcolor='#26a69a'),  # verde teal
            decreasing=dict(line=dict(color='#ef5350'), fillcolor='#ef5350')   # rojo coral
        ),
        row=1, col=1
    )

    # ── Bandas de Bollinger (sobre las velas) ──────────────────────────────
    fig_signals.add_trace(
        go.Scatter(
            x=df_results.index, y=df_results['BB_Upper'],
            name='BB Superior',
            line=dict(color='rgba(255,255,255,0.35)', width=1, dash='dot'),
            mode='lines'
        ),
        row=1, col=1
    )
    fig_signals.add_trace(
        go.Scatter(
            x=df_results.index, y=df_results['BB_Lower'],
            name='BB Inferior',
            line=dict(color='rgba(255,255,255,0.35)', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(255,255,255,0.04)',
            mode='lines'
        ),
        row=1, col=1
    )

    # ── Señales de COMPRA (triángulo arriba, sobre el High del día) ────────
    buy_signals = df_results[df_results['Prediction'] == 1]
    fig_signals.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Low'] * 0.995,   # ligeramente bajo el Low → no tapa la vela
            mode='markers',
            name='▲ Señal COMPRA',
            marker=dict(color='#00e676', size=11, symbol='triangle-up',
                        line=dict(color='#fff', width=1))
        ),
        row=1, col=1
    )

    # ── Señales de VENTA (triángulo abajo, bajo el Low del día) ────────────
    sell_signals = df_results[df_results['Prediction'] == 0]
    fig_signals.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals['High'] * 1.005,  # ligeramente sobre el High
            mode='markers',
            name='▼ Señal VENTA',
            marker=dict(color='#ff1744', size=11, symbol='triangle-down',
                        line=dict(color='#fff', width=1))
        ),
        row=1, col=1
    )

    # ── ROW 2: RSI ─────────────────────────────────────────────────────────
    fig_signals.add_trace(
        go.Scatter(
            x=df_results.index, y=df_results['RSI'],
            name='RSI (14)',
            line=dict(color='#ffa726', width=2)
        ),
        row=2, col=1
    )
    # Zonas de referencia RSI
    fig_signals.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.6)",
                          annotation_text="Sobrecompra (70)", row=2, col=1)
    fig_signals.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.6)",
                          annotation_text="Sobreventa (30)",  row=2, col=1)

    # ── ROW 3: VOLUMEN (barras coloreadas por dirección) ──────────────────
    # Verde si Close >= Open (día alcista), Rojo si Close < Open (día bajista)
    vol_colors = [
        '#26a69a' if c >= o else '#ef5350'
        for c, o in zip(df_results['Close'], df_results['Open'])
    ]
    fig_signals.add_trace(
        go.Bar(
            x=df_results.index,
            y=df_results['Volume'],
            name='Volumen',
            marker=dict(color=vol_colors),
            showlegend=True
        ),
        row=3, col=1
    )

    # ── LAYOUT GLOBAL ──────────────────────────────────────────────────────
    fig_signals.update_layout(
        height=780,
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified',
        xaxis3_rangeslider_visible=False   # Ocultar el slider automático de Candlestick
    )
    # Ajustar el eje Y del RSI para que quede entre 0-100
    fig_signals.update_yaxes(range=[0, 100], row=2, col=1)

    st.plotly_chart(fig_signals, use_container_width=True)
    
    # ───────────────────────────────────────────────────────────────────────
    # GRÁFICO 2: CURVA DE EQUITY (CAPITAL)
    # ───────────────────────────────────────────────────────────────────────
    
    fig_equity = go.Figure()
    
    fig_equity.add_trace(
        go.Scatter(x=df_results.index, y=df_results['Equity'],
                   name='Equity Curve', line=dict(color='cyan', width=3),
                   fill='tozeroy', fillcolor='rgba(0,255,255,0.1)')
    )
    
    # Línea de capital inicial
    fig_equity.add_hline(y=10000, line_dash="dash", line_color="white",
                         annotation_text="Capital Inicial: $10,000")
    
    fig_equity.update_layout(
        title="💰 Curva de Equity (Evolución del Capital)",
        xaxis_title="Fecha",
        yaxis_title="Capital ($)",
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # ───────────────────────────────────────────────────────────────────────
    # MÉTRICAS DE RENDIMIENTO
    # ───────────────────────────────────────────────────────────────────────
    
    capital_final = df_results['Equity'].iloc[-1]
    retorno_total = ((capital_final - 10000) / 10000) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💵 Capital Final", f"${capital_final:,.2f}")
    with col2:
        st.metric("📈 Retorno Total", f"{retorno_total:.2f}%")
    with col3:
        accuracy = (y_pred_test == y_test).mean() * 100
        st.metric("🎯 Accuracy", f"{accuracy:.2f}%")
    with col4:
        n_trades = df_results['Position'].diff().abs().sum() / 2
        st.metric("📊 N° Operaciones", f"{int(n_trades)}")
    
    # ───────────────────────────────────────────────────────────────────────
    # MATRIZ DE CONFUSIÓN (ANÁLISIS DE ERRORES)
    # ───────────────────────────────────────────────────────────────────────
    
    st.markdown("---")
    st.subheader("🔍 Matriz de Confusión (Control de Calidad)")
    
    cm = confusion_matrix(y_test, y_pred_test)
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicción: BAJA', 'Predicción: SUBE'],
        y=['Real: BAJA', 'Real: SUBE'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20}
    ))
    
    fig_cm.update_layout(
        title="Matriz de Confusión",
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Explicación de la Matriz
    st.info("""
    **🔬 Interpretación Ingenieril:**
    
    - **True Negatives (TN):** Correctamente identificó que el precio BAJARÍA
    - **False Positives (FP):** ⚠️ FALSA ALARMA - Predijo subida pero bajó (pérdida)
    - **False Negatives (FN):** ⚠️ FALLA NO DETECTADA - Predijo bajada pero subió (oportunidad perdida)
    - **True Positives (TP):** Correctamente identificó que el precio SUBIRÍA
    
    En control de calidad:
    - FP = Producto defectuoso que pasa inspección
    - FN = Producto bueno que es rechazado
    """)
    
    # ───────────────────────────────────────────────────────────────────────
    # REPORTE DE CLASIFICACIÓN
    # ───────────────────────────────────────────────────────────────────────
    
    with st.expander("📋 Reporte Detallado de Métricas"):
        report = classification_report(y_test, y_pred_test, 
                                       target_names=['BAJA', 'SUBE'],
                                       output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    
    # ═══════════════════════════════════════════════════════════════════════
    # 💾 SECCIÓN DE PERSISTENCIA (GUARDAR/CARGAR MODELO)
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("💾 Persistencia del Modelo")
    
    col_save, col_load = st.columns(2)
    
    with col_save:
        if st.button("💾 Guardar Modelo", use_container_width=True):
            try:
                # Guardar modelo en formato JSON (portable)
                model.save_model("quant_refinery_model.json")
                
                # Guardar scaler
                import pickle
                with open("scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)
                
                st.success("✅ Modelo guardado: quant_refinery_model.json")
                st.info("📦 Scaler guardado: scaler.pkl")
                
            except Exception as e:
                st.error(f"❌ Error al guardar: {e}")
    
    with col_load:
        if st.button("📂 Cargar Modelo", use_container_width=True):
            try:
                # Cargar modelo
                loaded_model = xgb.XGBClassifier()
                loaded_model.load_model("quant_refinery_model.json")
                
                # Cargar scaler
                import pickle
                with open("scaler.pkl", "rb") as f:
                    loaded_scaler = pickle.load(f)
                
                st.success("✅ Modelo cargado exitosamente")
                st.info("Modelo listo para predicciones en otra máquina")
                
            except Exception as e:
                st.error(f"❌ Error al cargar: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 📚 SECCIÓN DE AYUDA Y DOCUMENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════

with st.expander("📚 Guía de Uso - The Quant Refinery"):
    st.markdown("""
    ## 🎓 Guía para Estudiantes de Ingeniería
    
    ### 🏭 Analogías del Sistema
    
    | Concepto Financiero | Analogía Ingenieril |
    |---------------------|---------------------|
    | Precios OHLCV | Materia prima cruda (sin procesar) |
    | RSI, SMA, Bollinger | Procesos de refinado / Filtros |
    | Target (Sube/Baja) | Vector de fuerza predicho |
    | XGBoost | Planta de procesamiento inteligente |
    | Gestión de riesgo | Factor de seguridad (FS) |
    | Equity Curve | Eficiencia de la planta vs tiempo |
    
    ### ⚙️ Configuración Recomendada
    
    **Para Laptop (RTX 3060, 16GB RAM):**
    - Período: 365 días
    - Max Depth: 5
    - N° Árboles: 100
    - Dispositivo: GPU (CUDA)
    
    **Para Workstation (RTX 2070S, 32GB RAM):**
    - Período: 1095 días (3 años)
    - Max Depth: 7
    - N° Árboles: 300
    - Dispositivo: GPU (CUDA)
    
    ### 🎯 Interpretación de Resultados
    
    1. **Accuracy > 55%:** Modelo supera azar (50%)
    2. **Retorno Total > 0%:** Estrategia rentable
    3. **Equity Curve ascendente:** Crecimiento sostenido
    4. **Pocos FP (Falsas Alarmas):** Menos pérdidas innecesarias
    
    ### ⚠️ Limitaciones y Advertencias
    
    - ❌ NO usar en trading real sin validación exhaustiva
    - ❌ Resultados pasados NO garantizan resultados futuros
    - ✅ Herramienta EDUCATIVA para aprender ML aplicado
    - ✅ Útil para prototipar ideas de estrategias
    
    ### 🔧 Solución de Problemas
    
    **Error de descarga de datos:**
    → El sistema genera datos sintéticos automáticamente (modo demo)
    
    **Error CUDA (GPU no detectada):**
    → Cambiar a "CPU" en el selector de dispositivo
    
    **Modelo tarda mucho:**
    → Reducir N° de árboles o usar CPU
    """)

st.markdown("---")
st.caption("🏭 The Quant Refinery v3.4 | Velas Japonesas + XGBoost | Desarrollado con Streamlit + ❤️")