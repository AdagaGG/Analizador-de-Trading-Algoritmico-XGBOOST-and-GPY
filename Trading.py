"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🏭  THE QUANT REFINERY  v4.1  —  Unified View  🏭              ║
║                                                                              ║
║  CAMBIO PRINCIPAL v4.1:                                                      ║
║    Eliminación de st.tabs → flujo vertical único.                           ║
║    El módulo "Radar de Mercado" se integra como Sección 4 inline.           ║
║    Descarga multi-ticker con fallback individual por activo.                ║
║                                                                              ║
║  CHANGELOG:                                                                  ║
║    v4.1  Unified View. Sin tabs. Radar integrado inline.                    ║
║    v4.0  Arquitectura dual (Refinería + Radar en tabs).                     ║
║    v3.4  Candlestick + subplot Volumen.                                     ║
║    v3.2  Fix dtype int64 → float64 en Equity.                              ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pandas_ta  →  vendorizada en carpeta local (NO pip install)              ║
║    yfinance, plotly, xgboost, scikit-learn, pandas, numpy                   ║
║                                                                              ║
║  ANALOGÍAS INGENIERILES:                                                     ║
║    Precios OHLCV     →  Materia Prima Cruda                                 ║
║    Velas Japonesas   →  Diagrama de Fase (T vs P)                           ║
║    RSI / BB / MACD   →  Sensores del Proceso de Refinado                    ║
║    Target            →  Vector de Fuerza Predicho                           ║
║    XGBoost           →  Controlador PID Inteligente                         ║
║    Correlación       →  Acoplamiento térmico entre sistemas                 ║
║    Normalización     →  Calibración a origen común (Base 100)               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# BLOQUE A │ IMPORTS & CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

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
import warnings

# pandas_ta vendorizada: existe como carpeta local, no se instala por pip.
import pandas_ta as ta

warnings.filterwarnings("ignore")

# ─── Paleta SCADA ─────────────────────────────────────────────────────────────
CLR_CANDLE_UP   = "#26a69a"   # Verde teal
CLR_CANDLE_DOWN = "#ef5350"   # Rojo coral
CLR_BB          = "rgba(255,255,255,0.30)"
CLR_BB_FILL     = "rgba(255,255,255,0.04)"
CLR_RSI         = "#ffa726"
CLR_MACD        = "#42a5f5"
CLR_MACD_SIG    = "#ab47bc"

# ─── Benchmarks del Radar (lista fija, el ticker del usuario se agrega) ──────
BENCHMARKS_FIJOS = ["SPY", "QQQ", "BTC-USD", "GLD"]

# ─── Colores para el gráfico de rendimiento normalizado ──────────────────────
COLORES_RADAR = ["#ffffff", "#42a5f5", "#66bb6a", "#ffa726", "#ef5350"]
# Posición 0 siempre será el ticker del usuario (blanco, línea gruesa)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏭 The Quant Refinery v4.1",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOQUE B │ CAPA DE DATOS  —  Descarga, limpieza, indicadores, target
# ═══════════════════════════════════════════════════════════════════════════════


def _aplanar_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    🛡️  Parche universal de MultiIndex para yfinance.

    yfinance puede devolver:
      • Columnas simples:  ['Open', 'High', 'Low', 'Close', 'Volume']
      • MultiIndex 1-nivel: [('Close', ''), ('Open', ''), …]
      • MultiIndex 2-niveles (multi-ticker): [('Close','AAPL'), ('Close','SPY')]

    Este parche detecta el caso y devuelve siempre columnas planas strings.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # MultiIndex: tomamos el primer nivel (Price) si tiene 2 niveles
    df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]

    # Eliminar duplicados (puede pasar si yfinance repite columnas)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def _datos_sinteticos(dias: int) -> pd.DataFrame:
    """Fallback: genera OHLCV sintéticos (senoidal + browniano)."""
    idx   = pd.date_range(end=datetime.now(), periods=dias, freq="D")
    trend = np.linspace(100, 150, dias)
    wave  = 20 * np.sin(np.linspace(0, 4 * np.pi, dias))
    noise = np.random.randn(dias) * 3
    close = (trend + wave + noise).astype("float32")

    return pd.DataFrame({
        "Open":   (close * 0.995).astype("float32"),
        "High":   (close * 1.015).astype("float32"),
        "Low":    (close * 0.985).astype("float32"),
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 10_000_000, dias).astype("float32"),
    }, index=idx)


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_ticker(ticker: str, dias: int) -> pd.DataFrame:
    """
    📡 Descarga un solo ticker con fallback a sintéticos.
    Usado para el activo principal Y como fallback individual del Radar.
    """
    end   = datetime.now()
    start = end - timedelta(days=dias)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("respuesta vacía")
        df = _aplanar_multiindex(df)
        # float32 solo en columnas numéricas (Type Safety v3.1)
        num = df.select_dtypes(include=["number"]).columns
        df[num] = df[num].astype("float32")
        return df
    except Exception as exc:
        st.warning(f"⚠️ Descarga fallida para **{ticker}**: {exc} → modo demo.")
        return _datos_sinteticos(dias)


# ─── Indicadores técnicos ─────────────────────────────────────────────────────

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    ⚙️  PLANTA DE REFINADO  —  pandas_ta (vendorizada).

    Indicadores:
        RSI(14)           Sensor de sobrecalentamiento
        Bollinger(20,2)   Límites de control
        MACD(12,26,9)     Oscilador de momentum
        Volume_Norm       Volumen / media-20 (bypass si es 0 → Forex)
        Volatility        Rango diario / Close
        Returns           pct_change diario
    """
    data = df.copy()

    # ── RSI ─────────────────────────────────────────────────────────────────
    rsi = ta.rsi(data["Close"], length=14)
    if rsi is not None:
        data["RSI"] = rsi

    # ── Bollinger ───────────────────────────────────────────────────────────
    bb = ta.bbands(data["Close"], length=20, std=2)
    if bb is not None:
        data["BB_Lower"]  = bb.iloc[:, 0]
        data["BB_Middle"] = bb.iloc[:, 1]
        data["BB_Upper"]  = bb.iloc[:, 2]

    # ── MACD ────────────────────────────────────────────────────────────────
    macd = ta.macd(data["Close"], fast=12, slow=26, signal=9)
    if macd is not None:
        data["MACD"]        = macd.iloc[:, 0]
        data["MACD_Signal"] = macd.iloc[:, 1]
        data["MACD_Hist"]   = macd.iloc[:, 2]

    # ── Volatilidad & Retornos ──────────────────────────────────────────────
    data["Volatility"] = (data["High"] - data["Low"]) / data["Close"]
    data["Returns"]    = data["Close"].pct_change()

    # ── Volume normalizado (bypass Forex) ───────────────────────────────────
    if "Volume" in data.columns and data["Volume"].sum() > 0:
        vol_med = data["Volume"].rolling(20).mean().replace(0, 1)
        data["Volume_Norm"] = data["Volume"] / vol_med
    else:
        data["Volume_Norm"] = 0.0

    # Infinitos → NaN
    data = data.replace([np.inf, -np.inf], np.nan)

    # float32 en numéricos
    num = data.select_dtypes(include=["number"]).columns
    data[num] = data[num].astype("float32")
    return data


# ─── Target + split temporal ──────────────────────────────────────────────────

FEATURE_COLS = [
    "RSI", "BB_Upper", "BB_Middle", "BB_Lower",
    "MACD", "MACD_Signal", "MACD_Hist",
    "Volatility", "Volume_Norm", "Returns",
]


def preparar_dataset(df: pd.DataFrame, test_pct: int = 20):
    """
    🎯  Crea Target con .shift(-1)  →  prevención de Look-Ahead Bias.

    ⚠️  CAUSALIDAD TEMPORAL:
        Target[t] = 1  si  Close[t+1] > Close[t]   (precio sube mañana)
        Target[t] = 0  si  Close[t+1] ≤ Close[t]   (precio baja/plato)

        .shift(-1) alinea las etiquetas sin usar datos del futuro durante
        el entrenamiento.  La última fila se elimina (target inválido).

    Retorna:
        X_train_s, X_test_s, y_train, y_test, scaler, features_usados, df_clean
    """
    data = df.copy()

    # Target
    data["Future_Return"] = data["Close"].pct_change().shift(-1)
    data["Target"]        = (data["Future_Return"] > 0).astype(int)
    data = data.iloc[:-1]  # última fila sin target

    # Solo features que efectivamente existen
    features = [f for f in FEATURE_COLS if f in data.columns]

    # Eliminar NaN en features + target
    data = data.dropna(subset=features + ["Target"])

    X = data[features].values.astype("float32")
    y = data["Target"].values

    # ── Corte temporal ESTRICTO (sin shuffle) ───────────────────────────────
    split = int(len(X) * (1 - test_pct / 100))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Normalización
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype("float32")
    X_test_s  = scaler.transform(X_test).astype("float32")

    return X_train_s, X_test_s, y_train, y_test, scaler, features, data


# ═══════════════════════════════════════════════════════════════════════════════
# BLOQUE C │ CAPA ML  —  XGBoost
# ═══════════════════════════════════════════════════════════════════════════════


def entrenar_modelo(X_train, y_train, params: dict, device: str = "cpu"):
    """
    🤖 Controlador PID Inteligente (XGBoost).
        max_bin=256      →  Optimización VRAM para GPU 4 GB (RTX 3050)
        tree_method=hist →  Algoritmo eficiente
        device           →  'cuda' o 'cpu'; XGBoost cae a CPU si no hay GPU
    """
    model = xgb.XGBClassifier(
        max_depth        = params.get("max_depth", 5),
        learning_rate    = params.get("learning_rate", 0.1),
        n_estimators     = params.get("n_estimators", 100),
        tree_method      = "hist",
        device           = device,
        max_bin          = 256,
        random_state     = 42,
        eval_metric      = "logloss",
        use_label_encoder= False,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# BLOQUE D │ FUNCIONES DE GRÁFICOS  —  Cada una retorna un go.Figure
# ═══════════════════════════════════════════════════════════════════════════════


def fig_candlestick(df: pd.DataFrame) -> go.Figure:
    """
    🕯️  Velas japonesas con Bandas de Bollinger superpuestas.
        xaxis_rangeslider_visible=False  →  sin slider inferior.
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x     = df.index,
        open  = df["Open"],
        high  = df["High"],
        low   = df["Low"],
        close = df["Close"],
        name  = "OHLC",
        increasing = dict(line=dict(color=CLR_CANDLE_UP),   fillcolor=CLR_CANDLE_UP),
        decreasing = dict(line=dict(color=CLR_CANDLE_DOWN), fillcolor=CLR_CANDLE_DOWN),
    ))

    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], mode="lines", name="BB Superior",
            line=dict(color=CLR_BB, width=1, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], mode="lines", name="BB Inferior",
            line=dict(color=CLR_BB, width=1, dash="dot"),
            fill="tonexty", fillcolor=CLR_BB_FILL,
        ))

    fig.update_layout(
        title="🕯️  Análisis de Acción de Precio",
        yaxis_title="Precio",
        template="plotly_dark",
        height=520,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=30, t=50, b=20),
    )
    return fig


def fig_feature_importance(model, feature_names: list) -> go.Figure:
    """📊 Feature Importance — barras horizontales."""
    imp = pd.DataFrame({
        "Feature":     feature_names,
        "Importancia": model.feature_importances_,
    }).sort_values("Importancia", ascending=True)

    fig = go.Figure(go.Bar(
        x=imp["Importancia"],
        y=imp["Feature"],
        orientation="h",
        marker=dict(color=imp["Importancia"].tolist(), colorscale="Viridis"),
    ))
    fig.update_layout(
        title="🧠 Feature Importance",
        xaxis_title="Peso relativo",
        template="plotly_dark",
        height=380,
        margin=dict(l=110, r=20, t=45, b=30),
    )
    return fig


def fig_confusion(y_true, y_pred) -> go.Figure:
    """🎯 Matriz de Confusión — Heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Pred: BAJA ▼", "Pred: SUBE ▲"],
        y=["Real: BAJA ▼", "Real: SUBE ▲"],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=22, color="white"),
    ))
    fig.update_layout(
        title="🎯 Matriz de Confusión",
        template="plotly_dark",
        height=380,
        xaxis=dict(title="Predicción"),
        yaxis=dict(title="Realidad", autorange="reversed"),
        margin=dict(l=80, r=20, t=45, b=40),
    )
    return fig


def fig_correlacion(df_retornos: pd.DataFrame) -> go.Figure:
    """
    🔗 Correlación de retornos diarios (pct_change).
    Se usa pct_change (no precios brutos) para evitar correlaciones espurias.
    """
    corr = df_retornos.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=15),
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        title="🔗 Correlación — Retornos Diarios",
        template="plotly_dark",
        height=440,
        xaxis=dict(tickangle=20),
        margin=dict(l=60, r=40, t=50, b=50),
    )
    return fig


def fig_rendimiento_normalizado(df_closes: pd.DataFrame, ticker_usuario: str) -> go.Figure:
    """
    📈 Rendimiento normalizado Base 100.
        valor_norm[t] = (precio[t] / precio[0]) × 100
    La línea del usuario: blanco, grosor 3.5, opacidad 1.0.
    El resto: color asignado, grosor 1.8, opacidad 0.55.
    """
    df_norm = (df_closes / df_closes.iloc[0]) * 100

    fig = go.Figure()
    cols_ordenados = [ticker_usuario] + [c for c in df_norm.columns if c != ticker_usuario]

    for i, col in enumerate(cols_ordenados):
        if col not in df_norm.columns:
            continue
        es_usuario = (col == ticker_usuario)
        fig.add_trace(go.Scatter(
            x=df_norm.index,
            y=df_norm[col],
            name=col,
            mode="lines",
            line=dict(
                color = COLORES_RADAR[i % len(COLORES_RADAR)],
                width = 3.5 if es_usuario else 1.8,
            ),
            opacity = 1.0 if es_usuario else 0.55,
        ))

    fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.20)",
                  annotation_text="Base 100")

    fig.update_layout(
        title="📈 Rendimiento Relativo (Base 100)",
        xaxis_title="Fecha",
        yaxis_title="Valor normalizado",
        template="plotly_dark",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=55, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BLOQUE E │ SIDEBAR  —  Centro de Control
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎛️ Centro de Control")
    st.markdown("---")

    # ── 📡 Activo ─────────────────────────────────────────────────────────
    st.subheader("📡 Activo")
    ticker = st.text_input(
        "Ticker",
        value="BTC-USD",
        placeholder="AAPL, TSLA, BTC-USD …",
        help="Cualquier símbolo soportado por Yahoo Finance",
    ).strip().upper()

    periodo_dias = st.slider("Días históricos", 180, 1095, 365, step=30)

    st.markdown("---")

    # ── ⚙️ XGBoost ────────────────────────────────────────────────────────
    st.subheader("⚙️ XGBoost")
    max_depth     = st.slider("Max Depth",        3,  10,  5)
    learning_rate = st.slider("Learning Rate (η)", 0.01, 0.30, 0.10, step=0.01)
    n_estimators  = st.slider("N° de Árboles",   50, 500, 100, step=50)
    test_size     = st.slider("Test Size (%)",   10,  40,  20)

    st.markdown("---")

    # ── 🖥️ Dispositivo ────────────────────────────────────────────────────
    st.subheader("🖥️ Dispositivo")
    device_opt = st.radio("Procesador", ["CPU", "GPU (CUDA)"], index=0,
                          help="GPU requiere driver CUDA compatible")
    device = "cuda" if "GPU" in device_opt else "cpu"

    st.markdown("---")

    # ── 🚀 Botón ──────────────────────────────────────────────────────────
    ejecutar = st.button("🚀 EJECUTAR REFINERÍA", type="primary",
                         use_container_width=True)

    st.markdown("---")
    st.caption("🏭 The Quant Refinery v4.1 — Unified View")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOQUE F │ PIPELINE PRINCIPAL  —  Flujo vertical único, sin tabs
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🏭 The Quant Refinery v4.1")
st.markdown("*Sistema Adaptativo de Trading Algorítmico — Unified View*")
st.markdown("---")

if not ejecutar:
    st.info("👆 Configura los parámetros en el sidebar y pulsa **EJECUTAR REFINERÍA**.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#   SECCIÓN 1 │ PROCESAMIENTO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner("📡 Descargando datos del activo …"):
    df_raw = descargar_ticker(ticker, periodo_dias)

with st.spinner("⚙️ Calculando indicadores técnicos …"):
    df = calcular_indicadores(df_raw)

with st.spinner("🎯 Preparando dataset para ML …"):
    (X_train, X_test, y_train, y_test,
     scaler, features_usados, df_clean) = preparar_dataset(df, test_size)

c1, c2, c3 = st.columns(3)
c1.metric("📦 Datos totales",  str(len(df)))
c2.metric("🏋️ Train",          str(len(y_train)))
c3.metric("🧪 Test",           str(len(y_test)))


# ═══════════════════════════════════════════════════════════════════════════════
#   SECCIÓN 2 │ VISUALIZACIÓN DE PRECIO  —  Velas + Bollinger
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.plotly_chart(fig_candlestick(df), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#   SECCIÓN 3 │ DIAGNÓSTICO DEL MODELO  —  XGBoost
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🤖 Diagnóstico de IA (XGBoost)")

with st.spinner(f"🤖 Entrenando modelo en {device.upper()} …"):
    params = {
        "max_depth":     max_depth,
        "learning_rate": learning_rate,
        "n_estimators":  n_estimators,
    }
    model = entrenar_modelo(X_train, y_train, params, device)

y_pred   = model.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100

# KPIs del modelo
m1, m2, m3 = st.columns(3)
m1.metric("🎯 Accuracy",  f"{accuracy:.2f} %")
m2.metric("🌳 Árboles",   str(n_estimators))
m3.metric("📐 Max Depth", str(max_depth))

# ── 3 columnas: Reporte │ Confusión │ Feature Importance ────────────────────
col_rep, col_cm, col_fi = st.columns(3)

with col_rep:
    st.markdown("#### 📋 Reporte de Clasificación")
    report = classification_report(
        y_test, y_pred,
        target_names=["BAJA ▼", "SUBE ▲"],
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).T.round(2), use_container_width=True)

    st.markdown("""
    <small>
    <b>Precision:</b> de las predichas positivas, ¿cuántas fueron correctas?<br>
    <b>Recall:</b> de las realmente positivas, ¿cuántas las encontró?<br>
    <b>F1:</b> media armónica de Precision y Recall.
    </small>""", unsafe_allow_html=True)

with col_cm:
    st.markdown("#### 🎯 Matriz de Confusión")
    st.plotly_chart(fig_confusion(y_test, y_pred), use_container_width=True)

with col_fi:
    st.markdown("#### 🧠 Feature Importance")
    st.plotly_chart(fig_feature_importance(model, features_usados), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#   SECCIÓN 4 │ CONTEXTO GLOBAL & CORRELACIONES
# ═══════════════════════════════════════════════════════════════════════════════
#
#   LÓGICA DE DESCARGA ROBUSTA (3 niveles de defensa):
#
#   Nivel 1 → yf.download con LISTA de tickers + threads=True
#             Si el resultado tiene MultiIndex (Price, Ticker), extraer
#             'Close' con .xs("Close", level=0, axis=1)
#
#   Nivel 2 → Si Nivel 1 falla, descargar cada ticker INDIVIDUALMENTE
#             con try/except aislado: un activo que falle NO rompe los demás.
#
#   Nivel 3 → Si un ticker individual falla, se omite con st.warning
#             y la app continúa con los que se descargaron exitosamente.
#
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("📡 Contexto Global & Correlaciones")

activos_radar = [ticker] + [b for b in BENCHMARKS_FIJOS if b.upper() != ticker.upper()]

with st.spinner("📡 Descargando benchmarks …"):

    df_closes = None

    # ── NIVEL 1: Descarga conjunta ──────────────────────────────────────────
    try:
        end   = datetime.now()
        start = end - timedelta(days=periodo_dias)

        df_multi = yf.download(
            activos_radar,
            start=start,
            end=end,
            progress=False,
            threads=True,
        )

        if df_multi.empty:
            raise ValueError("DataFrame vacío en descarga conjunta")

        # Extraer 'Close' según estructura de columnas
        if isinstance(df_multi.columns, pd.MultiIndex):
            # Niveles: (Price, Ticker)  →  .xs extrae el slice Price=='Close'
            df_closes = df_multi.xs("Close", level=0, axis=1).copy()
        else:
            # Un solo ticker: columnas planas
            if "Close" in df_multi.columns:
                df_closes = df_multi[["Close"]].copy()
                df_closes.columns = [ticker]
            else:
                raise ValueError("columna 'Close' no encontrada en descarga conjunta")

        df_closes.columns = [str(c).strip() for c in df_closes.columns]

    except Exception as exc:
        # ── NIVEL 2: Fallback individual por activo ──────────────────────
        st.warning(f"⚠️ Descarga conjunta fallida ({exc}). Modo individual …")
        series_dict: dict[str, pd.Series] = {}

        for activo in activos_radar:
            try:                                          # ← NIVEL 3: aislado
                df_tmp = descargar_ticker(activo, periodo_dias)
                if "Close" in df_tmp.columns:
                    series_dict[activo] = df_tmp["Close"]
                else:
                    st.warning(f"⚠️ **{activo}**: sin columna 'Close'. Omitido.")
            except Exception as exc2:
                st.warning(f"⚠️ **{activo}** falló: {exc2}. Omitido.")

        if series_dict:
            df_closes = pd.DataFrame(series_dict)
        else:
            st.error("❌ No se pudo descargar ningún activo. Revisa tu conexión.")
            st.stop()

    # ── Alineación temporal & dtype ──────────────────────────────────────────
    df_closes = df_closes.dropna()
    df_closes = df_closes.astype("float32")

# ── Verificación mínima ───────────────────────────────────────────────────────
if df_closes is None or df_closes.empty or len(df_closes) < 2:
    st.error("❌ No hay suficientes datos alineados para la comparativa.")
    st.stop()

activos_exitosos = df_closes.columns.tolist()
st.info(f"📊 Activos cargados: **{', '.join(activos_exitosos)}**  |  "
        f"Fechas alineadas: {len(df_closes)} días")

# ── VISUALIZACIÓN 1: Correlación (PRIORITARIA) ──────────────────────────────
st.markdown("#### 🔗 Correlación entre activos")

df_retornos = df_closes.pct_change().dropna()
st.plotly_chart(fig_correlacion(df_retornos), use_container_width=True)

st.info("""
**📖 Cómo leer la matriz:**
- **+1.0 (rojo intenso):** movimiento casi idéntico → alta sincronía.
- **0.0 (blanco):** sin relación lineal.
- **−1.0 (azul intenso):** movimientos opuestos → correlación inversa.

*Analogía: dos bloques en contacto térmico perfecto (+1) vs dos sistemas
que intercambian calor en sentidos contrarios (−1).*
""")

# ── VISUALIZACIÓN 2: Rendimiento normalizado Base 100 ────────────────────────
st.markdown("#### 📈 Rendimiento Relativo")
st.plotly_chart(fig_rendimiento_normalizado(df_closes, ticker), use_container_width=True)

st.info("""
**📖 Cómo leer el gráfico:**
Todos los activos empiezan en **100** (= "invertí 100 unidades el día 0").
Si un activo vale **115** → esa inversión creció un **15 %**.
Si vale **85** → perdió un **15 %**.
Esto permite comparar BTC (90 k) vs GLD (2 k) en el mismo eje.
""")


# ═══════════════════════════════════════════════════════════════════════════════
#   FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

with st.expander("📚 Guía de Uso"):
    st.markdown("""
    ### 🏭 Analogías del Sistema

    | Concepto Financiero | Analogía Ingenieril |
    |---|---|
    | Precios OHLCV | Materia prima cruda |
    | RSI, Bollinger, MACD | Sensores del proceso de refinado |
    | Target (Sube/Baja) | Vector de fuerza predicho |
    | XGBoost | Controlador PID inteligente |
    | Correlación | Acoplamiento térmico |
    | Base 100 | Calibración a origen común |

    ### 🎯 Interpretación
    - **Accuracy > 55 %:** el modelo supera al azar (50 %).
    - **Feature Importance alto:** ese indicador pesa más en la decisión.
    - **Correlación +1 → −1:** sincronía → movimiento opuesto entre activos.

    ### ⚠️ Advertencias
    - ❌ NO usar en trading real sin validación exhaustiva.
    - ❌ Resultados pasados NO garantizan resultados futuros.
    - ✅ Herramienta **educativa** para aprender ML aplicado en finanzas.
    """)

st.markdown("---")
st.caption("🏭 The Quant Refinery v4.1 — Unified View  |  Streamlit + XGBoost + pandas_ta")
