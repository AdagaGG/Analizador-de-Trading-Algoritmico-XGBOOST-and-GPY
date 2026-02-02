"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🏭  THE QUANT REFINERY  v4.0  🏭                           ║
║           Sistema Adaptativo de Trading | Arquitectura Dual                  ║
║                                                                              ║
║  MÓDULOS:                                                                    ║
║    🏭  Refinería      → Análisis individual + ML (XGBoost)                  ║
║    📡  Radar          → Comparativa multi-activo + Correlación              ║
║                                                                              ║
║  CHANGELOG:                                                                  ║
║    v4.0  Reescritura completa. Arquitectura de dos módulos.                 ║
║    v3.4  Candlestick + subplot Volumen                                      ║
║    v3.2  Fix dtype int64 → float64 en Equity                               ║
║    v3.1  Type Safety, max_bin=256, Pandas 2.0                               ║
║                                                                              ║
║  DEPENDENCIAS LOCALES:                                                       ║
║    pandas_ta  →  vendorizada en carpeta local (NO pip install)              ║
║                                                                              ║
║  ANALOGÍAS INGENIERILES:                                                     ║
║    Precios OHLCV     →  Materia Prima Cruda                                 ║
║    Velas Japonesas   →  Diagrama de Fase (T vs P)                           ║
║    RSI / MACD / ADX  →  Sensores de Proceso (Refinado)                      ║
║    Target            →  Vector de Fuerza Predicho                           ║
║    XGBoost           →  Controlador PID Inteligente                         ║
║    Normalización     →  Calibración a origen común                          ║
║    Correlación       →  Acoplamiento térmico entre sistemas                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 │ IMPORTACIONES & CONFIGURACIÓN GLOBAL
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

# pandas_ta está vendorizada en carpeta local del proyecto.
# No se instala por pip; existe como carpeta sibling de este archivo.
import pandas_ta as ta

warnings.filterwarnings("ignore")

# ─── Paletа de colores (SCADA industrial) ────────────────────────────────────
CLR_UP      = "#26a69a"   # Verde teal  → vela alcista / señal positiva
CLR_DOWN    = "#ef5350"   # Rojo coral  → vela bajista / señal negativa
CLR_BB      = "rgba(255,255,255,0.30)"
CLR_BB_FILL = "rgba(255,255,255,0.04)"
CLR_RSI     = "#ffa726"   # Naranja
CLR_MACD    = "#42a5f5"   # Azul
CLR_SIGNAL  = "#ab47bc"   # Violeta
CLR_HIST_POS= "#26a69a"
CLR_HIST_NEG= "#ef5350"

# ─── Lista de benchmarks para el Radar ────────────────────────────────────────
BENCHMARKS_BASE = ["SPY", "QQQ", "BTC-USD", "ETH-USD", "GC=F"]

# ─── Colores asignados al Radar (paleta distinguible) ────────────────────────
RADAR_COLORS = {
    "SPY":     "#42a5f5",   # Azul
    "QQQ":     "#66bb6a",   # Verde
    "BTC-USD": "#ffa726",   # Naranja
    "ETH-USD": "#ab47bc",   # Violeta
    "GC=F":    "#ffee58",   # Amarillo
    "__USER__":"#ffffff",   # Blanco → el activo del usuario siempre se destaca
}

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🏭 The Quant Refinery v4.0",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 │ CAPA DE DATOS  —  Descarga, limpieza, indicadores
# ═══════════════════════════════════════════════════════════════════════════════


def _aplanar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    🛡️ Parche yfinance: aplanar MultiIndex y normalizar nombres.

    yfinance (especialmente con versiones recientes) puede devolver columnas
    como ('Close', 'AAPL').  Este parche las convierte a strings simples.
    También elimina duplicados y verifica la existencia de las columnas OHLCV.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalizar nombres (strip + title-case estándar)
    df.columns = [str(c).strip() for c in df.columns]

    # Mapeo case-insensitive a nombres canónicos
    _map = {
        "close": "Close", "high": "High", "low": "Low",
        "open": "Open", "volume": "Volume", "adj close": "Adj Close",
    }
    df.columns = [_map.get(c.lower(), c) for c in df.columns]

    # Eliminar columnas duplicadas (mantener primera ocurrencia)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_datos(ticker: str, dias: int = 365) -> pd.DataFrame:
    """
    📡 Descarga datos OHLCV de Yahoo Finance.

    Si la descarga falla por cualquier motivo (red, ticker inválido, …)
    genera un DataFrame sintético (senoidal + ruido browniano) para que
    la UI siempre pueda renderizarse en modo demo.

    Optimización de memoria: convierte columnas numéricas a float32.
    """
    end   = datetime.now()
    start = end - timedelta(days=dias)

    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("respuesta vacía de Yahoo Finance")

        df = _aplanar_columnas(df)

        # Verificar columnas críticas
        for col in ("Open", "High", "Low", "Close"):
            if col not in df.columns:
                raise KeyError(f"columna '{col}' faltante tras aplanar")

        # float32 solo en columnas numéricas (Type Safety v3.1)
        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].astype("float32")

        return df

    except Exception as exc:
        st.warning(f"⚠️ No se pudo descargar **{ticker}**: {exc}  →  modo demo.")
        return _datos_sinteticos(dias)


def _datos_sinteticos(dias: int) -> pd.DataFrame:
    """Genera datos OHLCV sintéticos para testing sin red."""
    idx    = pd.date_range(end=datetime.now(), periods=dias, freq="D")
    trend  = np.linspace(100, 150, dias)
    season = 20 * np.sin(np.linspace(0, 4 * np.pi, dias))
    noise  = np.random.randn(dias) * 3
    close  = (trend + season + noise).astype("float32")

    df = pd.DataFrame({
        "Open":   (close * 0.995).astype("float32"),
        "High":   (close * 1.015).astype("float32"),
        "Low":    (close * 0.985).astype("float32"),
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 10_000_000, dias).astype("float32"),
    }, index=idx)
    return df


# ─── Indicadores técnicos vía pandas_ta (vendorizada) ─────────────────────────

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    ⚙️ PLANTA DE REFINADO  —  convierte Materia Prima en señales.

    Indicadores calculados:
        RSI(14)          →  Sensor de sobrecalentamiento
        Bollinger(20,2)  →  Límites de control de calidad
        MACD(12,26,9)    →  Oscilador de momentum
        ADX(14)          →  Número de Reynolds (tendencia vs turbulencia)
        Volume_Norm      →  Volumen normalizado (bypass si es 0 → Forex)

    pandas_ta se usa directamente sobre la serie 'Close'.
    """
    data = df.copy()

    # ── RSI ───────────────────────────────────────────────────────────────
    data["RSI"] = ta.rsi(data["Close"], length=14)

    # ── Bollinger ─────────────────────────────────────────────────────────
    bb = ta.bbands(data["Close"], length=20, std=2)
    if bb is not None:
        data["BB_Upper"]  = bb.iloc[:, 2]   # BBU
        data["BB_Middle"] = bb.iloc[:, 1]   # BBM
        data["BB_Lower"]  = bb.iloc[:, 0]   # BBL

    # ── MACD ──────────────────────────────────────────────────────────────
    macd = ta.macd(data["Close"], fast=12, slow=26, signal=9)
    if macd is not None:
        data["MACD"]        = macd.iloc[:, 0]   # MACD línea
        data["MACD_Signal"] = macd.iloc[:, 1]   # Línea señal
        data["MACD_Hist"]   = macd.iloc[:, 2]   # Histograma

    # ── ADX ───────────────────────────────────────────────────────────────
    adx = ta.adx(high=data["High"], low=data["Low"], close=data["Close"], length=14)
    if adx is not None and "ADX_14" in adx.columns:
        data["ADX"] = adx["ADX_14"]

    # ── Volumen normalizado (parche Forex: si sum==0 → 0) ────────────────
    if "Volume" in data.columns and data["Volume"].sum() > 0:
        vol_media = data["Volume"].rolling(20).mean().replace(0, 1)
        data["Volume_Norm"] = data["Volume"] / vol_media
    else:
        data["Volume_Norm"] = 0.0

    # ── Retorno diario ────────────────────────────────────────────────────
    data["Returns"] = data["Close"].pct_change()

    # Reemplazar infinitos por NaN
    data = data.replace([np.inf, -np.inf], np.nan)

    # float32 en columnas numéricas (mantener Type Safety)
    num_cols = data.select_dtypes(include=["number"]).columns
    data[num_cols] = data[num_cols].astype("float32")

    return data


# ─── Target + preparación para ML ─────────────────────────────────────────────

# Lista de features que usa el modelo (orden canónico)
FEATURE_COLS = [
    "RSI", "BB_Upper", "BB_Middle", "BB_Lower",
    "MACD", "MACD_Signal", "MACD_Hist",
    "ADX", "Volume_Norm", "Returns",
]


def preparar_dataset(df: pd.DataFrame, test_pct: int = 20):
    """
    🎯 Crea Target con .shift(-1)  →  prevención de Look-Ahead Bias.

    ⚠️  ZONA CRÍTICA DE CAUSALIDAD:
        Target[t] = 1  si  Close[t+1] > Close[t]
        Target[t] = 0  si  Close[t+1] ≤ Close[t]

        .shift(-1) nos da el retorno del DÍA SIGUIENTE sin usar datos
        del futuro durante el entrenamiento.  El último row se elimina
        porque no tiene target válido.

    Retorna: X_train, X_test, y_train, y_test, scaler, features usados, df_clean
    """
    data = df.copy()

    # Target binario
    data["Future_Return"] = data["Close"].pct_change().shift(-1)
    data["Target"]        = (data["Future_Return"] > 0).astype(int)
    data = data.iloc[:-1]  # última fila sin target

    # Quedarnos solo con features que existen en el DataFrame
    features = [f for f in FEATURE_COLS if f in data.columns]

    # Eliminar NaN en features + target
    cols_necesarios = features + ["Target"]
    data = data.dropna(subset=cols_necesarios)

    X = data[features].values.astype("float32")
    y = data["Target"].values

    # ── Corte temporal ESTRICTO (sin shuffle) ──────────────────────────
    split = int(len(X) * (1 - test_pct / 100))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Normalización
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype("float32")
    X_test_s  = scaler.transform(X_test).astype("float32")

    return X_train_s, X_test_s, y_train, y_test, scaler, features, data


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 │ CAPA ML  —  XGBoost
# ═══════════════════════════════════════════════════════════════════════════════


def entrenar_modelo(X_train, y_train, params: dict, device: str = "cpu"):
    """
    🤖 Controlador PID Inteligente (XGBoost).

    max_bin=256   →  Optimización VRAM para GPUs de 4 GB (RTX 3050).
    tree_method='hist' + device  →  Acelera en GPU si está disponible.

    Si el usuario selecciona CUDA pero no hay GPU, el código cae a CPU
    de forma silenciosa (XGBoost lo maneja internamente con device='cpu').
    """
    model = xgb.XGBClassifier(
        max_depth       = params.get("max_depth", 5),
        learning_rate   = params.get("learning_rate", 0.1),
        n_estimators    = params.get("n_estimators", 100),
        tree_method     = "hist",
        device          = device,
        max_bin         = 256,
        random_state    = 42,
        eval_metric     = "logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 │ CAPA DE VISUALIZACIÓN  —  Gráficos Plotly
# ═══════════════════════════════════════════════════════════════════════════════


def grafico_candlestick(df: pd.DataFrame):
    """
    🕯️ Gráfico principal de la Refinería.

    Subplots:
        Row 1  →  Candlestick + Bandas de Bollinger
        Row 2  →  MACD (línea + señal + histograma)
        Row 3  →  RSI con zonas de sobrecompra/sobreventa
        Row 4  →  Volumen (barras coloreadas por dirección)

    Analogía: cada vela es un diagrama de fase instantáneo del precio.
    El cuerpo indica la fase final; las mechas indican las transiciones.
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "🕯️ Precio  —  Velas Japonesas + Bollinger",
            "📊 MACD",
            "📈 RSI (14)",
            "📦 Volumen",
        ),
        row_heights=[0.45, 0.22, 0.18, 0.15],
    )

    # ── ROW 1: Candlestick ──────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name="OHLC",
            increasing=dict(line=dict(color=CLR_UP),   fillcolor=CLR_UP),
            decreasing=dict(line=dict(color=CLR_DOWN), fillcolor=CLR_DOWN),
        ),
        row=1, col=1,
    )

    # Bollinger sobre las velas
    if "BB_Upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Upper"], mode="lines",
                       name="BB Superior",
                       line=dict(color=CLR_BB, width=1, dash="dot")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Lower"], mode="lines",
                       name="BB Inferior",
                       line=dict(color=CLR_BB, width=1, dash="dot"),
                       fill="tonexty", fillcolor=CLR_BB_FILL),
            row=1, col=1,
        )

    # ── ROW 2: MACD ─────────────────────────────────────────────────────
    if "MACD" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD"], mode="lines",
                       name="MACD", line=dict(color=CLR_MACD, width=1.5)),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines",
                       name="Señal", line=dict(color=CLR_SIGNAL, width=1.5)),
            row=2, col=1,
        )
        # Histograma MACD coloreado
        hist_colors = [CLR_HIST_POS if v >= 0 else CLR_HIST_NEG
                       for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(
            go.Bar(x=df.index, y=df["MACD_Hist"], name="Histograma",
                   marker=dict(color=hist_colors), showlegend=False),
            row=2, col=1,
        )

    # ── ROW 3: RSI ──────────────────────────────────────────────────────
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], mode="lines",
                       name="RSI", line=dict(color=CLR_RSI, width=2)),
            row=3, col=1,
        )
        fig.add_hline(y=70, line_dash="dash",
                      line_color="rgba(239,83,80,0.5)",
                      annotation_text="Sobrecompra", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash",
                      line_color="rgba(38,166,154,0.5)",
                      annotation_text="Sobreventa",  row=3, col=1)

    # ── ROW 4: Volumen ──────────────────────────────────────────────────
    if "Volume" in df.columns:
        vol_colors = [CLR_UP if c >= o else CLR_DOWN
                      for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volumen",
                   marker=dict(color=vol_colors)),
            row=4, col=1,
        )

    # ── Layout global ───────────────────────────────────────────────────
    fig.update_layout(
        height=820,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        hovermode="x unified",
        xaxis4_rangeslider_visible=False,   # Ocultar slider de Candlestick
        margin=dict(l=40, r=30, t=60, b=20),
    )
    fig.update_yaxes(range=[0, 100], row=3, col=1)   # RSI fijo 0-100

    return fig


def grafico_feature_importance(model, feature_names: list):
    """📊 Barras horizontales de Feature Importance del modelo XGBoost."""
    imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importancia": model.feature_importances_,
    }).sort_values("Importancia", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=imp["Importancia"],
            y=imp["Feature"],
            orientation="h",
            marker=dict(color=imp["Importancia"], colorscale="Viridis"),
        )
    )
    fig.update_layout(
        title="🧠 ¿Qué indicadores pesan más en la decisión?",
        xaxis_title="Importancia relativa",
        template="plotly_dark",
        height=420,
        margin=dict(l=100),
    )
    return fig


def grafico_confusion_matrix(y_true, y_pred):
    """🎯 Heatmap de la Matriz de Confusión con anotaciones."""
    cm = confusion_matrix(y_true, y_pred)
    labels_x = ["Pred: BAJA ▼", "Pred: SUBE ▲"]
    labels_y = ["Real: BAJA ▼", "Real: SUBE ▲"]

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels_x, y=labels_y,
            colorscale="Blues",
            text=cm, texttemplate="%{text}",
            textfont=dict(size=22, color="white"),
            colorbar=dict(title="Cantidad"),
        )
    )
    fig.update_layout(
        title="🎯 Matriz de Confusión  —  Control de Calidad",
        template="plotly_dark",
        height=380,
        xaxis=dict(title="Predicción"),
        yaxis=dict(title="Realidad", autorange="reversed"),
    )
    return fig


# ─── Gráficos del Radar de Mercado ────────────────────────────────────────────

def grafico_retorno_acumulado(df_norm: pd.DataFrame, ticker_usuario: str):
    """
    📡 Líneas de retorno acumulado normalizado.

    La línea del activo del usuario se resalta con mayor grosor y opacidad.
    Todos los demás activos se dibujan con línea más delgada y semi-transparente.

    Normalización previa (aplicada fuera):
        retorno_acum = (precio / precio[0] - 1) * 100
        → todos empiezan en 0 % el primer día.
    """
    fig = go.Figure()

    for col in df_norm.columns:
        is_user = (col == ticker_usuario)
        fig.add_trace(
            go.Scatter(
                x=df_norm.index,
                y=df_norm[col],
                name=col,
                mode="lines",
                line=dict(
                    color = RADAR_COLORS.get(col, RADAR_COLORS.get("__USER__")),
                    width = 3.5 if is_user else 1.8,
                ),
                opacity = 1.0 if is_user else 0.55,
            )
        )

    # Línea horizontal en 0 % (punto de partida común)
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)")

    fig.update_layout(
        title="📡 Radar de Mercado  —  Retorno Acumulado Normalizado (%)",
        xaxis_title="Fecha",
        yaxis_title="Retorno acumulado (%)",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        hovermode="x unified",
    )
    return fig


def grafico_correlacion(df_retornos: pd.DataFrame):
    """
    🔗 Heatmap de correlación basada en retornos diarios (pct_change).

    Analogía: acoplamiento térmico.  Dos activos con correlación cercana
    a +1 se comportan como dos bloques de metal en contacto térmico
    perfecto; cercana a -1, como dos sistemas que intercambian calor
    en sentidos opuestos.
    """
    corr = df_retornos.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",          # Rojo (+1) → Azul (-1)
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=14),
            colorbar=dict(title="Correlación"),
        )
    )
    fig.update_layout(
        title="🔗 Matriz de Correlación  —  Retornos Diarios",
        template="plotly_dark",
        height=480,
        xaxis=dict(tickangle=25),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 │ SIDEBAR  —  Panel de Control
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎛️ Centro de Control")
    st.markdown("---")

    # ── Fuente de datos ─────────────────────────────────────────────────
    st.subheader("📡 Activo")
    ticker = st.text_input("Ticker", value="AAPL",
                           placeholder="AAPL, TSLA, BTC-USD …",
                           help="Cualquier símbolo soportado por Yahoo Finance").strip().upper()
    periodo_dias = st.slider("Período histórico (días)", 180, 1095, 365, step=30)

    st.markdown("---")

    # ── Hiperparámetros ─────────────────────────────────────────────────
    st.subheader("⚙️ XGBoost")
    max_depth      = st.slider("Max Depth",       3, 10,  5)
    learning_rate  = st.slider("Learning Rate (η)", 0.01, 0.30, 0.10, step=0.01)
    n_estimators   = st.slider("N° de Árboles",   50, 500, 100, step=50)
    test_pct       = st.slider("Test Size (%)",   10,  40,  20)

    st.markdown("---")

    # ── Dispositivo ─────────────────────────────────────────────────────
    st.subheader("🖥️ Dispositivo")
    device_choice = st.radio("Procesador", ["CPU", "GPU (CUDA)"], index=0,
                             help="GPU requiere CUDA compatible")
    device = "cuda" if "GPU" in device_choice else "cpu"

    st.markdown("---")
    st.caption("🏭 The Quant Refinery v4.0")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 │ INTERFAZ PRINCIPAL  —  Tabs + Lógica
# ═══════════════════════════════════════════════════════════════════════════════

# ── Título global ────────────────────────────────────────────────────────────
st.title("🏭 The Quant Refinery v4.0")
st.markdown("*Sistema Adaptativo de Trading Algorítmico*")
st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_refinery, tab_radar = st.tabs([
    "🏭  Refinería  —  Análisis Individual",
    "📡  Radar de Mercado  —  Comparativa",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 │  🏭  REFINERÍA
# ═══════════════════════════════════════════════════════════════════════════════

with tab_refinery:

    # ── Botón de ejecución ────────────────────────────────────────────────
    ejecutar_refinery = st.button(
        "🚀 Ejecutar Refinería", type="primary", use_container_width=True
    )

    if not ejecutar_refinery:
        st.info("👆 Configura el ticker en el sidebar y pulsa **Ejecutar Refinería**.")
        st.stop()

    # ── Pipeline de datos ─────────────────────────────────────────────────
    with st.spinner(f"📡 Descargando datos de **{ticker}** …"):
        df_raw = descargar_datos(ticker, periodo_dias)

    with st.spinner("⚙️ Calculando indicadores técnicos …"):
        df = calcular_indicadores(df_raw)

    # ── Gráfico Candlestick ───────────────────────────────────────────────
    st.plotly_chart(grafico_candlestick(df), use_container_width=True)

    # ── Sección ML ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 Motor de Predicción  —  XGBoost")

    with st.spinner("🤖 Preparando dataset y entrenando modelo …"):
        (X_train, X_test, y_train, y_test,
         scaler, features_usados, df_clean) = preparar_dataset(df, test_pct)

        params = {
            "max_depth":     max_depth,
            "learning_rate": learning_rate,
            "n_estimators":  n_estimators,
        }
        model = entrenar_modelo(X_train, y_train, params, device)

    y_pred = model.predict(X_test)

    # ── KPIs de métricas ──────────────────────────────────────────────────
    accuracy = (y_pred == y_test).mean() * 100
    n_train  = len(y_train)
    n_test   = len(y_test)

    k1, k2, k3 = st.columns(3)
    k1.metric("🎯 Accuracy",   f"{accuracy:.2f} %")
    k2.metric("📦 Train / Test", f"{n_train} / {n_test}")
    k3.metric("🌳 Árboles",     str(n_estimators))

    # ── Feature Importance ────────────────────────────────────────────────
    st.plotly_chart(
        grafico_feature_importance(model, features_usados),
        use_container_width=True,
    )

    # ── Matriz de Confusión ───────────────────────────────────────────────
    st.plotly_chart(
        grafico_confusion_matrix(y_test, y_pred),
        use_container_width=True,
    )

    # Interpretación de la Matriz
    st.info("""
    **📖 Lectura de la Matriz (analogía de control de calidad):**

    | Celda | Significado |
    |---|---|
    | **True Positive (TP)** | Predijo ▲ y efectivamente subió → ✅ Acierto |
    | **True Negative (TN)** | Predijo ▼ y efectivamente bajó → ✅ Acierto |
    | **False Positive (FP)** | Predijo ▲ pero bajó → ⚠️ *Falsa alarma* (pérdida) |
    | **False Negative (FN)** | Predijo ▼ pero subió → ⚠️ *Fallo no detectado* (oportunidad perdida) |
    """)

    # ── Reporte detallado (expandible) ────────────────────────────────────
    with st.expander("📋 Reporte de Clasificación detallado"):
        report = classification_report(
            y_test, y_pred,
            target_names=["BAJA ▼", "SUBE ▲"],
            output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).T)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 │  📡  RADAR DE MERCADO
# ═══════════════════════════════════════════════════════════════════════════════

with tab_radar:

    st.subheader("📡 Radar de Mercado  —  Comparativa Multi-Activo")
    st.markdown(
        "Compara el rendimiento de tu activo contra benchmarks clave del mercado. "
        "La descarga se ejecuta **solo cuando pulsas el botón** para no ralentizar la app."
    )

    # ── Lista de activos a escanear ─────────────────────────────────────
    activos_radar = [ticker] + [b for b in BENCHMARKS_BASE if b != ticker]

    st.info(f"Activos a escanear: **{', '.join(activos_radar)}**")

    # ── Botón on-demand ──────────────────────────────────────────────────
    escanear = st.button("🔍 Escanear Mercado", type="primary",
                         use_container_width=True)

    if not escanear:
        st.info("👆 Pulsa **Escanear Mercado** para cargar los datos comparativos.")
        st.stop()

    # ── Descarga de todos los activos ─────────────────────────────────────
    dataframes_raw: dict[str, pd.DataFrame] = {}

    with st.spinner("📡 Descargando benchmarks …"):
        for activo in activos_radar:
            dataframes_raw[activo] = descargar_datos(activo, periodo_dias)

    # ── Construir DataFrame unificado de precios de cierre ────────────────
    #    Solo usamos 'Close' de cada activo para las comparativas.

    serie_closes: dict[str, pd.Series] = {}
    for nombre, df_tmp in dataframes_raw.items():
        if "Close" in df_tmp.columns:
            serie_closes[nombre] = df_tmp["Close"]

    df_closes = pd.DataFrame(serie_closes)
    # Alinear fechas: inner join implícito al crear DataFrame desde series
    df_closes = df_closes.dropna()

    if df_closes.empty or len(df_closes) < 2:
        st.error("❌ No se pudieron alinear datos entre los activos. "
                 "Intenta con un período más largo o otros tickers.")
        st.stop()

    # ── NORMALIZACIÓN: Retorno acumulado porcentual desde el día 1 ─────────
    #
    #    Fórmula:  retorno_acum[t] = (precio[t] / precio[0] − 1) × 100
    #
    #    ANALOGÍA: Es como calibrar todos los sensores a un origen común
    #    (punto de referencia cero).  Sin esto, Bitcoin a 90 k $ se vería
    #    como una línea plana frente al Oro a 2 k $.
    #
    df_norm = ((df_closes / df_closes.iloc[0]) - 1) * 100

    # ── Gráfico de retorno acumulado ──────────────────────────────────────
    st.plotly_chart(
        grafico_retorno_acumulado(df_norm, ticker),
        use_container_width=True,
    )

    # ── Tabla resumen de rendimientos ────────────────────────────────────
    st.subheader("📊 Resumen de Rendimientos")

    resumen = pd.DataFrame({
        "Activo": df_norm.columns,
        "Retorno Acum. (%)": df_norm.iloc[-1].values.round(2),
        "Precio Inicio": df_closes.iloc[0].values.round(2),
        "Precio Final":  df_closes.iloc[-1].values.round(2),
    })
    resumen = resumen.sort_values("Retorno Acum. (%)", ascending=False).reset_index(drop=True)
    st.dataframe(resumen, use_container_width=True)

    # ── MATRIZ DE CORRELACIÓN ─────────────────────────────────────────────
    #
    #    Se calcula sobre los retornos DIARIOS (pct_change), no sobre los
    #    precios brutos.  Los precios brutos son series no estacionarias
    #    (trending) y producen correlaciones espurias cercanas a 1.
    #    Los retornos diarios son (aproximadamente) estacionarios.
    #
    st.markdown("---")
    st.subheader("🔗 Acoplamiento entre Activos  —  Correlación Diaria")

    df_retornos_diarios = df_closes.pct_change().dropna()

    st.plotly_chart(
        grafico_correlacion(df_retornos_diarios),
        use_container_width=True,
    )

    st.info("""
    **📖 Cómo leer la matriz:**
    - **+1.0 (rojo intenso):** Movimiento casi idéntico → alta sincronía.
    - **0.0 (blanco):**        Sin relación lineal.
    - **−1.0 (azul intenso):** Movimientos opuestos → correlación inversa.

    *Analogía:  dos bloques en contacto térmico perfecto (+1) vs dos
    sistemas que intercambian calor en sentidos contrarios (−1).*
    """)