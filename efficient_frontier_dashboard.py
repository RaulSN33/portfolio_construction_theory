import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Efficient Frontier",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0d0d;
    color: #e8e8e0;
  }
  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #141414;
    border-right: 1px solid #2a2a2a;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 12px 16px;
  }
  div[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    color: #f0e68c;
  }

  /* Dataframe */
  .stDataFrame { border: 1px solid #2a2a2a; border-radius: 6px; }
  .stDataFrame th {
    font-family: 'IBM Plex Mono', monospace;
    background: #181818 !important;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #888 !important;
  }

  /* Divider accent */
  hr { border-color: #2a2a2a; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ────────────────────────────────────────────────────────────
def port_return(w, r):
    return float(w @ r)

def port_volatility(w, sigma):
    return float(np.sqrt(w @ sigma @ w))


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")

    raw_tickers = st.text_area(
        "Tickers (one per line or comma-separated)",
        value="F\nPFE\nCVX\nWMT\nSBUX\nAMZN\nDIS\nCOST\nMMM",
        height=200,
    )

    st.markdown("---")
    st.markdown("**Date Range**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.Timestamp("2015-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp("today"))

    st.markdown("---")
    rf_rate = st.number_input(
        "Monthly Risk-Free Rate",
        min_value=0.0,
        max_value=0.05,
        value=0.001,
        step=0.0001,
        format="%.4f",
    )

    run = st.button("▶  Run Analysis", use_container_width=True)


# ── Title ───────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Efficient Frontier")
st.markdown("*Monthly historical risk-return tradeoff with Capital Market Line*")
st.markdown("---")


# ── Main logic ──────────────────────────────────────────────────────────────────
if not run:
    st.info("Configure parameters in the sidebar and click **▶ Run Analysis** to begin.")
    st.stop()


# Parse tickers
tickers_raw = raw_tickers.replace(",", "\n").replace(" ", "\n")
stocks = [t.strip().upper() for t in tickers_raw.splitlines() if t.strip()]

if len(stocks) < 2:
    st.error("Please enter at least 2 tickers.")
    st.stop()


# ── Download data ────────────────────────────────────────────────────────────────
with st.spinner(f"Downloading data for: {', '.join(stocks)}…"):
    try:
        raw = yf.download(
            stocks,
            start=str(start_date),
            end=str(end_date),
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

# Handle single vs multi-ticker column structure
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"]
else:
    prices = raw[["Close"]]
    prices.columns = stocks

# Drop tickers with all-NaN
prices = prices.dropna(axis=1, how="all")
valid_stocks = prices.columns.tolist()

if len(valid_stocks) < 2:
    st.error("Not enough valid tickers after download. Check ticker symbols.")
    st.stop()

if set(valid_stocks) != set(stocks):
    removed = set(stocks) - set(valid_stocks)
    st.warning(f"Removed tickers with no data: {', '.join(removed)}")

prices = prices.dropna()
returns = prices.pct_change().dropna()

if returns.shape[0] < 12:
    st.error("Not enough monthly data points (need at least 12). Try a wider date range.")
    st.stop()


# ── Computations ─────────────────────────────────────────────────────────────────
miu = returns.mean()
sigma = returns.cov()
std_dev = returns.std()
n = len(valid_stocks)

ones = np.ones(n)
sigma_inv = np.linalg.pinv(sigma.values)

# Scalar helpers
a = float(ones @ sigma_inv @ ones)
b = float(ones @ sigma_inv @ miu.values)
c = float(miu.values @ sigma_inv @ miu.values)

# GMV
w_gmv = (sigma_inv @ ones) / (ones @ sigma_inv @ ones)
gmv_ret = port_return(w_gmv, miu.values)
gmv_std = port_volatility(w_gmv, sigma.values)

# Max Sharpe (tangency with rf)
ex_returns = miu.values - rf_rate
u = np.linalg.solve(sigma.values, ex_returns)
w_tan = u / u.sum()
tan_ret = port_return(w_tan, miu.values)
tan_std = port_volatility(w_tan, sigma.values)

# Equal weight
w_ew = np.ones(n) / n
ew_ret = port_return(w_ew, miu.values)
ew_std = port_volatility(w_ew, sigma.values)

# Efficient frontier curve
min_vol = np.sqrt(1 / a)
step = 0.001
num_pts = 65
range_vols = np.linspace(min_vol, min_vol + step * num_pts, num=num_pts)
result_miu = b / a + np.sqrt(c - b**2 / a) * np.sqrt(range_vols**2 - 1 / a)

# CML
cml_slope = (tan_ret - rf_rate) / tan_std
x_cml = np.linspace(0, max(range_vols) * 1.15, 200)
y_cml = rf_rate + cml_slope * x_cml

# Sharpe ratios
def sharpe(ret, vol, rf):
    return (ret - rf) / vol if vol > 0 else np.nan


# ── Top metrics ──────────────────────────────────────────────────────────────────
st.markdown("### Portfolio Summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Tickers", len(valid_stocks))
with c2:
    st.metric("Months", len(returns))
with c3:
    st.metric("GMV Return", f"{gmv_ret:.2%}")
with c4:
    st.metric("GMV Volatility", f"{gmv_std:.2%}")
with c5:
    st.metric("Tangency Return", f"{tan_ret:.2%}")
with c6:
    st.metric("Tangency Sharpe", f"{sharpe(tan_ret, tan_std, rf_rate):.2f}")


st.markdown("---")


# ── Portfolio weights & stats table ──────────────────────────────────────────────
st.markdown("### Portfolio Details")

portfolios = {
    "GMV": w_gmv,
    "Max Sharpe (Tangency)": w_tan,
    "Equal Weight": w_ew,
}
port_rets  = {"GMV": gmv_ret, "Max Sharpe (Tangency)": tan_ret, "Equal Weight": ew_ret}
port_stds  = {"GMV": gmv_std, "Max Sharpe (Tangency)": tan_std, "Equal Weight": ew_std}

rows = []
for name, w in portfolios.items():
    row = {"Portfolio": name}
    row["Monthly Return"] = f"{port_rets[name]:.4%}"
    row["Monthly Std Dev"] = f"{port_stds[name]:.4%}"
    row["Sharpe Ratio"]    = f"{sharpe(port_rets[name], port_stds[name], rf_rate):.3f}"
    for ticker, weight in zip(valid_stocks, w):
        row[ticker] = f"{weight:.2%}"
    rows.append(row)

table_df = pd.DataFrame(rows).set_index("Portfolio")
st.dataframe(table_df, use_container_width=True)


st.markdown("---")


# ── Plotly chart ──────────────────────────────────────────────────────────────────
st.markdown("### Efficient Frontier & Capital Market Line")

fig = go.Figure()

COLORS = {
    "frontier": "#888888",
    "cml":      "#ff6b6b",
    "assets":   "#7ec8e3",
    "gmv":      "#f0e68c",
    "tangency": "#90ee90",
    "rf":       "#ff6b6b",
    "ew":       "#c9a0ff",
}

BG = "#0d0d0d"
GRID = "#1e1e1e"

# Efficient frontier
fig.add_trace(go.Scatter(
    x=range_vols, y=result_miu,
    mode="lines",
    name="Efficient Frontier",
    line=dict(color=COLORS["frontier"], width=2.5),
    hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
))

# CML
fig.add_trace(go.Scatter(
    x=x_cml, y=y_cml,
    mode="lines",
    name="Capital Market Line",
    line=dict(color=COLORS["cml"], width=2, dash="dash"),
    hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
))

# Individual assets
fig.add_trace(go.Scatter(
    x=std_dev.values, y=miu.values,
    mode="markers+text",
    name="Assets",
    marker=dict(color=COLORS["assets"], size=9, symbol="circle",
                line=dict(color="#ffffff", width=0.5)),
    text=valid_stocks,
    textposition="top right",
    textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["assets"]),
    hovertemplate="<b>%{text}</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
))

# Equal weight
fig.add_trace(go.Scatter(
    x=[ew_std], y=[ew_ret],
    mode="markers+text",
    name="Equal Weight",
    marker=dict(color=COLORS["ew"], size=16, symbol="diamond",
                line=dict(color="#ffffff", width=1)),
    text=["1/N"],
    textposition="top right",
    textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["ew"]),
    hovertemplate="<b>Equal Weight</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
))

# GMV
fig.add_trace(go.Scatter(
    x=[gmv_std], y=[gmv_ret],
    mode="markers+text",
    name="GMV",
    marker=dict(color=COLORS["gmv"], size=18, symbol="star",
                line=dict(color="#ffffff", width=1)),
    text=["GMV"],
    textposition="top right",
    textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["gmv"]),
    hovertemplate="<b>GMV</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
))

# Tangency / Max Sharpe
fig.add_trace(go.Scatter(
    x=[tan_std], y=[tan_ret],
    mode="markers+text",
    name="Max Sharpe (Tangency)",
    marker=dict(color=COLORS["tangency"], size=18, symbol="star",
                line=dict(color="#ffffff", width=1)),
    text=["Tangency"],
    textposition="top right",
    textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["tangency"]),
    hovertemplate="<b>Tangency</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
))

# Risk-free rate
fig.add_trace(go.Scatter(
    x=[0], y=[rf_rate],
    mode="markers+text",
    name="Risk-Free Rate",
    marker=dict(color=COLORS["rf"], size=12, symbol="circle",
                line=dict(color="#ffffff", width=1)),
    text=["Rf"],
    textposition="top right",
    textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["rf"]),
    hovertemplate=f"<b>Risk-Free Rate</b><br>{rf_rate:.4%}<extra></extra>",
))

# Layout
fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="IBM Plex Mono", color="#e8e8e0"),
    xaxis=dict(
        title="Monthly Volatility (Std Dev)",
        tickformat=".1%",
        gridcolor=GRID,
        zerolinecolor=GRID,
        title_font=dict(size=12),
    ),
    yaxis=dict(
        title="Monthly Return",
        tickformat=".2%",
        gridcolor=GRID,
        zerolinecolor=GRID,
        title_font=dict(size=12),
    ),
    legend=dict(
        bgcolor="#141414",
        bordercolor="#2a2a2a",
        borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=60, r=40, t=40, b=60),
    height=580,
    hovermode="closest",
)

st.plotly_chart(fig, use_container_width=True)

st.caption(f"Data: Yahoo Finance · Interval: Monthly · RF Rate: {rf_rate:.4%}/month")