# Portfolio Construction Theory Dashboard

An interactive Streamlit app for exploring mean-variance portfolio optimization and performance attribution using real market data.

## Features

### Efficient Frontier
- Input any set of tickers and download historical prices via Yahoo Finance
- Compute and visualize both **unconstrained** (closed-form) and **constrained** (long-only, SLSQP) efficient frontiers
- Key portfolios: Global Minimum Variance (GMV), Max Sharpe (Tangency), and Equal-Weight
- Capital Market Line visualization
- Buy-and-hold backtesting with custom backtesting [see more!](https://github.com/RaulSN33/PortfolioBacktester)

### Performance Attribution
- Single-factor (CAPM) regression of any stock vs. S&P 500
- Static OLS and rolling OLS (12–60 month windows) for alpha and beta
- Variance decomposition into systematic and idiosyncratic components
- Cumulative return attribution charts

## Project Structure

```
portfolio_construction_theory/
├── __main__.py                         # App entry point (multi-page router)
├── requirements.txt
├── src/
│   ├── backend/backend.py              # Data fetching (yfinance, FRED)
│   ├── portfolio_construction/
│   │   ├── frontier.py                 # Unconstrained frontier (analytical)
│   │   ├── optimizations.py            # Constrained frontier (SLSQP)
│   │   └── performance_attribution.py  # Rolling CAPM regression
│   ├── dashboard/
│   │   ├── frontier_dashboard.py       # Plotly UI for frontier page
│   │   └── attribution_dashboard.py    # Plotly UI for attribution page
│   └── pages/
│       ├── efficient_frontier.py       # Frontier page logic
│       └── performance_attribution.py  # Attribution page logic
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the FRED API key

The Performance Attribution page pulls the risk-free rate from FRED (10-year Treasury, GS10). Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) and add it to `.streamlit/secrets.toml`:

```toml
FRED_API_KEY = "your_key_here"
```

### 3. Run the app

```bash
streamlit run __main__.py
```

Opens at `http://localhost:8501`.

### Alternative: Dev Container

Open in VS Code with the Dev Containers extension — the container is pre-configured to run on port 8501.

## Tech Stack

| Category | Libraries |
|---|---|
| Data | `yfinance`, `fredapi`, `pandas`, `numpy` |
| Optimization | `scipy` (SLSQP) |
| Regression | `statsmodels` |
| Visualization | `plotly`, `streamlit` |
| Backtesting | `PortfolioBacktester` |
