# Quantitative Portfolio Analytics

A Streamlit workbench for portfolio research, risk analysis, and reup / pullback decisions. Built around `yfinance` for market data, with twelve analysis pages covering screening, performance, risk decomposition, factor models, optimization, Monte Carlo, stress testing, currency exposure, and commitment pacing.

## Features

| Page | What it does |
|---|---|
| **Home** | Build a portfolio (manual table or CSV upload), set AUM, view live exposure metrics and per-position dollar breakdown |
| **Screener** | 30+ per-ticker metrics, Piotroski F-Score, multi-factor composite score, foreign-equity support |
| **Performance** | CAGR, Sharpe, Sortino, Calmar, Omega, alpha/beta, capture ratios, drawdown episodes, rolling metrics, monthly heatmap |
| **Risk Metrics** | Historical / parametric / Cornish-Fisher / Monte Carlo VaR & CVaR, component & marginal risk |
| **Statistical Tests** | Normality, stationarity (ADF / KPSS), Ljung-Box, ARCH-LM, variance-ratio, runs test |
| **Factor Models** | CAPM with full OLS output, custom multi-factor regression, return attribution, rolling alpha/beta |
| **Optimization** | Max-Sharpe, min-variance, Equal Risk Contribution, efficient frontier |
| **Monte Carlo** | Multivariate-normal and bootstrap simulation with fan chart and risk summary |
| **Stress Tests** | Replay 8 historical crises (GFC, COVID, 2022 bear, etc.), custom shocks |
| **Correlation** | Pearson / Spearman / Kendall matrices, rolling correlations, hierarchical clustering |
| **Currency & Rates** | FX exposure by currency / country, US Treasury yield curve, bond-ETF proxies, FX pair explorer |
| **Risk Decomposition** | Systematic vs idiosyncratic split, concentration / tail / correlation diagnostics, rule-based improvement suggestions |
| **Pacing & Reup** | Per-position vintage tracker, composite reup signals, bootstrap forward probabilities |

Supports US and foreign equities (Japanese, Hong Kong, Chinese, Korean, Taiwanese, European, etc.), 130/30-style long-short and leveraged portfolios, and multi-currency exposure analysis.

## Requirements

- **Python 3.11 or newer** (3.14 tested)
- **Internet access** for fetching market data via Yahoo Finance
- Windows, macOS, or Linux

## First-time setup

```bash
# 1. Clone the repository
git clone <your-github-url>
cd "Equity Screener"

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
#    Windows:
.venv\Scripts\activate
#    macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies (one-time, ~2-3 minutes)
pip install -r requirements.txt
```

## Running the app

### Option A — double-click (Windows)

Double-click `run.bat`. The console opens, Streamlit starts, and your browser launches at `http://localhost:8501` automatically.

To stop: close the console window, or run `stop.bat`.

### Option B — command line (any OS)

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Option C — desktop shortcut (Windows)

Right-click `run.bat` → Send to → Desktop (create shortcut). Optionally drag the shortcut into `shell:startup` to launch on Windows login.

## Quick start workflow

1. **Open the Home page** — manual table is pre-populated with a sample US portfolio.
2. **Set Fund AUM** in the sidebar (default $1M).
3. **Click "Fetch & validate portfolio"** — this pulls 5 years of price data and the S&P 500 benchmark.
4. **Browse the 12 analysis pages** in the left sidebar. Each page reads the loaded portfolio from session state.

For long-short or leveraged portfolios (e.g. 130/30), turn on both **Allow shorts** and **Allow leverage** in the sidebar before submitting.

## CSV upload formats

The Home page accepts CSVs with flexible column names. Recognized:

- **Ticker column**: `ticker`, `symbol`, `stock`, `asset`, `code`
- **Description**: `description`, `name`, `company`
- **Weight**: `weight`, `allocation`, `pct`, `percent`, `%`
- **Shares**: `shares`, `qty`, `quantity`
- **Cost basis**: `cost_basis`, `cost`, `avg_price`

Weights can be decimals (`0.15`) or percentages (`15`) — the format is auto-detected. See `sample_portfolio.csv` for a working example.

## Project structure

```
.
├── app.py                          # Home / landing / portfolio builder
├── theme.py                        # Visual theme (plotly + CSS + badges)
├── data.py                         # Price fetching, currency catalog, FX pairs
├── portfolio_input.py              # CSV parsing & weight normalization
├── screener.py                     # Per-ticker fundamental metrics
├── scoring.py                      # Multi-factor composite score
├── analytics.py                    # Performance ratios & drawdown analytics
├── risk.py                         # VaR family, component risk, stress tests
├── stats_tests.py                  # Statistical hypothesis tests
├── factor_models.py                # CAPM & multi-factor regression
├── optimization.py                 # Mean-variance optimization
├── monte_carlo.py                  # Path simulation
├── pages/
│   ├── 1_Screener.py
│   ├── 2_Performance.py
│   ├── 3_Risk_Metrics.py
│   ├── 4_Statistical_Tests.py
│   ├── 5_Factor_Models.py
│   ├── 6_Optimization.py
│   ├── 7_Monte_Carlo.py
│   ├── 8_Stress_Tests.py
│   ├── 9_Correlation.py
│   ├── 10_Currency_and_Rates.py
│   ├── 11_Risk_Decomposition.py
│   └── 12_Pacing_and_Reup.py
├── .streamlit/
│   └── config.toml                 # Theme configuration
├── requirements.txt                # Python dependencies
├── sample_portfolio.csv            # Example CSV input
├── run.bat                         # Windows launcher
├── stop.bat                        # Windows kill-switch
└── README.md
```

## Caveats

- **Yahoo Finance is unofficial** — yfinance occasionally returns stale or missing fields, especially for small caps or recently-listed names. The code defends against missing data but expect occasional NaNs.
- **Backward-looking analytics** — Sharpe, alpha, beta, VaR are all historical estimates. Past performance doesn't guarantee future results.
- **Bootstrap forward probabilities** assume iid returns from the historical distribution — fragile in regime shifts.
- **Not financial advice** — this is a diagnostic tool for research. Combine with fundamental analysis and your own risk framework.

## License

Private — intended for internal use.
