"""Single-equity research debrief — Bloomberg DES-style tearsheet.

Pulls everything yfinance knows about a single ticker and lays it out in the
dense one-page format institutional analysts use: company description,
classification, key valuation/profitability/health metrics, price chart with
moving averages, multi-period returns, recent financials, analyst targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import format_market_cap
from portfolio_input import clean_ticker, is_valid_ticker
from screener import fetch_financials, fetch_history, fetch_info
from theme import inject_css

inject_css()

st.title("Equity Tearsheet")
st.caption("Single-equity research debrief — description, classification, valuation, financials, and price action.")

# ─── Ticker entry ────────────────────────────────────────────────────────────
st.sidebar.header("Ticker")
default_ticker = st.session_state.get("eq_tearsheet_ticker", "AAPL")
ticker_input = st.sidebar.text_input(
    "Yahoo Finance symbol", value=default_ticker,
    help=(
        "US: AAPL, BRK-B, MSFT\n"
        "London: BARC.L\n"
        "Tokyo: 7203.T\n"
        "Hong Kong: 0700.HK\n"
        "Taiwan: 2330.TW\n"
        "Korea: 005930.KS"
    ),
)
ticker = clean_ticker(ticker_input)
period = st.sidebar.selectbox(
    "Price history period", ["1y", "2y", "5y", "10y", "max"], index=2,
)

if not ticker or not is_valid_ticker(ticker):
    st.warning("Enter a valid Yahoo Finance ticker symbol in the sidebar.")
    st.stop()

st.session_state["eq_tearsheet_ticker"] = ticker

# ─── Fetch ───────────────────────────────────────────────────────────────────
with st.spinner(f"Pulling {ticker} data..."):
    info = fetch_info(ticker)
    history = fetch_history(ticker, period=period)
    financials = fetch_financials(ticker)

# Treat the ticker as having "no data" only when BOTH the info payload and
# the price history are empty. Yahoo frequently returns an empty info dict
# for valid tickers when called from cloud IPs — in that case we can still
# render a meaningful tearsheet from price history alone.
info_has_signal = bool(info) and any(
    info.get(k) is not None
    for k in ("marketCap", "currentPrice", "regularMarketPrice",
              "shortName", "longName", "sharesOutstanding")
)
if not info_has_signal and history.empty:
    st.error(
        f"No data returned for **{ticker}**. This usually means one of:\n\n"
        "• The symbol is wrong — foreign tickers need exchange suffixes "
        "(e.g. `7203.T`, `BARC.L`, `0700.HK`).\n"
        "• Yahoo Finance is rate-limiting this server. Wait a minute and "
        "try again, or refresh the page to retry."
    )
    st.stop()

if not info_has_signal:
    # We have history but no rich info — warn the user that some fundamental
    # fields will read "—" while still rendering price-based panels.
    st.info(
        "Yahoo Finance returned only price data for this ticker. Fundamental "
        "fields below may show \"—\" while the rich-info endpoint is "
        "unavailable. Refresh the page in a minute to retry."
    )


def _get(key, default=None):
    return info.get(key, default)


def _fmt_money(v, currency: str = "USD") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format_market_cap(v, currency)


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.2%}"


def _fmt_num(v, fmt: str = "{:.2f}") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return "—"


# ─── Headline header ─────────────────────────────────────────────────────────
name = _get("longName") or _get("shortName") or ticker
country = _get("country", "")
exchange = _get("exchange", "") or _get("fullExchangeName", "")
currency = (_get("currency") or "USD").upper()
sector = _get("sector", "—")
industry = _get("industry", "—")
website = _get("website", "")

price = _get("currentPrice") or _get("regularMarketPrice")
prev_close = _get("regularMarketPreviousClose") or _get("previousClose")
day_change = ((price / prev_close) - 1) if (price and prev_close) else None
day_change_abs = (price - prev_close) if (price and prev_close) else None

st.markdown(f"## {name}")
st.caption(
    f"**{ticker}**  ·  {country or '—'}  ·  {exchange or '—'}  ·  "
    f"Listed in {currency}  ·  **{sector}** · {industry}"
    + (f"  ·  [Company website]({website})" if website else "")
)

c = st.columns(5)
c[0].metric(f"Price ({currency})", _fmt_num(price, "{:,.2f}"),
              f"{day_change:+.2%}" if day_change is not None else None)
c[1].metric("Market Cap", _fmt_money(_get("marketCap"), currency))
c[2].metric("52-week high", _fmt_num(_get("fiftyTwoWeekHigh"), "{:,.2f}"))
c[3].metric("52-week low", _fmt_num(_get("fiftyTwoWeekLow"), "{:,.2f}"))
c[4].metric("Beta (5Y)", _fmt_num(_get("beta")))

st.markdown("---")

# ─── Tabs: Description / Valuation / Health / Price / Financials / Analysts ──
tab_desc, tab_val, tab_health, tab_price, tab_financials, tab_analyst = st.tabs([
    "Description", "Valuation", "Financial Health", "Price & Returns",
    "Financials", "Analyst Targets",
])

# Tab 1: Description
with tab_desc:
    st.markdown("##### Business description")
    summary = _get("longBusinessSummary") or "_No business description available from Yahoo Finance for this ticker._"
    st.write(summary)

    st.markdown("---")
    st.markdown("##### Company profile")
    address_parts = [_get("address1"), _get("city"), _get("state"), _get("country"), _get("zip")]
    address = ", ".join(p for p in address_parts if p)
    profile = {
        "Sector": sector,
        "Industry": industry,
        "Country": country,
        "Exchange": exchange,
        "Currency": currency,
        "Employees": f"{_get('fullTimeEmployees', 0):,}" if _get("fullTimeEmployees") else "—",
        "Headquarters": address or "—",
        "Phone": _get("phone", "—") or "—",
        "Website": website or "—",
    }
    pdf = pd.DataFrame(list(profile.items()), columns=["Field", "Value"])
    st.dataframe(pdf, hide_index=True, width="stretch")

    # Key officers
    officers = _get("companyOfficers") or []
    if officers:
        st.markdown("##### Key officers")
        rows = []
        for o in officers[:10]:
            rows.append({
                "Name": o.get("name", "—"),
                "Title": o.get("title", "—"),
                "Age": o.get("age", "—"),
                "Total pay": _fmt_money(o.get("totalPay"), currency)
                              if o.get("totalPay") else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# Tab 2: Valuation
with tab_val:
    st.markdown("##### Valuation multiples")
    c = st.columns(4)
    c[0].metric("P/E (trailing)", _fmt_num(_get("trailingPE")))
    c[1].metric("P/E (forward)", _fmt_num(_get("forwardPE")))
    c[2].metric("PEG ratio", _fmt_num(_get("pegRatio") or _get("trailingPegRatio")))
    c[3].metric("Price / Book", _fmt_num(_get("priceToBook")))

    c = st.columns(4)
    c[0].metric("Price / Sales (TTM)", _fmt_num(_get("priceToSalesTrailing12Months")))
    c[1].metric("EV / EBITDA", _fmt_num(_get("enterpriseToEbitda")))
    c[2].metric("EV / Revenue", _fmt_num(_get("enterpriseToRevenue")))
    c[3].metric("Enterprise Value", _fmt_money(_get("enterpriseValue"), currency))

    st.markdown("---")
    st.markdown("##### Dividend & shareholder returns")
    c = st.columns(4)
    c[0].metric("Dividend yield", _fmt_pct(_get("dividendYield") / 100 if _get("dividendYield") and _get("dividendYield") > 1 else _get("dividendYield")))
    c[1].metric("Trailing div / share", _fmt_num(_get("trailingAnnualDividendRate"), "{:,.2f}"))
    c[2].metric("Payout ratio", _fmt_pct(_get("payoutRatio")))
    c[3].metric("5Y avg div yield", _fmt_pct(_get("fiveYearAvgDividendYield") / 100 if _get("fiveYearAvgDividendYield") and _get("fiveYearAvgDividendYield") > 1 else _get("fiveYearAvgDividendYield")))

    st.markdown("---")
    st.markdown("##### Share data")
    c = st.columns(4)
    c[0].metric("Shares outstanding", _fmt_money(_get("sharesOutstanding"), ""))
    c[1].metric("Float shares", _fmt_money(_get("floatShares"), ""))
    c[2].metric("Held by insiders", _fmt_pct(_get("heldPercentInsiders")))
    c[3].metric("Held by institutions", _fmt_pct(_get("heldPercentInstitutions")))

# Tab 3: Financial Health
with tab_health:
    st.markdown("##### Profitability")
    c = st.columns(4)
    c[0].metric("Return on Equity", _fmt_pct(_get("returnOnEquity")))
    c[1].metric("Return on Assets", _fmt_pct(_get("returnOnAssets")))
    c[2].metric("Gross margin", _fmt_pct(_get("grossMargins")))
    c[3].metric("Net profit margin", _fmt_pct(_get("profitMargins")))

    c = st.columns(4)
    c[0].metric("Operating margin", _fmt_pct(_get("operatingMargins")))
    c[1].metric("EBITDA margin", _fmt_pct(_get("ebitdaMargins")))
    c[2].metric("Revenue (TTM)", _fmt_money(_get("totalRevenue"), currency))
    c[3].metric("Net income", _fmt_money(_get("netIncomeToCommon"), currency))

    st.markdown("---")
    st.markdown("##### Balance sheet & liquidity")
    de = _get("debtToEquity")
    de_disp = f"{de / 100:.2f}" if de and de > 5 else _fmt_num(de)
    c = st.columns(4)
    c[0].metric("Debt / Equity", de_disp)
    c[1].metric("Current ratio", _fmt_num(_get("currentRatio")))
    c[2].metric("Quick ratio", _fmt_num(_get("quickRatio")))
    c[3].metric("Total cash", _fmt_money(_get("totalCash"), currency))

    c = st.columns(4)
    c[0].metric("Total debt", _fmt_money(_get("totalDebt"), currency))
    c[1].metric("Operating cash flow", _fmt_money(_get("operatingCashflow"), currency))
    c[2].metric("Free cash flow", _fmt_money(_get("freeCashflow"), currency))
    c[3].metric("EBITDA", _fmt_money(_get("ebitda"), currency))

    st.markdown("---")
    st.markdown("##### Growth")
    c = st.columns(4)
    c[0].metric("Revenue growth (YoY)", _fmt_pct(_get("revenueGrowth")))
    c[1].metric("Earnings growth (YoY)", _fmt_pct(_get("earningsGrowth")))
    c[2].metric("EPS (TTM)", _fmt_num(_get("trailingEps"), "{:.2f}"))
    c[3].metric("EPS (forward)", _fmt_num(_get("forwardEps"), "{:.2f}"))

# Tab 4: Price & Returns
with tab_price:
    if history.empty or "Close" not in history.columns:
        st.warning(f"No price history available for {ticker}.")
    else:
        close = history["Close"]

        st.markdown(f"##### {ticker} — {period} price history")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Close",
                                    line=dict(color="#1e40af", width=2)))
        if len(close) >= 50:
            fig.add_trace(go.Scatter(x=close.index, y=close.rolling(50).mean(),
                                        name="50-day MA",
                                        line=dict(color="#FF8200", dash="dash", width=1.5)))
        if len(close) >= 200:
            fig.add_trace(go.Scatter(x=close.index, y=close.rolling(200).mean(),
                                        name="200-day MA",
                                        line=dict(color="#b91c1c", dash="dash", width=1.5)))
        # 52-week markers
        w52_high = _get("fiftyTwoWeekHigh")
        w52_low = _get("fiftyTwoWeekLow")
        if w52_high:
            fig.add_hline(y=w52_high, line_dash="dot", line_color="#15803d",
                            annotation_text="52W high", annotation_position="right")
        if w52_low:
            fig.add_hline(y=w52_low, line_dash="dot", line_color="#b45309",
                            annotation_text="52W low", annotation_position="right")
        fig.update_layout(height=460, hovermode="x unified",
                            yaxis_title=f"Price ({currency})")
        st.plotly_chart(fig, width="stretch")

        # Volume sub-chart
        if "Volume" in history.columns:
            fig = px.bar(history["Volume"], title="Trading volume")
            fig.update_layout(showlegend=False, height=200)
            st.plotly_chart(fig, width="stretch")

        # Multi-period return table
        st.markdown("##### Period returns")
        def _ret(days: int) -> float:
            if len(close) < days + 1:
                return np.nan
            return float(close.iloc[-1] / close.iloc[-days - 1] - 1)

        ytd_start = close[close.index.year == close.index[-1].year]
        ytd = float(close.iloc[-1] / ytd_start.iloc[0] - 1) if len(ytd_start) > 0 else np.nan

        ret_rows = {
            "1D": _ret(1), "1W": _ret(5), "1M": _ret(21), "3M": _ret(63),
            "YTD": ytd, "1Y": _ret(252), "3Y": _ret(756), "5Y": _ret(1260),
        }
        ret_df = pd.DataFrame([{k: f"{v:+.2%}" if pd.notna(v) else "—"
                                  for k, v in ret_rows.items()}])
        st.dataframe(ret_df, hide_index=True, width="stretch")

        # Distance from 52w high/low
        last_close = float(close.iloc[-1])
        c = st.columns(3)
        if w52_high:
            c[0].metric("% from 52W high",
                          f"{(last_close / w52_high - 1):+.2%}")
        if w52_low:
            c[1].metric("% from 52W low",
                          f"{(last_close / w52_low - 1):+.2%}")
        c[2].metric("Current close", _fmt_num(last_close, "{:,.2f}"))

# Tab 5: Financials
with tab_financials:
    inc = financials.get("income", pd.DataFrame())
    bs = financials.get("balance", pd.DataFrame())
    cf = financials.get("cashflow", pd.DataFrame())

    if inc.empty:
        st.info("Annual income-statement data not available for this ticker.")
    else:
        st.markdown("##### Annual income statement")
        keep_rows = ["Total Revenue", "Gross Profit", "Operating Income",
                       "Net Income", "EBITDA", "EBIT", "Diluted EPS",
                       "Basic EPS", "Operating Expense"]
        avail = [r for r in keep_rows if r in inc.index]
        disp_inc = inc.loc[avail].copy()
        # Format numerics with money formatting
        for col in disp_inc.columns:
            disp_inc[col] = disp_inc[col].apply(
                lambda v: _fmt_money(v, currency) if abs(v or 0) > 1000 else _fmt_num(v, "{:,.2f}")
            )
        # Columns are dates — format
        disp_inc.columns = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c)
                              for c in disp_inc.columns]
        st.dataframe(disp_inc, width="stretch")

    if not bs.empty:
        st.markdown("##### Annual balance sheet")
        keep_rows = ["Total Assets", "Total Liabilities Net Minority Interest",
                       "Stockholders Equity", "Total Debt", "Net Debt",
                       "Cash And Cash Equivalents", "Working Capital",
                       "Current Assets", "Current Liabilities"]
        avail = [r for r in keep_rows if r in bs.index]
        if avail:
            disp_bs = bs.loc[avail].copy()
            for col in disp_bs.columns:
                disp_bs[col] = disp_bs[col].apply(
                    lambda v: _fmt_money(v, currency) if abs(v or 0) > 1000 else _fmt_num(v, "{:,.2f}")
                )
            disp_bs.columns = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c)
                                 for c in disp_bs.columns]
            st.dataframe(disp_bs, width="stretch")

    if not cf.empty:
        st.markdown("##### Annual cash flow")
        keep_rows = ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure",
                       "Investing Cash Flow", "Financing Cash Flow",
                       "Repurchase Of Capital Stock", "Cash Dividends Paid"]
        avail = [r for r in keep_rows if r in cf.index]
        if avail:
            disp_cf = cf.loc[avail].copy()
            for col in disp_cf.columns:
                disp_cf[col] = disp_cf[col].apply(
                    lambda v: _fmt_money(v, currency) if abs(v or 0) > 1000 else _fmt_num(v, "{:,.2f}")
                )
            disp_cf.columns = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c)
                                 for c in disp_cf.columns]
            st.dataframe(disp_cf, width="stretch")

# Tab 6: Analyst targets
with tab_analyst:
    st.markdown("##### Sell-side analyst consensus")
    rec_key = _get("recommendationKey", "")
    rec_mean = _get("recommendationMean")
    num_analysts = _get("numberOfAnalystOpinions")

    c = st.columns(4)
    c[0].metric("Mean recommendation", rec_key.replace("_", " ").title() if rec_key else "—",
                  help="1 = Strong Buy, 5 = Strong Sell")
    c[1].metric("Recommendation score", _fmt_num(rec_mean, "{:.2f}"))
    c[2].metric("# analysts", num_analysts if num_analysts else "—")
    c[3].metric("Current price", _fmt_num(price, "{:,.2f}"))

    st.markdown("##### Price targets")
    c = st.columns(4)
    c[0].metric("Target mean", _fmt_num(_get("targetMeanPrice"), "{:,.2f}"))
    c[1].metric("Target median", _fmt_num(_get("targetMedianPrice"), "{:,.2f}"))
    c[2].metric("Target high", _fmt_num(_get("targetHighPrice"), "{:,.2f}"))
    c[3].metric("Target low", _fmt_num(_get("targetLowPrice"), "{:,.2f}"))

    tmp = _get("targetMeanPrice")
    if tmp and price:
        upside = (tmp / price - 1)
        st.metric("Implied upside (vs mean target)", f"{upside:+.2%}",
                    help="Mean analyst price target / current price − 1")

    # Earnings estimates table
    st.markdown("---")
    st.markdown("##### Key forward expectations")
    fwd = {
        "Forward P/E": _fmt_num(_get("forwardPE")),
        "Forward EPS": _fmt_num(_get("forwardEps"), "{:.2f}"),
        "PEG": _fmt_num(_get("pegRatio") or _get("trailingPegRatio")),
        "Earnings growth (YoY)": _fmt_pct(_get("earningsGrowth")),
        "Revenue growth (YoY)": _fmt_pct(_get("revenueGrowth")),
        "Earnings Q growth (YoY)": _fmt_pct(_get("earningsQuarterlyGrowth")),
    }
    st.dataframe(pd.DataFrame(list(fwd.items()), columns=["Metric", "Value"]),
                   hide_index=True, width="stretch")
