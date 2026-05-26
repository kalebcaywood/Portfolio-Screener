"""Portfolio input parsing, validation, and weight normalization."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

# Allowed ticker forms (Yahoo Finance grammar):
#   - US equity: AAPL, MSFT, NVDA, BRK-B, BF-B
#   - US index: ^GSPC, ^VIX, ^N225, ^FTSE, ^HSI, ^STOXX50E
#   - FX: EURUSD=X, USDJPY=X, GBPUSD=X
#   - Futures: DX=F, ES=F, GC=F
#   - Crypto: BTC-USD, ETH-EUR
#   - Tokyo: 7203.T (Toyota), 9984.T (SoftBank) — all-numeric main
#   - Hong Kong: 0700.HK (Tencent), 9988.HK (Alibaba) — 4-digit
#   - Shanghai/Shenzhen: 600519.SS (Moutai), 000858.SZ — 6-digit
#   - Korea: 005930.KS (Samsung), 035420.KQ (Naver) — 6-digit
#   - Taiwan: 2330.TW (TSMC), 2317.TW (Foxconn)
#   - India: RELIANCE.NS, TCS.BO — long alpha main
#   - London: BARC.L, ULVR.L, AZN.L
#   - Continental Europe: SAP.DE, BNP.PA, MC.PA, ASML.AS, NESN.SW, ENI.MI
#   - Multi-segment: VOLV-B.ST (Volvo Stockholm)
#   - Australia: BHP.AX, CBA.AX
#   - Brazil: PETR4.SA, VALE3.SA — alphanumeric main
#   - Canada: SHOP.TO, RY.TO
TICKER_RE = re.compile(
    r"^("
    r"\^[A-Z0-9.]{1,12}"                                    # Indices: ^GSPC, ^STOXX50E
    r"|[A-Z]{6,8}=X"                                         # FX pairs: EURUSD=X
    r"|[A-Z]{2,5}=F"                                         # Futures: DX=F, ES=F
    r"|[A-Z0-9]{2,8}-[A-Z]{3,4}"                            # Crypto: BTC-USD, ETH-EUR
    r"|[A-Z0-9]{1,10}(?:[-.][A-Z0-9]{1,5}){0,2}"           # Equities (incl. all-numeric, 0-2 suffixes)
    r")$"
)

TICKER_ALIASES = {"ticker", "symbol", "stock", "asset", "code", "security", "instrument"}
DESC_ALIASES = {"description", "desc", "name", "company", "fullname", "longname", "shortname", "label"}
WEIGHT_ALIASES = {"weight", "allocation", "alloc", "pct", "percent", "share", "targetweight",
                  "target_weight", "wt", "%"}
SHARES_ALIASES = {"shares", "qty", "quantity", "units", "position", "size", "lots"}
COST_ALIASES = {"cost_basis", "cost", "avg_price", "average_cost", "price_paid", "basis",
                "cost_per_share", "avg_cost"}


def _norm_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_column(columns: Iterable[str], aliases: set[str]) -> str | None:
    norm_aliases = {_norm_colname(a) for a in aliases}
    for c in columns:
        if _norm_colname(c) in norm_aliases:
            return c
    return None


def clean_ticker(t) -> str:
    """Normalize a ticker symbol.

    Uppercases and strips whitespace. For US share classes only, converts
    BRK.B → BRK-B (Yahoo's preferred form). Does NOT touch foreign tickers
    like 7203.T, 0700.HK, BARC.L — the dot there is the exchange suffix,
    not a share-class separator.
    """
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    s = str(t).strip().upper()
    # Only convert ".A/.B/.C" patterns on short pure-letter mains (US share classes).
    # This deliberately leaves "BT.L" (UK), "PTT.BK" (Thai), "7203.T" (Japan) etc. untouched.
    if re.match(r"^[A-Z]{2,4}\.[ABC]$", s):
        s = s.replace(".", "-")
    return s


def is_valid_ticker(t: str) -> bool:
    return bool(t) and bool(TICKER_RE.match(t))


def parse_csv(file_or_buffer) -> dict:
    """Parse a portfolio CSV with flexible column naming.

    Returns:
        {
            "df": cleaned DataFrame with [ticker, description, ...],
            "mode": "weight" | "shares" | "equal",
            "errors": list[str],
            "warnings": list[str],
        }
    """
    out: dict = {"df": None, "mode": None, "errors": [], "warnings": []}

    try:
        df = pd.read_csv(file_or_buffer)
    except UnicodeDecodeError:
        try:
            file_or_buffer.seek(0)
        except Exception:
            pass
        try:
            df = pd.read_csv(file_or_buffer, encoding="latin-1")
        except Exception as e:
            out["errors"].append(f"Could not read CSV (encoding issue): {e}")
            return out
    except Exception as e:
        out["errors"].append(f"Could not parse CSV: {e}")
        return out

    if df.empty:
        out["errors"].append("CSV is empty.")
        return out

    df.columns = [str(c).strip() for c in df.columns]
    ticker_col = _find_column(df.columns, TICKER_ALIASES)
    if ticker_col is None:
        out["errors"].append(
            f"No ticker column found. Expected one of: {sorted(TICKER_ALIASES)}. "
            f"Got columns: {list(df.columns)}"
        )
        return out

    desc_col = _find_column(df.columns, DESC_ALIASES)
    weight_col = _find_column(df.columns, WEIGHT_ALIASES)
    shares_col = _find_column(df.columns, SHARES_ALIASES)
    cost_col = _find_column(df.columns, COST_ALIASES)

    res = pd.DataFrame({"ticker": df[ticker_col].apply(clean_ticker)})
    res["description"] = (df[desc_col].astype(str).str.strip() if desc_col is not None
                            else "")

    if weight_col is not None:
        cleaned = (df[weight_col].astype(str).str.replace("%", "", regex=False)
                                              .str.replace(",", "", regex=False)
                                              .str.strip())
        res["weight"] = pd.to_numeric(cleaned, errors="coerce")
        out["mode"] = "weight"
    elif shares_col is not None:
        res["shares"] = pd.to_numeric(df[shares_col], errors="coerce")
        if cost_col is not None:
            res["cost_basis"] = pd.to_numeric(df[cost_col], errors="coerce")
        out["mode"] = "shares"
        out["warnings"].append(
            "Using 'shares' column — weights will be computed as shares × latest price."
        )
    else:
        out["mode"] = "equal"
        out["warnings"].append(
            f"No weight or shares column found. Recognized weight columns: "
            f"{sorted(WEIGHT_ALIASES)}. Equal weighting will be applied."
        )

    # Validate ticker format
    bad_mask = ~res["ticker"].apply(is_valid_ticker)
    if bad_mask.any():
        bad = res.loc[bad_mask, "ticker"].tolist()
        out["warnings"].append(
            f"Removed {bad_mask.sum()} invalid ticker symbol(s): "
            f"{', '.join(repr(t) for t in bad[:10])}"
            + (" ..." if len(bad) > 10 else "")
        )
        res = res[~bad_mask]

    if res.empty:
        out["errors"].append("No valid tickers remain after validation.")
        return out

    # Consolidate duplicate tickers
    if res["ticker"].duplicated().any():
        dups = res.loc[res["ticker"].duplicated(keep=False), "ticker"].unique().tolist()
        out["warnings"].append(f"Consolidated duplicate ticker rows: {', '.join(dups)}")
        agg: dict = {"description": "first"}
        for col in ("weight", "shares"):
            if col in res.columns:
                agg[col] = "sum"
        if "cost_basis" in res.columns:
            agg["cost_basis"] = "mean"
        res = res.groupby("ticker", as_index=False).agg(agg)

    out["df"] = res.reset_index(drop=True)
    return out


def normalize_weights(weights: pd.Series, allow_short: bool = False,
                       allow_leverage: bool = False) -> tuple[pd.Series, list[str]]:
    """Clean and normalize a weight series.

    Behavior by mode:
        long-only, no leverage:  w / abs_sum  → gross = net = 100%
        long-short, no leverage: w / net_sum  → net = 100%, gross can be > 100%
        any-side, leverage on:   no rescaling (after percent detection) — respects
                                  130/30, market-neutral, leveraged-long, etc.

    Percent auto-detect: if absolute sum is in [5, 500] the input is treated as
    percentages (so 130/30 in percent → decimals).
    """
    info: list[str] = []
    w = pd.to_numeric(weights, errors="coerce")

    n_nan = int(w.isna().sum())
    if n_nan:
        info.append(f"{n_nan} weight(s) were missing or non-numeric — replaced with 0.")
        w = w.fillna(0)

    if (w == 0).all():
        raise ValueError("All weights are zero. Provide non-zero weights or use equal weighting.")

    if not allow_short and (w < 0).any():
        n_neg = int((w < 0).sum())
        info.append(f"{n_neg} negative weight(s) clipped to 0 (enable 'Allow shorts' to keep them).")
        w = w.clip(lower=0)

    signed = float(w.sum())
    abs_sum = float(w.abs().sum())
    if abs_sum == 0:
        raise ValueError("All weights are zero after cleaning.")

    # Percent auto-detect — extended upper bound so 130/30-style gross (≈160) is caught
    if 5 <= abs_sum <= 500:
        info.append(
            f"Weights summed to {signed:.2f} (gross {abs_sum:.2f}) — interpreted as percentages."
        )
        w = w / 100
        signed = float(w.sum())
        abs_sum = float(w.abs().sum())

    if allow_leverage:
        # Respect the input as-given. Just report exposure metrics.
        if abs(signed) < 1e-4:
            info.append(f"Market-neutral portfolio: net ≈ 0, gross {abs_sum:.2%}.")
        else:
            lev = abs_sum / abs(signed)
            info.append(
                f"Net exposure: {signed:.2%}, gross exposure: {abs_sum:.2%}, leverage: {lev:.2f}×"
            )
        return w, info

    if not allow_short:
        # Long-only: gross == net, normalize to 1.0
        if abs(abs_sum - 1.0) > 0.005:
            info.append(f"Normalized to gross/net exposure = 100% (was {abs_sum:.4f}).")
        w = w / abs_sum
    else:
        # Long-short without leverage: normalize to net = 1.0
        if signed <= 0.005:
            raise ValueError(
                f"Net exposure is {signed:.4f}; cannot normalize to 100% net. "
                "For market-neutral or 130/30-style strategies, enable "
                "**Allow leverage** in the sidebar — that keeps your weights as-entered."
            )
        if abs(signed - 1.0) > 0.005:
            info.append(f"Normalized to net exposure = 100% (was {signed:.4f}).")
        w = w / signed

    return w, info


def validate_price_data(prices: pd.DataFrame, tickers: list[str],
                         min_obs: int = 30) -> tuple[list[str], list[str], list[str]]:
    """Validate fetched price data.

    Returns (final_valid_tickers, missing_tickers, low_history_descriptions).
    """
    missing = [t for t in tickers if t not in prices.columns]
    present = [t for t in tickers if t in prices.columns]

    final: list[str] = []
    low: list[str] = []
    for t in present:
        n = int(prices[t].dropna().shape[0])
        if n < min_obs:
            low.append(f"{t} ({n}d)")
        else:
            final.append(t)
    return final, missing, low


def df_to_weight_series(df: pd.DataFrame, mode: str,
                         latest_prices: pd.Series | None = None) -> pd.Series:
    """Turn a parsed portfolio DataFrame into a raw weight series."""
    if mode == "shares" and "shares" in df.columns and latest_prices is not None:
        d = df.copy()
        d["price"] = d["ticker"].map(latest_prices)
        d["value"] = d["shares"] * d["price"]
        d = d.dropna(subset=["value"])
        return d.set_index("ticker")["value"]
    if mode == "weight" and "weight" in df.columns:
        return df.set_index("ticker")["weight"]
    return pd.Series(1.0, index=df["ticker"].tolist())
