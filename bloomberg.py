"""Parse Bloomberg-format multi-fund holdings files into clean, geo-tagged data.

Bloomberg identifiers look like 'SRCE US', '2018 HK', '2670 JP', or the long
form 'AAPL US Equity'. This module converts them to Yahoo Finance tickers and
tags each row with country / region derived from the exchange code — no API
call required, so it works on the full file instantly.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Trailing security-type tokens in Bloomberg's long form (TICKER EXCH Equity)
_SECURITY_TYPES = {"Equity", "Index", "Corp", "Comdty", "Curncy", "Govt", "Pfd", "Mtge"}

# Bloomberg exchange/composite code → (Yahoo suffix, Country, Region)
EXCHANGE_MAP: dict[str, tuple[str, str, str]] = {
    # ── North America ──
    "US": ("", "United States", "North America"),
    "UN": ("", "United States", "North America"),
    "UW": ("", "United States", "North America"),
    "UQ": ("", "United States", "North America"),
    "UA": ("", "United States", "North America"),
    "UR": ("", "United States", "North America"),
    "UP": ("", "United States", "North America"),
    "CN": (".TO", "Canada", "North America"),
    "CT": (".TO", "Canada", "North America"),
    "CV": (".V", "Canada", "North America"),
    "MM": (".MX", "Mexico", "North America"),
    "MF": (".MX", "Mexico", "North America"),
    # ── Europe ──
    "LN": (".L", "United Kingdom", "Europe"),
    "GR": (".DE", "Germany", "Europe"),
    "GY": (".DE", "Germany", "Europe"),
    "FP": (".PA", "France", "Europe"),
    "IM": (".MI", "Italy", "Europe"),
    "IT": (".MI", "Italy", "Europe"),
    "SM": (".MC", "Spain", "Europe"),
    "NA": (".AS", "Netherlands", "Europe"),
    "SW": (".SW", "Switzerland", "Europe"),
    "VX": (".SW", "Switzerland", "Europe"),
    "SE": (".SW", "Switzerland", "Europe"),
    "SS": (".ST", "Sweden", "Europe"),
    "NO": (".OL", "Norway", "Europe"),
    "DC": (".CO", "Denmark", "Europe"),
    "FH": (".HE", "Finland", "Europe"),
    "BB": (".BR", "Belgium", "Europe"),
    "AV": (".VI", "Austria", "Europe"),
    "PL": (".LS", "Portugal", "Europe"),
    "ID": (".IR", "Ireland", "Europe"),
    "PW": (".WA", "Poland", "Europe"),
    "GA": (".AT", "Greece", "Europe"),
    "RM": (".ME", "Russia", "Europe"),
    "RU": (".ME", "Russia", "Europe"),
    "TI": (".IS", "Turkey", "Europe"),
    "HB": (".BD", "Hungary", "Europe"),
    "CK": (".PR", "Czech Republic", "Europe"),
    "CP": (".PR", "Czech Republic", "Europe"),
    "LX": (".L", "Luxembourg", "Europe"),
    # ── Asia / Pacific ──
    "JP": (".T", "Japan", "Asia"),
    "JT": (".T", "Japan", "Asia"),
    "HK": (".HK", "Hong Kong", "Asia"),
    "C1": (".SS", "China", "Asia"),
    "C2": (".SZ", "China", "Asia"),
    "CH": (".SS", "China", "Asia"),
    "CG": (".SS", "China", "Asia"),
    "CS": (".SZ", "China", "Asia"),
    "KS": (".KS", "South Korea", "Asia"),
    "KP": (".KS", "South Korea", "Asia"),
    "TT": (".TW", "Taiwan", "Asia"),
    "SP": (".SI", "Singapore", "Asia"),
    "AU": (".AX", "Australia", "Oceania"),
    "AT": (".AX", "Australia", "Oceania"),
    "IN": (".NS", "India", "Asia"),
    "IS": (".NS", "India", "Asia"),
    "IB": (".BO", "India", "Asia"),
    "IJ": (".JK", "Indonesia", "Asia"),
    "MK": (".KL", "Malaysia", "Asia"),
    "PM": (".PS", "Philippines", "Asia"),
    "NZ": (".NZ", "New Zealand", "Oceania"),
    "TB": (".BK", "Thailand", "Asia"),
    "VN": (".VN", "Vietnam", "Asia"),
    "PA": (".KAR", "Pakistan", "Asia"),
    # ── Middle East / Africa ──
    "SJ": (".JO", "South Africa", "Africa"),
    "AB": (".SR", "Saudi Arabia", "Middle East"),
    "UH": (".AD", "United Arab Emirates", "Middle East"),
    "DU": (".AE", "United Arab Emirates", "Middle East"),
    "QD": (".QA", "Qatar", "Middle East"),
    "EY": (".CA", "Egypt", "Africa"),
    "IT_IL": (".TA", "Israel", "Middle East"),
    # ── South America ──
    "BZ": (".SA", "Brazil", "South America"),
    "BS": (".SA", "Brazil", "South America"),
    "CI": (".SN", "Chile", "South America"),
    "AR": (".BA", "Argentina", "South America"),
    "CB": (".CL", "Colombia", "South America"),
}


def parse_money(val) -> float:
    """Parse '$1,193.20', '(1,234.00)', or numeric → float. NaN on failure."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("$", "").replace(",", "").replace("(", "").replace(")", "").strip()
    if s in ("", "-", "—"):
        return np.nan
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return np.nan


def parse_bloomberg_ticker(raw) -> dict:
    """Parse a Bloomberg identifier into components.

    Examples:
        'SRCE US'         → symbol=SRCE, exch=US,  yahoo=SRCE
        'III LN'          → symbol=III,  exch=LN,  yahoo=III.L
        '2018 HK'         → symbol=2018, exch=HK,  yahoo=2018.HK
        'AAPL US Equity'  → symbol=AAPL, exch=US,  yahoo=AAPL  (Equity token stripped)
    """
    blank = {"symbol": "", "exch": "", "yahoo": "", "country": "Unknown",
             "region": "Unknown", "security_type": ""}
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return blank
    tokens = str(raw).strip().split()
    if not tokens:
        return blank

    security_type = ""
    if tokens[-1] in _SECURITY_TYPES:
        security_type = tokens[-1]
        tokens = tokens[:-1]
    if not tokens:
        return blank

    if len(tokens) == 1:
        symbol, exch = tokens[0], "US"          # no exchange code → assume US
    else:
        exch = tokens[-1]
        symbol = " ".join(tokens[:-1])

    suffix, country, region = EXCHANGE_MAP.get(exch, ("", "Unknown", "Unknown"))
    yahoo_symbol = symbol.replace("/", "-").replace(" ", "-")  # BRK/B → BRK-B
    yahoo = f"{yahoo_symbol}{suffix}"
    return {"symbol": symbol, "exch": exch, "yahoo": yahoo,
            "country": country, "region": region, "security_type": security_type}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_col(columns, aliases: list[str]) -> str | None:
    norm_map = {_norm(c): c for c in columns}
    for a in aliases:
        if _norm(a) in norm_map:
            return norm_map[_norm(a)]
    return None


def load_bloomberg_csv(file_or_buffer) -> dict:
    """Load a Bloomberg multi-fund holdings CSV.

    Returns {df, errors, warnings}. The cleaned df has columns:
        fund, raw_ticker, symbol, exch, yahoo, country, region,
        security_type, quantity (optional), market_value (optional)
    """
    out: dict = {"df": None, "errors": [], "warnings": []}
    try:
        df = pd.read_csv(file_or_buffer)
    except UnicodeDecodeError:
        try:
            file_or_buffer.seek(0)
        except Exception:
            pass
        df = pd.read_csv(file_or_buffer, encoding="latin-1")
    except Exception as e:
        out["errors"].append(f"Could not read CSV: {e}")
        return out

    if df.empty:
        out["errors"].append("File is empty.")
        return out

    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    fund_col = _find_col(df.columns, ["fund", "portfolio", "account", "strategy", "sleeve"])
    ticker_col = _find_col(df.columns, ["ticker", "security", "symbol", "identifier",
                                         "bbticker", "bloomberg"])
    qty_col = _find_col(df.columns, ["quantity", "shares", "qty", "position", "units"])
    mv_col = _find_col(df.columns, ["marketvalue", "value", "mv", "marketval",
                                     "notional", "exposure", "mktval"])

    if ticker_col is None:
        out["errors"].append(
            f"No ticker/security column found. Columns present: {list(df.columns)}"
        )
        return out
    if mv_col is None and qty_col is None:
        out["errors"].append("Need at least a 'Market Value' or 'Quantity' column.")
        return out

    rec = pd.DataFrame()
    rec["fund"] = df[fund_col].astype(str).str.strip() if fund_col else "Portfolio"
    rec["raw_ticker"] = df[ticker_col].astype(str).str.strip()

    parsed = rec["raw_ticker"].apply(parse_bloomberg_ticker)
    rec["symbol"] = parsed.apply(lambda d: d["symbol"])
    rec["exch"] = parsed.apply(lambda d: d["exch"])
    rec["yahoo"] = parsed.apply(lambda d: d["yahoo"])
    rec["country"] = parsed.apply(lambda d: d["country"])
    rec["region"] = parsed.apply(lambda d: d["region"])
    rec["security_type"] = parsed.apply(lambda d: d["security_type"])

    if qty_col:
        rec["quantity"] = pd.to_numeric(df[qty_col], errors="coerce")
    if mv_col:
        rec["market_value"] = df[mv_col].apply(parse_money)

    # Drop rows with no usable value
    if "market_value" in rec.columns:
        n_bad = int(rec["market_value"].isna().sum())
        if n_bad:
            out["warnings"].append(f"{n_bad} rows had unparseable market value and were dropped.")
            rec = rec.dropna(subset=["market_value"])
        rec = rec[rec["market_value"] != 0]

    if rec.empty:
        out["errors"].append("No usable rows after cleaning.")
        return out

    n_unknown = int((rec["country"] == "Unknown").sum())
    if n_unknown:
        unknown_codes = sorted(rec.loc[rec["country"] == "Unknown", "exch"].unique())
        out["warnings"].append(
            f"{n_unknown} rows from {len(unknown_codes)} unmapped exchange code(s): "
            f"{', '.join(unknown_codes[:20])}"
            + (" ..." if len(unknown_codes) > 20 else "")
            + " — grouped as 'Unknown' in the geographic breakdown."
        )

    out["df"] = rec.reset_index(drop=True)
    return out


def looks_like_bloomberg(df: pd.DataFrame) -> bool:
    """Heuristic: does this raw DataFrame look like a Bloomberg multi-fund file?

    True if there's a market-value column AND the ticker column contains
    space-separated Bloomberg identifiers (e.g. 'AAPL US').
    """
    if df is None or df.empty:
        return False
    cols = list(df.columns)
    ticker_col = _find_col(cols, ["ticker", "security", "symbol", "identifier", "bloomberg"])
    mv_col = _find_col(cols, ["marketvalue", "value", "mv", "marketval", "notional",
                               "exposure", "mktval"])
    if ticker_col is None:
        return False
    sample = df[ticker_col].astype(str).head(50)
    # Bloomberg identifiers are "SYMBOL EXCH" — a space with a known exchange code
    space_with_code = sample.apply(
        lambda s: len(s.split()) >= 2 and (
            s.split()[-1] in EXCHANGE_MAP or s.split()[-1] in _SECURITY_TYPES
        )
    )
    has_bb_tickers = space_with_code.mean() > 0.3
    return bool(mv_col is not None and has_bb_tickers)


def aggregate_to_portfolio(df: pd.DataFrame, top_n: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Aggregate a parsed holdings df (possibly many funds) into ONE combined
    portfolio keyed by Yahoo ticker, with weights computed from Market Value.

    Returns (portfolio_df, meta) where portfolio_df has columns
    [ticker, description, weight, market_value] and meta carries coverage info.
    """
    work = df.copy()
    if "market_value" not in work.columns:
        raise ValueError("aggregate_to_portfolio requires a market_value column.")
    # Only equity-like rows with a resolved Yahoo ticker
    work = work[work["yahoo"].astype(str).str.strip() != ""]

    agg = (work.groupby("yahoo", as_index=False)
                .agg(market_value=("market_value", "sum"),
                     symbol=("symbol", "first"),
                     country=("country", "first"),
                     region=("region", "first"),
                     raw_ticker=("raw_ticker", "first"),
                     n_funds=("fund", "nunique")))
    agg = agg[agg["market_value"] > 0]
    agg = agg.sort_values("market_value", ascending=False).reset_index(drop=True)

    total_mv = float(agg["market_value"].sum())
    n_total = len(agg)
    coverage = 1.0

    if top_n is not None and n_total > top_n:
        kept = agg.head(top_n).copy()
        coverage = float(kept["market_value"].sum() / total_mv) if total_mv else 0.0
        agg = kept

    agg["weight"] = agg["market_value"] / agg["market_value"].sum()
    # Build a readable description
    def _describe(row):
        base = f"{row['symbol']} · {row['country']}"
        if row["n_funds"] > 1:
            base += f" · in {int(row['n_funds'])} funds"
        return base
    agg["description"] = agg.apply(_describe, axis=1)

    portfolio = agg[["yahoo", "description", "weight", "market_value"]].rename(
        columns={"yahoo": "ticker"}
    )
    meta = {
        "n_total_unique": n_total,
        "n_kept": len(portfolio),
        "coverage": coverage,
        "total_mv": total_mv,
        "n_funds": int(work["fund"].nunique()),
        "n_positions": len(work),
    }
    return portfolio.reset_index(drop=True), meta
