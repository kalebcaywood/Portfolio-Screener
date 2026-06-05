"""Preset screening universes — curated ticker lists for one-click browsing.

Only tickers are needed: the screener fetches name / sector / industry / country
live from yfinance, so these lists stay small and reliable (no external
constituent scraping, no extra dependencies). Lists favour liquid, large-cap
leaders that resolve cleanly on Yahoo Finance.
"""
from __future__ import annotations

# ── Dow Jones Industrial Average (30) ────────────────────────────────────────
DOW_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]

# ── Mega-cap leaders (≈ the 50 largest U.S. companies) ────────────────────────
MEGACAP_50 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO",
    "TSLA", "JPM", "WMT", "V", "UNH", "XOM", "MA", "PG", "JNJ", "HD", "COST",
    "ORCL", "MRK", "ABBV", "CVX", "KO", "AMD", "PEP", "ADBE", "CRM", "BAC",
    "NFLX", "TMO", "ACN", "LIN", "MCD", "ABT", "CSCO", "WFC", "DHR", "QCOM",
    "TXN", "PM", "DIS", "IBM", "GE", "CAT", "VZ", "NOW", "INTU", "ISRG",
]

# ── GICS sector baskets (sector leaders) ─────────────────────────────────────
SECTORS: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO",
        "ACN", "TXN", "QCOM", "INTC", "IBM", "NOW", "INTU", "AMAT", "MU",
        "LRCX", "ADI",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "GM",
        "F", "MAR",
    ],
    "Consumer Staples": [
        "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "TGT",
        "KMB", "GIS",
    ],
    "Financials": [
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SPGI",
        "BLK", "C", "SCHW", "CB",
    ],
    "Health Care": [
        "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "PFE", "AMGN",
        "BMY", "CVS", "MDT", "ISRG", "GILD",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "VLO", "WMB",
        "KMI",
    ],
    "Industrials": [
        "GE", "CAT", "RTX", "HON", "UNP", "BA", "LMT", "DE", "UPS", "ETN",
        "MMM", "EMR", "GD", "NOC",
    ],
    "Materials": [
        "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE", "DOW", "DD",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "WELL", "SPG", "O", "CCI", "PSA",
    ],
}


def _broad() -> list[str]:
    """Union of every sector basket (deduped) — a ~150-name broad-market browse."""
    seen: dict[str, None] = {}
    for names in SECTORS.values():
        for t in names:
            seen.setdefault(t, None)
    return list(seen)


# ── Public registry ──────────────────────────────────────────────────────────
# Top-level presets shown in the picker. Sector baskets are exposed separately
# (the UI offers a sector sub-selector) so this stays tidy.
PRESETS: dict[str, list[str]] = {
    "Dow 30": DOW_30,
    "Mega-cap 50": MEGACAP_50,
    "Broad market (~150)": _broad(),
}

SECTOR_NAMES = list(SECTORS.keys())


def get_universe(name: str) -> list[str]:
    """Resolve a preset name (top-level or a sector) to its ticker list."""
    if name in PRESETS:
        return list(PRESETS[name])
    if name in SECTORS:
        return list(SECTORS[name])
    return []


def all_preset_names() -> list[str]:
    """All selectable preset names: top-level presets + each sector."""
    return list(PRESETS.keys()) + [f"Sector — {s}" for s in SECTOR_NAMES]


def resolve(name: str) -> list[str]:
    """Resolve a picker label (incl. the 'Sector — X' form) to tickers."""
    if name.startswith("Sector — "):
        return get_universe(name.removeprefix("Sector — "))
    return get_universe(name)
