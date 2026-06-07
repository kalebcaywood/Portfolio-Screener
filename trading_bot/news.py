"""News / social-sentiment scanner — deterministic aggregation layer.

IMPORTANT separation of concerns:
  - The *gathering* of headlines (web search / fetch, Truth Social mentions) is
    done by the agent, because that needs live web tools. The agent writes its
    findings to news_input.json as structured records (one per headline) with a
    sentiment label and the ticker(s) it refers to.
  - This module is the *deterministic* half: it filters by recency, aggregates
    per-ticker sentiment, and turns it into trading-relevant sets according to a
    configurable mode. The LLM informs; it does not pull the trigger.

news_input.json schema (list of objects):
  {
    "symbol": "NVDA",                 # ticker the headline is about (uppercase)
    "sentiment": "positive",          # positive | negative | neutral
    "headline": "Nvidia raises guidance",
    "source": "Reuters",
    "url": "https://...",
    "date": "2026-05-29T13:00:00-04:00",  # ISO8601; optional
    "social": false                   # true if from Truth Social / X etc.
  }

Modes (config.news.mode):
  off        -> news ignored entirely
  alert      -> build watchlist + per-ticker scores, but DO NOT affect trading
  filter     -> additionally block BUYS on net-negative names (default, safe)
  autonomous -> additionally surface positive non-held names as buy candidates
                (still subject to ALL risk-guard limits + universe/tradability).
                NOT recommended.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HERE = os.path.dirname(__file__)
INPUT_PATH = os.path.join(HERE, "news_input.json")
OUTPUT_PATH = os.path.join(HERE, "news_signals.json")
UNIVERSE_META_PATH = os.path.join(HERE, "universe.json")

_SENT_VAL = {"positive": 1, "negative": -1, "neutral": 0}


def load_universe_meta() -> list[dict]:
    if not os.path.exists(UNIVERSE_META_PATH):
        return []
    with open(UNIVERSE_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("stocks", [])


def match_text_to_symbols(text: str, meta: list[dict] | None = None) -> list[str]:
    """Deterministically map a headline to in-universe tickers via name/aliases.

    Word-boundary-ish substring match on lowercased text. Returns sorted symbols.
    This is the safe mapper: it can ONLY ever return tickers in the universe, so
    a stray mention can't conjure an off-universe trade.
    """
    meta = meta if meta is not None else load_universe_meta()
    t = f" {text.lower()} "
    hits = set()
    for row in meta:
        sym = row["symbol"]
        # explicit ticker mention like " NVDA " or "$NVDA"
        if f" {sym.lower()} " in t or f"${sym.lower()}" in t:
            hits.add(sym)
            continue
        for alias in row.get("aliases", []):
            if alias and alias in t:
                hits.add(sym)
                break
    return sorted(hits)


def _parse_dt(s: str | None):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        return dt
    except ValueError:
        return None


def aggregate(records: list[dict], config: dict, universe: list[str],
              now: datetime | None = None) -> dict:
    now = now or datetime.now(ET)
    news_cfg = config.get("news", {})
    lookback = float(news_cfg.get("sentiment_lookback_hours", 48))
    cutoff = now - timedelta(hours=lookback)
    universe_set = {s.upper() for s in universe}

    per_ticker: dict[str, dict] = {}
    for rec in records:
        sym = str(rec.get("symbol", "")).upper().strip()
        if not sym:
            continue
        dt = _parse_dt(rec.get("date"))
        if dt is not None and dt < cutoff:
            continue  # too old
        sent = str(rec.get("sentiment", "neutral")).lower()
        val = _SENT_VAL.get(sent, 0)
        slot = per_ticker.setdefault(
            sym, {"score": 0, "pos": 0, "neg": 0, "neu": 0,
                  "in_universe": sym in universe_set, "headlines": []}
        )
        slot["score"] += val
        slot["pos"] += sent == "positive"
        slot["neg"] += sent == "negative"
        slot["neu"] += sent == "neutral"
        if len(slot["headlines"]) < 5:
            slot["headlines"].append({
                "sentiment": sent,
                "headline": rec.get("headline", ""),
                "source": rec.get("source", ""),
                "url": rec.get("url", ""),
                "social": bool(rec.get("social", False)),
            })

    blocked = sorted(s for s, d in per_ticker.items()
                     if d["in_universe"] and d["score"] < 0)
    positive = sorted(s for s, d in per_ticker.items()
                      if d["in_universe"] and d["score"] > 0)

    # non-universe positive mentions -> watchlist (alert only, never auto-traded
    # unless universe is expanded by a human)
    max_wl = int(news_cfg.get("max_watchlist", 25))
    watchlist = sorted(
        ({"symbol": s, "score": d["score"], "pos": d["pos"], "neg": d["neg"],
          "social_only": all(h["social"] for h in d["headlines"]) if d["headlines"] else False,
          "sample": d["headlines"][0]["headline"] if d["headlines"] else ""}
         for s, d in per_ticker.items() if not d["in_universe"] and d["score"] > 0),
        key=lambda x: x["score"], reverse=True,
    )[:max_wl]

    # autonomous candidates (only if mode==autonomous): positive, in-universe-or-
    # whitelisted, NOT blocked. Still goes through full risk guard downstream.
    mode = news_cfg.get("mode", "filter")
    autonomous_buys = positive if mode == "autonomous" else []

    return {
        "as_of": now.isoformat(),
        "mode": mode,
        "enabled": bool(news_cfg.get("enabled", False)),
        "lookback_hours": lookback,
        "per_ticker": per_ticker,
        "blocked_symbols": blocked,
        "positive_symbols": positive,
        "autonomous_buy_candidates": autonomous_buys,
        "watchlist": watchlist,
    }


def load_signals(config: dict) -> dict | None:
    """Load previously-computed news signals if news is enabled and present."""
    if not config.get("news", {}).get("enabled", False):
        return None
    if config.get("news", {}).get("mode", "filter") == "off":
        return None
    if not os.path.exists(OUTPUT_PATH):
        return None
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    import sys
    # quick mapper: `python -m trading_bot.news match "headline text ..."`
    if len(sys.argv) > 1 and sys.argv[1] == "match":
        text = " ".join(sys.argv[2:])
        print(json.dumps(match_text_to_symbols(text)))
        return

    p = argparse.ArgumentParser(description="news/sentiment aggregator")
    p.add_argument("--input", default=INPUT_PATH)
    p.add_argument("--out", default=OUTPUT_PATH)
    args = p.parse_args()

    with open(os.path.join(HERE, "config.json")) as f:
        config = json.load(f)
    records = []
    if os.path.exists(args.input):
        with open(args.input, "r", encoding="utf-8") as f:
            records = json.load(f)

    sig = aggregate(records, config, config["universe"])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2)

    print(f"=== news signals @ {sig['as_of']} (mode={sig['mode']}, {len(records)} headlines) ===")
    print(f"blocked (negative, in universe): {sig['blocked_symbols'] or '-'}")
    print(f"positive (in universe)        : {sig['positive_symbols'] or '-'}")
    if sig["watchlist"]:
        print("watchlist (mentioned, NOT in universe — alert only):")
        for w in sig["watchlist"]:
            tag = " [social-only]" if w["social_only"] else ""
            print(f"   {w['symbol']:6} score {w['score']:+d}{tag}  — {w['sample'][:70]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
