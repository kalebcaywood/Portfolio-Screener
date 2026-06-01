# Autonomous Trading Bot

A small, transparent, **technical-rules** trading bot for the Robinhood *agentic*
cash account (`••••9511`). It runs as a scheduled Claude agent during market
hours: deterministic Python computes the signals and enforces the risk limits;
the agent only executes the orders the code approves.

> ⚠️ **Real money.** Default capital is a <$500 test sleeve. Start in dry-run,
> watch it for a few sessions, then flip to live. Markets carry risk of loss.

## Design principle

**Decisions and risk limits live in code, not in the LLM.** The agent cannot
improvise a trade, oversize a position, or skip a safety check — it can only
place orders that `risk_guard.py` has already approved and written to
`approved_orders.json`.

```
market data (yfinance)  ─┐
                         ├─► strategy.py ─► candidates ─► risk_guard.py ─► approved_orders.json ─► agent places via MCP ─► journal.py
account truth (MCP)    ──┘        (signals)                 (hard limits)                              (executes)         (state + log)
```

## Strategy (daily bars)

- **Buy** when: fast SMA(10) crosses **above** slow SMA(30) *and* price is above
  the SMA(50) trend filter *and* RSI(14) < 70 (not overbought).
- **Sell** when (first to trigger): price hits the **ATR trailing stop**, or
  +15% **take-profit**, or fast crosses **below** slow (death cross), or RSI ≥ 78.

Tune everything in `config.json → strategy`.

## Risk limits (`config.json → risk`)

| Limit | Default | Meaning |
|---|---|---|
| `max_position_notional` | $50 | most $ per single position |
| `min_order_notional` | $10 | skip trivially small orders |
| `max_total_deployed` | $200 | most $ invested at once |
| `max_positions` | 3 | most concurrent holdings |
| `max_new_buys_per_day` | 2 | throttle entries |
| `daily_loss_limit` | $15 | realized loss → halt new buys |
| `cash_buffer` | $10 | always keep this much cash |
| `min_hold_days` | 1 | reduce churn / cash-settlement issues |
| `regular_hours_only` | true | trade only 9:30–16:00 ET |

The account is **cash** (not margin), so settlement matters: the bot is
low-turnover, only spends real buying power, and won't rebuy a name it sold the
same day (good-faith-violation guard).

## Files

| File | Role |
|---|---|
| `config.json` | all knobs + `enabled` / `dry_run` switches |
| `strategy.py` | pure indicator + signal logic |
| `risk_guard.py` | hard limits; the safety layer |
| `plan.py` | fetch data → signals → risk → `approved_orders.json` |
| `journal.py` | `state.json` (stops/entries/daily counters) + `trades.jsonl` |
| `backtest.py` | historical sanity-check |
| `AGENT_RUNBOOK.md` | exact per-cycle procedure the scheduled agent follows |

## Operating it

```bash
# Backtest the rules
.venv\Scripts\python.exe -m trading_bot.backtest --period 3y

# Dry-run plan against an account snapshot (no orders placed)
.venv\Scripts\python.exe -m trading_bot.plan --snapshot trading_bot/account_snapshot.json

# Inspect bot memory / trade log
.venv\Scripts\python.exe -m trading_bot.journal show
type trading_bot\trades.jsonl
```

## Universe (`config.json → universe` + `universe.json`)

~75 highly liquid large-caps + sector ETFs. `universe.json` holds each name's
company name, sector, and **aliases** (brands/people) so news mentions map to
tickers deterministically (`python -m trading_bot.news match "headline"`).
The bot *scans* all of them; with the current cap it *holds* ~2 at a time.

To add a name: put the symbol in `config.json → universe` **and** add a matching
metadata row in `universe.json` (a sync check runs in tests).

## Schedules (two scheduled tasks)

| Task | When (ET) | Does |
|---|---|---|
| `trading-bot-news` | ~8:20 AM & ~12:20 PM, Mon–Fri | **News only** — refresh sentiment filter + watchlist. Never trades. "Stay ahead of the news." |
| `trading-bot-intraday` | hourly ~10:35–14:35, Mon–Fri | **Intraday monitoring + trading** — catch stops/targets fast, act on signals. Reuses morning news (no re-scan). |
| `trading-bot` | ~3:48 PM, Mon–Fri | Full close cycle — account → fresh news scan → plan → trade → journal |

Both run only while the app is open (a missed run fires on next launch). Click
**Run now** once on each (in the *Scheduled* sidebar) to pre-approve tool perms.

## Scaling capital (the one knob)

Funded with $113 to start. As you add cash, raise these in `config.json → risk`
(everything else already scales off live buying power):

| Knob | $113 start | ~$1k | ~$5k |
|---|---|---|---|
| `max_position_notional` | 50 | 150 | 500 |
| `max_total_deployed` | 200 | 900 | 4500 |
| `max_positions` | 6 | 8 | 12 |
| `max_new_buys_per_day` | 6 | 8 | 12 |
| `max_trades_per_day` | 12 | 16 | 24 |
| `daily_loss_limit` | 15 | 60 | 250 |

Bigger universe = better *selection*; these caps control *capacity*. Note: at
$113 the binding limit is real cash — `max_total_deployed $200 ÷ $50` ≈ 4
positions until you fund more.

## Cadence & cash-account safety

Runs intraday (hourly) plus the close, and **may make multiple trades a day**.
This is safe in a cash account because of one invariant the risk guard always
enforces: **buys never exceed the broker's reported buying power (settled cash)**,
so good-faith violations cannot occur. T+1 settlement means same-day *sale
proceeds* aren't redeployable until they settle — a natural throttle, not a risk.
PDT rules don't apply to cash accounts. `max_trades_per_day` is a runaway guard.

### Switches
- **Dry-run vs live:** `config.json → "dry_run": true|false`. Starts `true`.
- **Kill switch (two ways):**
  1. set `config.json → "enabled": false`, or
  2. create an empty file `trading_bot/HALT`.
  Either one makes the bot place **nothing** on the next cycle.

## News / social sentiment (`config.json → news`)

A scanner layer adds news- and Truth-Social-awareness. Gathering headlines is
done by the agent (live web search); `news.py` then deterministically aggregates
them into per-ticker sentiment. **The LLM informs trades; it never pulls the
trigger.**

Modes (`news.mode`):

| Mode | Effect |
|---|---|
| `off` | news ignored |
| `alert` | builds a watchlist + sentiment log; **no** effect on trading |
| `filter` *(default)* | additionally **blocks buys** on net-negative-news names |
| `autonomous` | additionally lets positive news trigger buys — **only** for in-universe, tradable names, still under every risk cap. Not recommended. |

Key safety properties:
- A company **mentioned but not in `config.universe`** (e.g. a stock Trump posts
  about) goes to the **watchlist (alert only)** — never auto-traded. You add it
  to the universe by hand if you want it tradable.
- Negative coverage can stop a buy the technicals wanted; it can never *cause* one
  (except explicit `autonomous` mode).
- `require_positive_for_buy: true` makes positive news a *precondition* for any buy.

Why not auto-buy whatever gets mentioned? Headline/social trades are manipulation-
prone and late (by the time a post is scraped, the move's done — e.g. the public
record shows Trump *bought* Palantir before praising it). This design captures the
signal without making you the exit liquidity.

## Caveats
- yfinance data is unofficial; the planner cross-checks against live MCP quotes.
- A MA-crossover system underperforms buy-and-hold in strong bull markets (it
  sits in cash between signals) in exchange for smaller drawdowns. Not a
  money-printer — it's a disciplined, capped, auditable rules engine.
- **Not financial advice.**
