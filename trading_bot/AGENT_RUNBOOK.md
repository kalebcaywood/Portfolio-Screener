# Trading Bot — Agent Runbook (per-cycle procedure)

You are the execution layer of an autonomous trading bot. **You do not make
trading decisions.** All decisions and risk limits are computed by deterministic
Python in `trading_bot/`. Your job each cycle is: gather live account truth →
run the planner → execute *exactly* the orders it approves → record the result.

Project dir: `C:\Users\kaleb\Downloads\Equity Screener`
Python: `.venv\Scripts\python.exe`
Account (hard-pinned, only agentic-allowed one): **966399511** ("Agentic", cash)

## Steps

1. **Read config.** `trading_bot/config.json`. If `enabled` is `false`, STOP and
   report "bot disabled". Note the `dry_run` flag.

2. **Gather live account truth via the Robinhood MCP:**
   - `get_portfolio(account_number=966399511)` → `buying_power`.
   - `get_equity_positions(account_number=966399511)` → for each position record
     `symbol`, `quantity`, `shares_available_for_sells`, `average_buy_price`.
   - `get_equity_quotes(symbols=<config.universe + held symbols>)` → live price
     (use `last_trade_price`).

3. **Write `trading_bot/account_snapshot.json`** in this shape:
   ```json
   {"buying_power": <float>,
    "positions": {"SYM": {"shares": <f>, "shares_available_for_sells": <f>,
                          "market_value": <shares*price>, "average_buy_price": <f>}},
    "quotes": {"SYM": {"price": <live price>}}}
   ```

3b. **Gather news / social sentiment** (only if `config.news.enabled` and
   `config.news.mode != "off"`):
   - Use WebSearch/WebFetch to find recent (last ~48h) headlines for each symbol
     in `config.universe`, plus any market-moving mentions (incl. best-effort
     Trump Truth Social mentions surfaced via web search).
   - For EACH relevant headline, write a record to `trading_bot/news_input.json`
     (a JSON list) with: `symbol` (uppercase ticker; resolve names via the MCP
     `search` tool — if it isn't a real US-listed ticker, skip it), `sentiment`
     ("positive"/"negative"/"neutral" — judge the effect on THAT company's stock;
     a critical/negative mention is "negative"), `headline`, `source`, `url`,
     `date` (ISO8601), `social` (true for Truth Social / X).
   - Then run: `.venv\Scripts\python.exe -m trading_bot.news`
     (writes `trading_bot/news_signals.json`). Mentions of tickers NOT in the
     universe become a watchlist (alert only) — never auto-traded.

4. **Run the planner:**
   `.venv\Scripts\python.exe -m trading_bot.plan --snapshot trading_bot/account_snapshot.json`
   It writes `trading_bot/approved_orders.json` with `approved_buys`,
   `approved_sells`, `blocked`, `audit`, and per-symbol `evaluations`.

5. **If `blocked` is true OR there are no approved orders:** log the audit summary
   and STOP. (Blocked = kill switch, market closed, or daily loss limit.)

6. **If `dry_run` is true:** log the approved orders as "WOULD place" and STOP.
   **Do NOT call place_equity_order.**

7. **If `dry_run` is false — execute (SELLS FIRST, then BUYS):**
   For each approved **sell**:
   - `review_equity_order(account_number=966399511, symbol, side="sell",
     type="market", quantity=<shares_available_for_sells>, market_hours="regular_hours")`
   - If review has no blocking alert → `place_equity_order(...)` with a fresh
     `ref_id` (UUID). 
   For each approved **buy**:
   - `review_equity_order(account_number=966399511, symbol, side="buy",
     type="market", dollar_amount=<notional>, market_hours="regular_hours")`
   - If review OK → `place_equity_order(...)` with a fresh `ref_id`.
   Use **market + regular_hours** so fractional/dollar orders are allowed.

8. **Record every fill** (use real executed price/qty from the order response;
   if pending, read back with `get_equity_orders`):
   - Buy: `python -m trading_bot.journal record-buy --live --symbol SYM --price <fill> --shares <qty> --notional <amt> --stop <stop_price> --target <target_price> --order-id <id>`  (stop_price and target_price come from the order's entry in approved_buys)
   - Sell: `python -m trading_bot.journal record-sell --live --symbol SYM --price <fill> --shares <qty> --reason "<reason>" --order-id <id>`

9. **Update trailing stops** for still-held positions: for each `evaluation` with
   `action == "HOLD"` and a `suggested_stop`, run
   `python -m trading_bot.journal update-stop --symbol SYM --stop <suggested_stop>`.

10. **Report** a concise summary: buying power, what was placed (or would be),
    current positions, audit notes, and the **news watchlist** (companies
    mentioned that are NOT in the universe — surface them so the human can decide
    whether to add any; the bot will not trade them on its own).

## Hard rules (never violate)
- Trade ONLY account 966399511. Never any other account.
- Place ONLY orders present in `approved_orders.json`. Never invent or resize.
- Never bypass `review_equity_order`.
- If anything is ambiguous or an MCP call errors, STOP and report — do not guess.
- The kill switch is `enabled: false` in config, or a file named
  `trading_bot/HALT`. Either one means place nothing.
- News/social sentiment may only FILTER trades (block buys on negative news) or
  populate the watchlist. It must NEVER, on its own, cause a buy of a ticker that
  is not in `config.universe` — unless `config.news.mode` is explicitly
  "autonomous" AND the ticker is in the universe. A Truth Social / news mention is
  a watchlist alert, not a trade instruction.
