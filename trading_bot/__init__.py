"""Autonomous technical-rules trading bot for the Robinhood agentic (cash) account.

Decisions and risk limits live in deterministic Python (strategy.py, risk_guard.py).
The scheduled Claude agent only executes orders this package approves.
"""
