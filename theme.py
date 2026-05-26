"""Shared visual theme: plotly template, typography CSS, badge helper, status line."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ─── Color palette ────────────────────────────────────────────────────────────
PRIMARY      = "#1e40af"
ACCENT_BLUE  = "#3b82f6"
ACCENT_TEAL  = "#0891b2"
ACCENT_PURPLE = "#7c3aed"
SUCCESS      = "#15803d"
WARNING      = "#b45309"
DANGER       = "#b91c1c"
NEUTRAL      = "#475569"
BG           = "#ffffff"
SOFT_BG      = "#f8fafc"
TEXT         = "#0f172a"
MUTED_TEXT   = "#64748b"
GRID         = "#e2e8f0"

COLORWAY = [
    "#1e40af", "#0891b2", "#15803d", "#b45309", "#7c3aed",
    "#be185d", "#0f766e", "#a16207", "#1d4ed8", "#9333ea",
    "#525252", "#0e7490",
]

# ─── Plotly template — registered globally ──────────────────────────────────
pio.templates["quantlab"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                   size=12, color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        title=dict(
            font=dict(family="Inter", size=14, color=TEXT),
            x=0.0, xanchor="left", pad=dict(l=4, t=8),
        ),
        xaxis=dict(
            gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID,
            tickfont=dict(size=11, color=MUTED_TEXT),
            title=dict(font=dict(size=12, color=TEXT)),
            showline=True, mirror=False,
        ),
        yaxis=dict(
            gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID,
            tickfont=dict(size=11, color=MUTED_TEXT),
            title=dict(font=dict(size=12, color=TEXT)),
            showline=True, mirror=False,
        ),
        legend=dict(
            font=dict(size=11, color=TEXT),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=GRID, borderwidth=1,
            orientation="h", y=-0.2, x=0,
        ),
        hoverlabel=dict(
            font=dict(family="Inter", size=12),
            bgcolor="white", bordercolor=GRID,
        ),
        margin=dict(t=50, b=50, l=60, r=20),
        colorscale=dict(
            sequential=[[0, "#dbeafe"], [1, "#1e3a8a"]],
            diverging=[[0, "#b91c1c"], [0.5, "#f8fafc"], [1, "#1e40af"]],
        ),
    )
)
pio.templates.default = "plotly_white+quantlab"


# ─── CSS — kept narrow so it doesn't override Streamlit's icon font ──────────
# Key principle: only set font-family on `body` and let it inherit. Streamlit
# uses Material Symbols (a ligature icon font) on chevrons, expander toggles,
# sidebar collapse buttons, etc. Those elements set their own font-family
# inline or via more-specific class selectors — leave them alone.
_CSS = """
<style>
@import url('https://rsms.me/inter/inter.css');

/* Inter on the body only — cascade handles text, while icon-specific
   font-family declarations from Streamlit's CSS still win on icon elements */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Tabular numbers on numeric data only — no font-family change */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    font-feature-settings: 'tnum' 1, 'cv11' 1;
}

/* Headings — weight / color only, NEVER touch font-family */
.stApp h1 {
    font-weight: 600;
    letter-spacing: -0.025em;
    color: #0f172a;
    margin-bottom: 0.5rem;
}
.stApp h2 {
    font-weight: 600;
    letter-spacing: -0.015em;
    color: #0f172a;
    margin-top: 1.5rem;
}
.stApp h3 {
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #1e293b;
}
.stApp h5 {
    font-weight: 600;
    color: #1e293b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 11px;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

/* Caption softening */
[data-testid="stCaptionContainer"] {
    color: #64748b;
}

/* Metric card visual */
[data-testid="stMetric"] {
    background: #f8fafc;
    padding: 14px 16px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: #cbd5e1;
}
[data-testid="stMetricLabel"] {
    font-size: 11px;
    color: #64748b;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-size: 22px;
    font-weight: 600;
    color: #0f172a;
    line-height: 1.2;
}
[data-testid="stMetricDelta"] {
    font-size: 12px;
    font-weight: 500;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

/* DataFrame border */
.stDataFrame {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    overflow: hidden;
}

/* Section divider */
hr {
    margin: 1.5rem 0;
    border-color: #e2e8f0;
    border-top-width: 1px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px;
    font-weight: 500;
    color: #64748b;
    padding: 8px 14px;
}
.stTabs [aria-selected="true"] {
    color: #1e40af;
    font-weight: 600;
}

/* Buttons */
.stButton button[kind="primary"] {
    background-color: #1e40af;
    border-color: #1e40af;
    font-weight: 500;
}
.stButton button[kind="primary"]:hover {
    background-color: #1e3a8a;
    border-color: #1e3a8a;
}
.stButton button {
    border-radius: 6px;
    font-weight: 500;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    border-radius: 6px;
    border-color: #cbd5e1;
}

/* Alerts */
.stAlert {
    border-radius: 6px;
}

/* Status bar */
.qstatus {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 6px 14px;
    margin-bottom: 1rem;
    font-size: 12px;
    color: #475569;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
}
.qstatus .qstatus-key {
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 10px;
    margin-right: 6px;
}
.qstatus .qstatus-val {
    color: #0f172a;
    font-weight: 500;
}

/* Badges */
.qbadge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    line-height: 1;
}
.qbadge-reup, .qbadge-low, .qbadge-success { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.qbadge-add, .qbadge-info                  { background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
.qbadge-hold, .qbadge-neutral              { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
.qbadge-trim, .qbadge-medium, .qbadge-warning { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.qbadge-exit, .qbadge-high, .qbadge-danger { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
</style>
"""


def inject_css() -> None:
    """Inject the global CSS. Call this once per page after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def setup_page(title: str, layout: str = "wide") -> None:
    """Convenience: set_page_config + inject CSS in one call."""
    st.set_page_config(page_title=title, layout=layout)
    inject_css()


def badge(label: str, kind: str | None = None) -> str:
    """Return HTML for a colored pill badge."""
    k = (kind or label).lower().replace(" ", "-").replace("/", "-")
    return f'<span class="qbadge qbadge-{k}">{label}</span>'


def status_line(positions: int, aum: float, last_date, benchmark: str = "SPX",
                 period: str = "", net: float | None = None,
                 gross: float | None = None) -> None:
    """Render a one-line status bar at the top of analytics pages."""
    if hasattr(last_date, "strftime"):
        last_str = last_date.strftime("%Y-%m-%d")
    else:
        last_str = str(last_date)

    parts = [
        ("Positions", str(positions)),
        ("AUM", f"${aum:,.0f}"),
    ]
    if net is not None:
        parts.append(("Net", f"{net:.1%}"))
    if gross is not None:
        parts.append(("Gross", f"{gross:.1%}"))
    if period:
        parts.append(("Lookback", period))
    parts.append(("Through", last_str))
    parts.append(("Benchmark", benchmark))

    spans = [
        f'<span><span class="qstatus-key">{k}</span><span class="qstatus-val">{v}</span></span>'
        for k, v in parts
    ]
    st.markdown(f'<div class="qstatus">{"".join(spans)}</div>',
                unsafe_allow_html=True)


# ─── DataFrame styling helpers ────────────────────────────────────────────────

REC_PALETTE: dict[str, tuple[str, str]] = {
    "REUP": ("#dcfce7", "#15803d"),
    "ADD":  ("#dbeafe", "#1e40af"),
    "HOLD": ("#f1f5f9", "#475569"),
    "TRIM": ("#fef3c7", "#b45309"),
    "EXIT": ("#fee2e2", "#b91c1c"),
}

SEVERITY_PALETTE: dict[str, tuple[str, str]] = {
    "high":   ("#fee2e2", "#b91c1c"),
    "medium": ("#fef3c7", "#b45309"),
    "info":   ("#dbeafe", "#1e40af"),
    "low":    ("#dcfce7", "#15803d"),
}


def style_recommendation_column(df: pd.DataFrame, col: str = "recommendation"):
    """Apply background + text color to a recommendation column."""
    def _style(val):
        if not isinstance(val, str):
            return ""
        bg, fg = REC_PALETTE.get(val.upper(), ("", ""))
        if not bg:
            return ""
        return f"background-color: {bg}; color: {fg}; font-weight: 600;"
    return df.style.map(_style, subset=[col] if col in df.columns else [])


def style_severity_column(df: pd.DataFrame, col: str = "severity"):
    """Apply background + text color to a severity column."""
    def _style(val):
        if not isinstance(val, str):
            return ""
        bg, fg = SEVERITY_PALETTE.get(val.lower(), ("", ""))
        if not bg:
            return ""
        return f"background-color: {bg}; color: {fg}; font-weight: 600;"
    return df.style.map(_style, subset=[col] if col in df.columns else [])


pio.templates.default = "plotly_white+quantlab"
