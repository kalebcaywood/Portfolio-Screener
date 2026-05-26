"""Shared visual theme — University of Tennessee palette + plotly template + badges."""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Path to the Power T logo. We search for any common image file in assets/
# and pick the first match. Falls back to a CSS-only "UT" mark if none found.
_ASSETS_DIR = Path(__file__).parent / "assets"
_LOGO_CANDIDATES = [
    "power_t.png", "power_t.jpg", "power_t.jpeg", "power_t.svg", "power_t.webp",
    "Power T.jpg", "Power T.png", "Power T.jpeg",
    "powerT.png", "powerT.jpg", "logo.png", "logo.jpg",
]
_MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml", ".webp": "image/webp",
}


def _logo_data_uri() -> str | None:
    """Return a base64 data URI for the Power T, or None if no logo file is found."""
    if not _ASSETS_DIR.exists():
        return None
    # Try named candidates first
    for name in _LOGO_CANDIDATES:
        path = _ASSETS_DIR / name
        if path.exists() and path.is_file():
            return _encode_image(path)
    # Otherwise, take the first image file in assets/
    for path in sorted(_ASSETS_DIR.iterdir()):
        if path.suffix.lower() in _MIME_BY_EXT:
            return _encode_image(path)
    return None


def _encode_image(path: Path) -> str | None:
    try:
        mime = _MIME_BY_EXT.get(path.suffix.lower(), "image/png")
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None

# ─── University of Tennessee brand colors ────────────────────────────────────
UT_ORANGE     = "#FF8200"   # Tennessee Orange (primary)
UT_SMOKEY     = "#58595B"   # Smokey gray
UT_WHITE      = "#FFFFFF"
UT_VALLEY     = "#006C93"   # Valley blue
UT_LECONTE    = "#8D2048"   # Leconte burgundy
UT_REGALIA    = "#4B306A"   # Regalia purple
UT_SUNSPHERE  = "#FED535"   # Sunsphere yellow (use sparingly)
UT_LEGACY     = "#B7A57A"   # Legacy tan
UT_ROCK       = "#E8E8E8"   # Rock gray
UT_RIVER      = "#517C96"   # River blue
UT_GLOBE      = "#0C2340"   # Globe deep navy

# Functional aliases used throughout the app
PRIMARY      = UT_ORANGE
SUCCESS      = "#15803d"
WARNING      = "#b45309"
DANGER       = "#b91c1c"
TEXT         = "#1a1a1a"
MUTED_TEXT   = UT_SMOKEY
BG           = UT_WHITE
SOFT_BG      = "#f7f7f5"
GRID         = "#e5e5e5"

# Plotly colorway — orange first, then complementary UT palette + neutrals
COLORWAY = [
    UT_ORANGE,
    UT_VALLEY,
    UT_LECONTE,
    UT_REGALIA,
    UT_LEGACY,
    UT_SMOKEY,
    UT_RIVER,
    "#0f766e",   # neutral dark teal
    "#a16207",   # neutral ochre
    "#7c3aed",   # neutral violet
    "#0e7490",   # neutral cyan-dark
    "#525252",   # neutral mid-gray
]

# ─── Plotly template ──────────────────────────────────────────────────────────
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
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=GRID, borderwidth=1,
            orientation="h", y=-0.2, x=0,
        ),
        hoverlabel=dict(
            font=dict(family="Inter", size=12),
            bgcolor="white", bordercolor=UT_ORANGE,
        ),
        margin=dict(t=50, b=50, l=60, r=20),
        colorscale=dict(
            # Sequential: white → Tennessee Orange
            sequential=[[0, "#fff4e6"], [0.5, "#ffb04d"], [1, UT_ORANGE]],
            # Diverging: burgundy (negative) → white → orange (positive)
            diverging=[[0, UT_LECONTE], [0.5, "#ffffff"], [1, UT_ORANGE]],
        ),
    )
)
pio.templates.default = "plotly_white+quantlab"


# ─── CSS — narrow scope so Streamlit's icon font keeps working ───────────────
_CSS = """
<style>
@import url('https://rsms.me/inter/inter.css');

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Tabular numbers only on numeric displays */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    font-feature-settings: 'tnum' 1, 'cv11' 1;
}

/* Headings — weight / color only, never touch font-family */
.stApp h1 {
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #1a1a1a;
    margin-bottom: 0.5rem;
}
.stApp h2 {
    font-weight: 600;
    letter-spacing: -0.015em;
    color: #1a1a1a;
    margin-top: 1.5rem;
}
.stApp h3 {
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #2a2a2a;
}
.stApp h5 {
    font-weight: 600;
    color: #2a2a2a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 11px;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

/* Captions */
[data-testid="stCaptionContainer"] {
    color: #58595B;
}

/* Metric cards — white background with a thin orange top border for brand */
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 14px 16px;
    border-radius: 6px;
    border: 1px solid #e5e5e5;
    border-top: 3px solid #FF8200;
    transition: box-shadow 0.2s, border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 1px 3px rgba(255, 130, 0, 0.15);
    border-color: #FF8200;
}
[data-testid="stMetricLabel"] {
    font-size: 11px;
    color: #58595B;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-size: 22px;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1.2;
}
[data-testid="stMetricDelta"] {
    font-size: 12px;
    font-weight: 500;
}

/* Sidebar — light gray with orange left border */
[data-testid="stSidebar"] {
    background-color: #f7f7f5;
    border-right: 3px solid #FF8200;
}

/* DataFrame */
.stDataFrame {
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    overflow: hidden;
}

/* Section divider */
hr {
    margin: 1.5rem 0;
    border-color: #e5e5e5;
    border-top-width: 1px;
}

/* Tabs — orange active state */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid #e5e5e5;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px;
    font-weight: 500;
    color: #58595B;
    padding: 8px 14px;
}
.stTabs [aria-selected="true"] {
    color: #FF8200;
    font-weight: 700;
    border-bottom-color: #FF8200 !important;
}

/* Buttons — primary in Tennessee Orange */
.stButton button[kind="primary"] {
    background-color: #FF8200;
    border-color: #FF8200;
    color: #ffffff;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.stButton button[kind="primary"]:hover {
    background-color: #E07300;
    border-color: #E07300;
    color: #ffffff;
}
.stButton button[kind="primary"]:active {
    background-color: #BF6300;
    border-color: #BF6300;
}
.stButton button {
    border-radius: 4px;
    font-weight: 500;
}

/* Inputs — focus outline in orange */
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus,
.stSelectbox [data-baseweb="select"]:focus-within {
    border-color: #FF8200 !important;
    box-shadow: 0 0 0 1px #FF8200 !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    border-radius: 4px;
    border-color: #cbd5e1;
}

/* Links */
a, .stApp a {
    color: #FF8200;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}

/* Alerts */
.stAlert {
    border-radius: 6px;
    border-left: 4px solid #FF8200;
}

/* Slider — orange track */
.stSlider [role="slider"] {
    background-color: #FF8200 !important;
}

/* Checkbox / radio — orange when selected */
.stCheckbox [aria-checked="true"], .stRadio [aria-checked="true"] {
    background-color: #FF8200 !important;
    border-color: #FF8200 !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #FF8200 !important;
}

/* UT brand header strip — used at the top of the Home page */
.ut-header {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 18px 0 16px 0;
    border-bottom: 3px solid #FF8200;
    margin-bottom: 24px;
}
.ut-logo-wrap {
    display: flex;
    align-items: center;
    flex-shrink: 0;
}
.ut-mark {
    background: #FF8200;
    color: #ffffff;
    font-weight: 800;
    font-size: 22px;
    letter-spacing: 0.04em;
    padding: 10px 16px;
    border-radius: 4px;
    line-height: 1;
    font-family: 'Inter', -apple-system, sans-serif;
    box-shadow: 0 2px 4px rgba(255, 130, 0, 0.2);
}
.ut-wordmark {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.ut-supratitle {
    color: #FF8200;
    font-weight: 700;
    font-size: 10px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}
.ut-title {
    color: #1a1a1a;
    font-weight: 700;
    font-size: 22px;
    letter-spacing: -0.015em;
    line-height: 1.15;
}

/* Sidebar brand bar */
.ut-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 12px;
    margin-bottom: 8px;
    border-bottom: 2px solid #FF8200;
}
.ut-sidebar-mark {
    background: #FF8200;
    color: #ffffff;
    font-weight: 800;
    font-size: 14px;
    letter-spacing: 0.04em;
    padding: 4px 9px;
    border-radius: 3px;
    line-height: 1;
}
.ut-sidebar-text {
    color: #58595B;
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

/* Status bar */
.qstatus {
    background: #f7f7f5;
    border: 1px solid #e5e5e5;
    border-left: 4px solid #FF8200;
    border-radius: 4px;
    padding: 8px 14px;
    margin-bottom: 1rem;
    font-size: 12px;
    color: #58595B;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
}
.qstatus .qstatus-key {
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 10px;
    margin-right: 6px;
}
.qstatus .qstatus-val {
    color: #1a1a1a;
    font-weight: 600;
}

/* Badges */
.qbadge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    line-height: 1;
}
.qbadge-reup, .qbadge-low, .qbadge-success { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.qbadge-add, .qbadge-info                  { background: #fff4e6; color: #c25e00; border: 1px solid #ffd7a8; }
.qbadge-hold, .qbadge-neutral              { background: #f1f5f9; color: #58595B; border: 1px solid #e5e5e5; }
.qbadge-trim, .qbadge-medium, .qbadge-warning { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.qbadge-exit, .qbadge-high, .qbadge-danger { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
</style>
"""


def inject_css() -> None:
    """Inject the global CSS. Call once per page after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def setup_page(title: str, layout: str = "wide") -> None:
    """set_page_config + inject CSS in one call."""
    st.set_page_config(page_title=title, layout=layout)
    inject_css()


def ut_header(title: str = "Quantitative Portfolio Analytics",
               supratitle: str = "University of Tennessee") -> None:
    """Render the UT-branded header — Power T logo on the left, wordmark on the right."""
    data_uri = _logo_data_uri()
    if data_uri:
        mark_html = (
            f'<img src="{data_uri}" alt="UT" '
            'style="height:72px; width:auto; display:block;">'
        )
    else:
        mark_html = '<div class="ut-mark">UT</div>'

    st.markdown(
        f"""
<div class="ut-header">
    <div class="ut-logo-wrap">{mark_html}</div>
    <div class="ut-wordmark">
        <div class="ut-supratitle">{supratitle}</div>
        <div class="ut-title">{title}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def ut_sidebar_brand(label: str = "Portfolio Analytics") -> None:
    """Render a small UT brand bar at the top of the sidebar."""
    data_uri = _logo_data_uri()
    if data_uri:
        mark_html = (
            f'<img src="{data_uri}" alt="UT" '
            'style="height:36px; width:auto; display:block;">'
        )
    else:
        mark_html = '<div class="ut-sidebar-mark">UT</div>'

    st.sidebar.markdown(
        f"""
<div class="ut-sidebar-brand">
    {mark_html}
    <div class="ut-sidebar-text">{label}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def badge(label: str, kind: str | None = None) -> str:
    """Return HTML for a colored pill badge."""
    k = (kind or label).lower().replace(" ", "-").replace("/", "-")
    return f'<span class="qbadge qbadge-{k}">{label}</span>'


def status_line(positions: int, aum: float, last_date, benchmark: str = "SPX",
                 period: str = "", net: float | None = None,
                 gross: float | None = None) -> None:
    """One-line status bar for analytics pages."""
    if hasattr(last_date, "strftime"):
        last_str = last_date.strftime("%Y-%m-%d")
    else:
        last_str = str(last_date)
    parts = [("Positions", str(positions)), ("AUM", f"${aum:,.0f}")]
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
    st.markdown(f'<div class="qstatus">{"".join(spans)}</div>', unsafe_allow_html=True)


# ─── DataFrame styling helpers ────────────────────────────────────────────────

REC_PALETTE: dict[str, tuple[str, str]] = {
    "REUP": ("#dcfce7", "#15803d"),
    "ADD":  ("#fff4e6", "#c25e00"),
    "HOLD": ("#f1f5f9", "#58595B"),
    "TRIM": ("#fef3c7", "#b45309"),
    "EXIT": ("#fee2e2", "#b91c1c"),
}

SEVERITY_PALETTE: dict[str, tuple[str, str]] = {
    "high":   ("#fee2e2", "#b91c1c"),
    "medium": ("#fef3c7", "#b45309"),
    "info":   ("#fff4e6", "#c25e00"),
    "low":    ("#dcfce7", "#15803d"),
}


def style_recommendation_column(df: pd.DataFrame, col: str = "recommendation"):
    def _style(val):
        if not isinstance(val, str):
            return ""
        bg, fg = REC_PALETTE.get(val.upper(), ("", ""))
        if not bg:
            return ""
        return f"background-color: {bg}; color: {fg}; font-weight: 600;"
    return df.style.map(_style, subset=[col] if col in df.columns else [])


def style_severity_column(df: pd.DataFrame, col: str = "severity"):
    def _style(val):
        if not isinstance(val, str):
            return ""
        bg, fg = SEVERITY_PALETTE.get(val.lower(), ("", ""))
        if not bg:
            return ""
        return f"background-color: {bg}; color: {fg}; font-weight: 600;"
    return df.style.map(_style, subset=[col] if col in df.columns else [])


pio.templates.default = "plotly_white+quantlab"
