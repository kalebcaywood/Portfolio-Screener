"""Shared visual theme — University of Tennessee palette + plotly template + badges."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


# SHA-256 of the access password. Plaintext is intentionally not in source.
# To change the password, compute the new hash with:
#   python -c "import hashlib; print(hashlib.sha256(b'<new_pw>').hexdigest())"
_PASSWORD_HASH = "d54123de468bd42ea00dafbd777f85fe5fa1ff6404d9838c007953c25c92a1c5"


def require_password() -> None:
    """Show a password gate; halt the script if the user isn't authenticated.

    Call this once at the very top of the entry script, after st.set_page_config.
    Once authenticated, st.session_state['authenticated'] = True persists for
    the duration of the browser session — every page in the multipage app
    reads through this same flag.
    """
    if st.session_state.get("authenticated", False):
        return

    bg = _login_bg_data_uri()
    logo = _logo_data_uri()

    # Full-screen backdrop: the user's image (under a dark scrim so the white
    # card and text stay readable over any photo), or a soft solid fallback
    # when no background image has been dropped into assets/ yet.
    if bg:
        backdrop = (
            f'[data-testid="stAppViewContainer"], .stApp {{'
            f'  background-image: linear-gradient(rgba(8,12,20,0.45), rgba(8,12,20,0.62)), url("{bg}");'
            f'  background-size: cover;'
            f'  background-position: center;'
            f'  background-repeat: no-repeat;'
            f'}}'
        )
    else:
        backdrop = 'body, .stApp { background: #f7f7f5; }'

    # Brand mark inside the card: real Power T if available, else CSS "UT" box.
    mark_html = (
        f'<img src="{logo}" alt="UT" style="height:66px;width:auto;display:block;margin-bottom:10px;">'
        if logo else '<div class="gate-mark">UT</div>'
    )

    st.markdown(
        f"""
<style>
{backdrop}
/* Keep Streamlit's header out of the way for a clean full-screen gate */
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}

/* Dark glassy login card, centered over the backdrop (matches the terminal
   theme and keeps the dark password input cohesive). */
.st-key-gate_card {{
    max-width: 420px;
    margin: 10vh auto 0 auto;
    padding: 34px 34px 26px 34px;
    background: rgba(13, 18, 30, 0.82);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #FF8200;
    border-radius: 14px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.6);
}}
.gate-mark {{
    background: #FF8200;
    color: white;
    font-weight: 800;
    font-size: 22px;
    padding: 10px 16px;
    border-radius: 4px;
    display: inline-block;
    letter-spacing: 0.05em;
    line-height: 1;
    margin-bottom: 10px;
}}
.gate-supra {{
    color: #FF9A33;
    font-weight: 700;
    font-size: 10px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}}
.gate-title {{
    font-weight: 700;
    font-size: 23px;
    color: #F2F5FA;
    letter-spacing: -0.015em;
    margin: 4px 0 4px 0;
}}
.gate-caption {{
    color: #9AA6BC;
    font-size: 13px;
    margin: 4px 0 16px 0;
}}
</style>
        """,
        unsafe_allow_html=True,
    )

    # Everything (brand header + form) lives inside one keyed container so the
    # CSS above can render it as a single floating card.
    with st.container(key="gate_card"):
        st.markdown(
            f"""
<div>{mark_html}</div>
<div class="gate-supra">University of Tennessee</div>
<div class="gate-title">Quantitative Portfolio Analytics</div>
<div class="gate-caption">Access required. Enter password to continue.</div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("auth_form", clear_on_submit=False, border=False):
            pw = st.text_input("Password", type="password", label_visibility="collapsed",
                                 placeholder="Password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            attempt = hashlib.sha256(pw.encode("utf-8")).hexdigest()
            if attempt == _PASSWORD_HASH:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")

    st.stop()


def render_logout_button() -> None:
    """Optional sidebar logout button — clears auth so user is re-prompted."""
    if st.session_state.get("authenticated", False):
        with st.sidebar:
            if st.button("Sign out", type="secondary", width="stretch"):
                st.session_state["authenticated"] = False
                st.rerun()

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


# Background image for the password screen. Drop a file named login_bg.<ext>
# (or background.<ext>) into assets/ and it becomes the full-screen backdrop.
_BG_CANDIDATES = [
    "login_bg.jpg", "login_bg.jpeg", "login_bg.png", "login_bg.webp",
    "background.jpg", "background.jpeg", "background.png", "background.webp",
    "login.jpg", "login.jpeg", "login.png", "login.webp",
]


def _login_bg_data_uri(max_w: int = 1920, quality: int = 82) -> str | None:
    """Return a base64 data URI for the password-screen background, if present.

    The image is embedded directly in the page, so we downscale/recompress it
    (max 1920px wide, JPEG q82) to keep the gate fast even if a large photo is
    dropped in. Falls back to the raw file if Pillow isn't available.
    """
    if not _ASSETS_DIR.exists():
        return None
    for name in _BG_CANDIDATES:
        path = _ASSETS_DIR / name
        if not (path.exists() and path.is_file()):
            continue
        try:
            import io

            from PIL import Image

            img = Image.open(path).convert("RGB")
            if img.width > max_w:
                img = img.resize((max_w, round(img.height * max_w / img.width)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return _encode_image(path)  # raw fallback (no resize)
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

# ─── Dark "terminal" palette (active theme) ──────────────────────────────────
# Deep navy-charcoal canvas, slightly lifted surfaces, Tennessee Orange accent.
TERM_BG        = "#0B0E17"   # app canvas
TERM_SURFACE   = "#141B29"   # cards / sidebar / panels
TERM_SURFACE_2 = "#1C2438"   # raised elements (hover, headers)
TERM_BORDER    = "#26314A"   # hairline borders
TERM_BORDER_HI = "#33405E"   # brighter border (hover/focus)

# Functional aliases used throughout the app (now dark-mode values)
PRIMARY      = UT_ORANGE
SUCCESS      = "#34D399"   # positive (brighter for dark)
WARNING      = "#FBBF24"
DANGER       = "#F87171"   # negative
TEXT         = "#E6EAF2"   # primary text (off-white)
MUTED_TEXT   = "#8A95AB"   # secondary text
BG           = TERM_BG
SOFT_BG      = TERM_SURFACE
GRID         = "#212B40"   # chart gridlines on dark

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
            bgcolor="rgba(20,27,41,0.85)",
            bordercolor=GRID, borderwidth=1,
            orientation="h", y=-0.2, x=0,
        ),
        hoverlabel=dict(
            font=dict(family="Inter", size=12, color=TEXT),
            bgcolor=TERM_SURFACE_2, bordercolor=UT_ORANGE,
        ),
        margin=dict(t=50, b=50, l=60, r=20),
        colorscale=dict(
            # Sequential: dark surface → Tennessee Orange
            sequential=[[0, "#10203a"], [0.5, "#b3600f"], [1, UT_ORANGE]],
            # Diverging: red (negative) → dark neutral → orange (positive)
            diverging=[[0, "#F87171"], [0.5, "#141B29"], [1, UT_ORANGE]],
        ),
    )
)
pio.templates.default = "plotly_dark+quantlab"


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
    color: #F2F5FA;
    margin-bottom: 0.5rem;
}
.stApp h2 {
    font-weight: 600;
    letter-spacing: -0.015em;
    color: #E6EAF2;
    margin-top: 1.5rem;
}
.stApp h3 {
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #D4DBEA;
}
.stApp h5 {
    font-weight: 700;
    color: #FF8200;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 11px;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

/* Captions */
[data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p {
    color: #8A95AB;
}

/* Metric cards — dark panel with a subtle left accent (restrained: a single
   hairline + left bar reads cleaner than an orange top border on every card) */
[data-testid="stMetric"] {
    background: #141B29;
    padding: 14px 16px 14px 18px;
    border-radius: 8px;
    border: 1px solid #26314A;
    border-left: 3px solid #2E3A55;
    transition: border-color 0.18s, box-shadow 0.18s, transform 0.18s;
}
[data-testid="stMetric"]:hover {
    border-left-color: #FF8200;
    border-color: #33405E;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] {
    font-size: 11px;
    color: #8A95AB;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-size: 23px;
    font-weight: 700;
    color: #F2F5FA;
    line-height: 1.2;
    font-feature-settings: 'tnum' 1;
}
[data-testid="stMetricDelta"] {
    font-size: 12px;
    font-weight: 600;
}

/* Sidebar — dark panel with orange left border */
[data-testid="stSidebar"] {
    background-color: #0E1422;
    border-right: 3px solid #FF8200;
}

/* DataFrame / data editor — terminal-style framed grid */
.stDataFrame, [data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {
    border: 1px solid #26314A;
    border-radius: 8px;
    overflow: hidden;
}
/* Table column headers — uppercase muted, like a terminal */
.stDataFrame [data-testid="stDataFrameResizable"] [role="columnheader"],
[data-testid="stDataFrame"] [role="columnheader"] {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 10.5px;
    font-weight: 700;
    color: #8A95AB;
}
/* Static tables (st.table) */
.stTable, [data-testid="stTable"] table {
    border-color: #26314A;
}

/* Section divider */
hr {
    margin: 1.5rem 0;
    border-color: #26314A;
    border-top-width: 1px;
}

/* ──────────────────────────────────────────────────────────────────────────
   Top navigation (st.navigation position="top") — center the tabs.
   Streamlit doesn't expose a single canonical selector for the top nav;
   different versions/themes use different testids. Cast a wide net.
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stHeader"],
header[data-testid="stHeader"],
header.stAppHeader,
[data-testid="stMainBlockContainer"] > div:first-child > [data-testid="stHeader"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

/* The nav container itself */
[data-testid="stTopNav"],
[data-testid="stTopNavigation"],
[data-testid="stNavigation"],
[data-testid="stHeader"] nav,
[data-testid="stHeader"] [role="navigation"],
[data-testid="stHeader"] [role="menubar"],
[data-testid="stHeader"] [role="tablist"],
header nav,
header [role="navigation"],
header [role="menubar"],
header [role="tablist"] {
    display: flex !important;
    flex: 1 1 auto !important;
    margin-left: auto !important;
    margin-right: auto !important;
    justify-content: center !important;
    width: 100% !important;
}

/* Force the inner list of items to be centered */
[data-testid="stTopNav"] > div,
[data-testid="stTopNav"] > ul,
[data-testid="stHeader"] nav > div,
[data-testid="stHeader"] nav > ul,
[data-testid="stHeader"] [role="navigation"] > div,
[data-testid="stHeader"] [role="tablist"] > div,
header nav > div,
header nav > ul,
header [role="tablist"] > div {
    display: flex !important;
    justify-content: center !important;
    margin: 0 auto !important;
    flex-grow: 1 !important;
}

/* Section headers in the top nav (Streamlit groups pages under headers) */
[data-testid="stTopNav"] [data-testid*="navSectionHeader"],
[data-testid="stHeader"] [data-testid*="sectionHeader"] {
    text-align: center;
}

/* Style the active item in Tennessee Orange */
[data-testid="stTopNav"] a, header nav a, header [role="tab"] {
    font-weight: 500;
    color: #C8D0E0;
}
[data-testid="stTopNav"] a[aria-current="page"],
header nav a[aria-current="page"],
header [role="tab"][aria-selected="true"] {
    color: #FF8200 !important;
    font-weight: 700 !important;
}

/* If the toolbar (right-side menu/decoration area) is pushing nav left,
   shrink it so the nav has room to center. */
[data-testid="stHeader"] > div:last-child:not([data-testid*="nav" i]) {
    flex: 0 0 auto !important;
}

/* ── Center the top navigation ────────────────────────────────────────────
   This Streamlit build renders the page nav inside [data-testid="stToolbar"]
   using the rc-overflow widget. On Streamlit Cloud the toolbar also holds a
   "Fork"/GitHub badge on the right, so simply centering the flex would center
   the [nav + badge] group, not the nav. Instead we absolutely-center the nav
   container itself so it sits at true screen-center while the badge stays at
   the right edge. */
/* NOTE: top-nav centering intentionally removed. Streamlit's rc-overflow nav
   measures available width to decide how many of the 5 section tabs fit, and
   any attempt to reposition/resize/justify it collapsed every page into a
   bogus "N more" overflow button. Left as Streamlit's default so the 5 section
   dropdowns render correctly. */

/* Tabs — orange active state (for in-page st.tabs, not top nav) */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid #26314A;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px;
    font-weight: 500;
    color: #8A95AB;
    padding: 8px 14px;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #C8D0E0;
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
    border-color: #2E3A55;
}

/* Links */
a, .stApp a {
    color: #FF9A33;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}

/* Alerts — dark tint with orange accent */
.stAlert {
    border-radius: 8px;
    border-left: 4px solid #FF8200;
    background: #141B29;
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

/* UT brand header strip — top of every page (above the top-bar nav) */
.ut-header {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 12px 0 12px 0;
    border-bottom: 3px solid #FF8200;
    margin-bottom: 8px;
    margin-top: -1rem;
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
    font-size: 32px;
    letter-spacing: 0.04em;
    padding: 14px 22px;
    border-radius: 4px;
    line-height: 1;
    font-family: 'Inter', -apple-system, sans-serif;
    box-shadow: 0 2px 4px rgba(255, 130, 0, 0.2);
}
.ut-wordmark {
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.ut-supratitle {
    color: #FF8200;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}
.ut-title {
    color: #F2F5FA;
    font-weight: 700;
    font-size: 34px;
    letter-spacing: -0.02em;
    line-height: 1.1;
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
    color: #8A95AB;
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

/* Status bar — dark terminal strip */
.qstatus {
    background: #141B29;
    border: 1px solid #26314A;
    border-left: 4px solid #FF8200;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 1rem;
    font-size: 12px;
    color: #C8D0E0;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
}
.qstatus .qstatus-key {
    color: #6B7589;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 10px;
    margin-right: 6px;
}
.qstatus .qstatus-val {
    color: #F2F5FA;
    font-weight: 600;
    font-feature-settings: 'tnum' 1;
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
.qbadge-reup, .qbadge-low, .qbadge-success { background: rgba(52,211,153,0.14); color: #4ADE80; border: 1px solid rgba(52,211,153,0.35); }
.qbadge-add, .qbadge-info                  { background: rgba(255,130,0,0.15); color: #FFA64D; border: 1px solid rgba(255,130,0,0.40); }
.qbadge-hold, .qbadge-neutral              { background: rgba(138,149,171,0.14); color: #AEB8CC; border: 1px solid rgba(138,149,171,0.30); }
.qbadge-trim, .qbadge-medium, .qbadge-warning { background: rgba(251,191,36,0.14); color: #FBBF24; border: 1px solid rgba(251,191,36,0.35); }
.qbadge-exit, .qbadge-high, .qbadge-danger { background: rgba(248,113,113,0.15); color: #FB7185; border: 1px solid rgba(248,113,113,0.38); }

/* Expanders — dark panel */
[data-testid="stExpander"] {
    border: 1px solid #26314A;
    border-radius: 8px;
    background: #111826;
}
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover p {
    color: #FF8200;
}

/* Dividers Streamlit renders as its own element */
[data-testid="stDivider"] hr { border-color: #26314A; }

/* Code / inline mono */
.stApp code {
    background: #1C2438;
    color: #FFB066;
    border: 1px solid #26314A;
    border-radius: 4px;
    padding: 1px 5px;
}
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
            'style="height:104px; width:auto; display:block;">'
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

# Dark-tinted cell backgrounds with bright text, to blend with the dark grid.
REC_PALETTE: dict[str, tuple[str, str]] = {
    "REUP": ("#11321F", "#4ADE80"),
    "ADD":  ("#33240F", "#FFA64D"),
    "HOLD": ("#1E2638", "#AEB8CC"),
    "TRIM": ("#332A0D", "#FBBF24"),
    "EXIT": ("#331A1A", "#FB7185"),
}

SEVERITY_PALETTE: dict[str, tuple[str, str]] = {
    "high":   ("#331A1A", "#FB7185"),
    "medium": ("#332A0D", "#FBBF24"),
    "info":   ("#33240F", "#FFA64D"),
    "low":    ("#11321F", "#4ADE80"),
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
