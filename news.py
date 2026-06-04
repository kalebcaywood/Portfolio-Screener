"""WSJ headline ticker — a circulating news liner for the top of every page.

Pulls public Wall Street Journal RSS feeds (Markets / Business / Tech), parses
them with the standard library (no extra dependency), caches for 15 minutes,
and renders a UT-themed marquee that scrolls headlines across the page. Each
headline links back to the WSJ article — this displays only short headline
text with attribution, the conventional use of an RSS feed.
"""
from __future__ import annotations

import html
import urllib.request
import xml.etree.ElementTree as ET

import streamlit as st

# Public WSJ / Dow Jones RSS endpoints. Markets first — most relevant for a
# portfolio-analytics audience — then Business and Tech for breadth.
WSJ_FEEDS: dict[str, str] = {
    "Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "Business": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
    "Technology": "https://feeds.a.dj.com/rss/RSSWSJD.xml",
}

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Height of the fixed ticker bar (px). The Streamlit header, sidebar, and main
# content are offset downward by this amount so the ticker sits above the nav.
TICKER_H = 34
# Streamlit's default top padding on the main block container (6rem = 96px),
# which clears the 60px header. We add TICKER_H so content still clears the
# header after it's been pushed down by the ticker.
_MAIN_PAD_BASE = 96


def _fetch_feed(url: str, timeout: int = 8) -> list[dict]:
    """Fetch and parse a single RSS feed into a list of {title, link} dicts."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
    except Exception:
        return []

    items: list[dict] = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        if not title or not link:
            continue
        # Drop boilerplate / roundup filler that reads poorly in a ticker.
        if title.lower().startswith(("tech, media", "market talk")):
            continue
        items.append({"title": title, "link": link})
    return items


@st.cache_data(ttl=900, show_spinner=False)
def fetch_wsj_headlines(max_items: int = 24) -> list[dict]:
    """Return a de-duplicated, interleaved list of WSJ headlines.

    Cached for 15 minutes so page loads stay fast and we stay polite to the
    feed. On total network failure this returns an empty list and the caller
    simply renders nothing.
    """
    per_feed = {name: _fetch_feed(url) for name, url in WSJ_FEEDS.items()}

    # Round-robin interleave so the ribbon mixes Markets / Business / Tech
    # rather than dumping one section before the next.
    seen: set[str] = set()
    merged: list[dict] = []
    idx = 0
    feeds = list(per_feed.values())
    while feeds and len(merged) < max_items:
        progressed = False
        for f in feeds:
            if idx < len(f):
                item = f[idx]
                key = item["title"].lower()
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
                    if len(merged) >= max_items:
                        break
                progressed = True
        if not progressed:
            break
        idx += 1
    return merged


def render_news_ticker(label: str = "WSJ") -> None:
    """Render the scrolling WSJ headline ribbon. No-op if no headlines load."""
    headlines = fetch_wsj_headlines()
    if not headlines:
        return

    # Build one run of the headline track (anchors separated by a diamond).
    sep = '<span class="news-tkr-sep">&#9670;</span>'
    links = sep.join(
        f'<a href="{html.escape(h["link"], quote=True)}" target="_blank" '
        f'rel="noopener noreferrer" title="Open on WSJ.com">'
        f'{html.escape(h["title"])}</a>'
        for h in headlines
    )
    # Duplicate the run so the -50% keyframe loops seamlessly with no gap.
    track = links + sep + links

    # Scroll speed scales with headline count so it reads at a steady pace
    # regardless of how many items loaded.
    duration = max(40, min(220, len(headlines) * 7))

    st.markdown(
        f"""
<style>
/* ── Ticker pinned as a full-width bar at the very top, above the nav ──
   It's position:fixed so it overlays the very top of the viewport; the
   Streamlit header (top nav), sidebar, and main content are each pushed
   down by the ticker height so nothing is hidden underneath it. These
   offsets live here (not in theme.py) so that if the feed is unreachable
   and the ticker doesn't render, the layout stays in its normal position. */
.news-tkr {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    width: 100%;
    z-index: 1000000;          /* above the Streamlit header (z 999990) */
    display: flex;
    align-items: stretch;
    height: {TICKER_H}px;
    margin: 0;
    background: #0E1422;
    border: none;
    border-bottom: 1px solid #26314A;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    overflow: hidden;
}}
/* Push the nav header, sidebar, and main content below the fixed ticker. */
header[data-testid="stHeader"] {{ top: {TICKER_H}px !important; }}
[data-testid="stSidebar"] {{ top: {TICKER_H}px !important; }}
[data-testid="stMainBlockContainer"] {{
    padding-top: {_MAIN_PAD_BASE + TICKER_H}px !important;
}}
.news-tkr-label {{
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 7px;
    background: #FF8200;
    color: #ffffff;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    padding: 0 14px;
    white-space: nowrap;
}}
.news-tkr-label::before {{
    content: "";
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #ffffff;
    box-shadow: 0 0 0 0 rgba(255,255,255,0.7);
    animation: news-tkr-pulse 1.8s infinite;
}}
.news-tkr-view {{
    position: relative;
    flex: 1;
    overflow: hidden;
    display: flex;
    align-items: center;
}}
.news-tkr-track {{
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
    padding-left: 100%;
    will-change: transform;
    animation: news-tkr-scroll {duration}s linear infinite;
}}
.news-tkr-track a {{
    color: #C8D0E0;
    font-size: 13px;
    font-weight: 500;
    text-decoration: none;
    padding: 0 2px;
}}
.news-tkr-track a:hover {{
    color: #FF8200;
    text-decoration: underline;
}}
.news-tkr-sep {{
    color: #FF8200;
    font-size: 8px;
    margin: 0 14px;
    position: relative;
    top: -1px;
}}
.news-tkr:hover .news-tkr-track {{
    animation-play-state: paused;
}}
@keyframes news-tkr-scroll {{
    0%   {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
}}
@keyframes news-tkr-pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(255,255,255,0.7); }}
    70%  {{ box-shadow: 0 0 0 6px rgba(255,255,255,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(255,255,255,0); }}
}}
</style>
<div class="news-tkr">
    <div class="news-tkr-label">{html.escape(label)} &middot; Markets</div>
    <div class="news-tkr-view">
        <div class="news-tkr-track">{track}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
