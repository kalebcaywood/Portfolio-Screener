"""Excel correlation workbook — a PivotTable-ready export of the holdings grid.

The centrepiece is the "Pivot Data" sheet: one row per ordered pair of holdings,
tagged with each side's sector, region and weight. Drop it into an Excel
PivotTable (Sector A on rows, Sector B on columns, Average of Correlation in
values) and you get the same grid the app draws, sliceable any way you like.

Pairs are written in BOTH directions. A single direction would give a triangular
pivot with half the grid blank; both directions make it symmetric, which is what
someone expects when they drop it into a PivotTable.

The By Sector / By Region / By Sector x Region tabs are built with AVERAGEIFS
against that sheet rather than pre-computed numbers, so they stay live if rows
are filtered or edited.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Arial"
HDR_FILL = PatternFill("solid", fgColor="1F3864")
HDR_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=10)
TITLE_FONT = Font(name=FONT, bold=True, size=13, color="1F3864")
BODY = Font(name=FONT, size=10)
NOTE = Font(name=FONT, size=9, italic=True, color="595959")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# Green = uncorrelated (diversifying), red = moves together (concentrated).
# Deliberately inverted vs. the usual heat scale: for a portfolio, high
# correlation is the bad outcome.
SCALE = ColorScaleRule(
    start_type="num", start_value=-0.2, start_color="FF2E7D32",
    mid_type="num", mid_value=0.35, mid_color="FFFFF3CD",
    end_type="num", end_value=1.0, end_color="FFC62828",
)

# Pivot Data column layout — referenced by the AVERAGEIFS formulas below.
COLS = ["Ticker A", "Sector A", "Region A", "Weight A",
        "Ticker B", "Sector B", "Region B", "Weight B",
        "Correlation", "Same Sector", "Same Region", "Pair Weight"]
C_SECA, C_SECB = "B", "F"
C_REGA, C_REGB = "C", "G"
C_RHO = "I"


def _style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill, cell.font = HDR_FILL, HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        cell.border = BOX
    # Use the string form: ws.cell() would MATERIALISE that cell, inflating
    # max_row so the next append() starts a row too low and leaves a blank.
    ws.freeze_panes = f"A{row + 1}"


def _matrix_sheet(wb, title, labels, crit_a, crit_b, first, last, counts=None):
    """A group x group grid of AVERAGEIFS against Pivot Data.

    IFERROR guards groups holding a single name: with no distinct pairs,
    AVERAGEIFS divides by zero.
    """
    ws = wb.create_sheet(title)
    ws["A1"] = title
    ws["A1"].font = TITLE_FONT
    ws["A2"] = ("Average correlation between every holding in the row group and "
                "every holding in the column group. The diagonal is the group's "
                "internal cohesion (self-pairs excluded); blank means the group "
                "holds a single name.")
    ws["A2"].font = NOTE
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(4, len(labels)))
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[2].height = 30

    top = 4
    ws.cell(row=top, column=1, value="Group")
    for j, lab in enumerate(labels):
        txt = lab if counts is None else f"{lab} ({counts.get(lab, 0)})"
        ws.cell(row=top, column=2 + j, value=txt)
    _style_header(ws, top, len(labels) + 1)

    # Ranges are derived from the rows actually written, not from a counter, so
    # they cannot drift out of step with the data.
    rho = f"'Pivot Data'!${C_RHO}${first}:${C_RHO}${last}"
    a_rng = f"'Pivot Data'!${crit_a}${first}:${crit_a}${last}"
    b_rng = f"'Pivot Data'!${crit_b}${first}:${crit_b}${last}"
    for i, ra in enumerate(labels):
        r = top + 1 + i
        lab = ra if counts is None else f"{ra} ({counts.get(ra, 0)})"
        c0 = ws.cell(row=r, column=1, value=lab)
        c0.font = Font(name=FONT, bold=True, size=10)
        c0.border = BOX
        for j, rb in enumerate(labels):
            # Match on the raw labels: the row/column headers carry a "(n)"
            # count suffix, so pointing AVERAGEIFS at them would match nothing.
            # Quotes inside a label would terminate the string literal early.
            qa, qb = ra.replace('"', '""'), rb.replace('"', '""')
            f = f'=IFERROR(AVERAGEIFS({rho},{a_rng},"{qa}",{b_rng},"{qb}"),"")'
            cell = ws.cell(row=r, column=2 + j, value=f)
            cell.number_format = "0.00"
            cell.font = BODY
            cell.border = BOX
            cell.alignment = Alignment(horizontal="center")
    last = get_column_letter(1 + len(labels))
    ws.conditional_formatting.add(f"B{top+1}:{last}{top+len(labels)}", SCALE)
    ws.column_dimensions["A"].width = 30
    for j in range(len(labels)):
        ws.column_dimensions[get_column_letter(2 + j)].width = 16
    return ws


def build_workbook(corr: pd.DataFrame, holdings: pd.DataFrame,
                   sector_of: dict[str, str], path: str,
                   window: str = "2y", scope: str = "All funds") -> str:
    """Write the correlation workbook to ``path`` and return it."""
    tickers = [t for t in corr.columns if t in set(holdings["yahoo"])]
    mv = holdings.groupby("yahoo")["market_value"].sum().reindex(tickers).fillna(0.0)
    tot = float(mv.sum()) or 1.0
    w = {t: float(mv[t]) / tot for t in tickers}
    region_of = dict(zip(holdings["yahoo"], holdings["region"]))
    name_of = (dict(zip(holdings["yahoo"], holdings["security_name"]))
               if "security_name" in holdings.columns else {})
    sec = {t: sector_of.get(t, "Unknown") for t in tickers}
    reg = {t: region_of.get(t, "Unknown") for t in tickers}

    wb = Workbook()

    # ── Pivot Data: both directions, so a PivotTable comes out symmetric ──
    ws = wb.active
    ws.title = "Pivot Data"
    ws.append(COLS)
    _style_header(ws, 1, len(COLS))
    n = 0
    for a in tickers:
        for b in tickers:
            if a == b:
                continue
            v = corr.loc[a, b]
            if pd.isna(v):
                continue
            ws.append([a, sec[a], reg[a], w[a], b, sec[b], reg[b], w[b],
                       round(float(v), 4),
                       "Yes" if sec[a] == sec[b] else "No",
                       "Yes" if reg[a] == reg[b] else "No",
                       round(w[a] * w[b], 8)])
            n += 1
    first, last = 2, ws.max_row          # what was actually written
    assert last - first + 1 == n, f"wrote {n} pairs but rows span {first}-{last}"
    for r in range(first, last + 1):
        for c in range(1, len(COLS) + 1):
            ws.cell(row=r, column=c).font = BODY
        ws.cell(row=r, column=4).number_format = "0.00%"
        ws.cell(row=r, column=8).number_format = "0.00%"
        ws.cell(row=r, column=9).number_format = "0.00"
        ws.cell(row=r, column=12).number_format = "0.000000"
    ws.conditional_formatting.add(f"I{first}:I{last}", SCALE)
    for col, wd in zip("ABCDEFGHIJKL",
                       [12, 20, 16, 11, 12, 20, 16, 11, 12, 12, 12, 13]):
        ws.column_dimensions[col].width = wd
    ws.auto_filter.ref = f"A1:L{last}"

    # ── Pre-built grids ──
    sectors = sorted({sec[t] for t in tickers})
    regions = sorted({reg[t] for t in tickers})
    scount = {s: sum(1 for t in tickers if sec[t] == s) for s in sectors}
    rcount = {r: sum(1 for t in tickers if reg[t] == r) for r in regions}
    _matrix_sheet(wb, "By Sector", sectors, C_SECA, C_SECB, first, last, scount)
    _matrix_sheet(wb, "By Region", regions, C_REGA, C_REGB, first, last, rcount)

    # ── Holdings reference ──
    hw = wb.create_sheet("Holdings")
    hw.append(["Ticker", "Security Name", "Sector", "Region", "Weight", "Funds"])
    _style_header(hw, 1, 6)
    fmap = holdings.groupby("yahoo")["fund"].apply(
        lambda s: ", ".join(sorted(set(s.astype(str)))))
    for t in sorted(tickers, key=lambda x: -w[x]):
        hw.append([t, name_of.get(t, ""), sec[t], reg[t], w[t], fmap.get(t, "")])
    for r in range(2, len(tickers) + 2):
        for c in range(1, 7):
            hw.cell(row=r, column=c).font = BODY
        hw.cell(row=r, column=5).number_format = "0.00%"
    for col, wd in zip("ABCDEF", [12, 34, 22, 16, 10, 34]):
        hw.column_dimensions[col].width = wd
    hw.auto_filter.ref = f"A1:F{len(tickers) + 1}"

    # ── Read me, first tab ──
    rm = wb.create_sheet("Read me", 0)
    rm["A1"] = "Holdings correlation workbook"
    rm["A1"].font = Font(name=FONT, bold=True, size=16, color="1F3864")
    rows = [
        ("", ""),
        ("Scope", scope),
        ("Window", f"{window} of daily total returns"),
        ("Holdings", len(tickers)),
        ("Pairs in Pivot Data", n),
        ("Book-wide average correlation", f"=ROUND(AVERAGE('Pivot Data'!I{first}:I{last}),3)"),
        ("Highest pair", f"=ROUND(MAX('Pivot Data'!I{first}:I{last}),3)"),
        ("Lowest pair", f"=ROUND(MIN('Pivot Data'!I{first}:I{last}),3)"),
        ("", ""),
        ("TAB", "WHAT IT IS"),
        ("Pivot Data", "One row per pair of holdings. This is the sheet to build "
                       "a PivotTable from: Sector A on Rows, Sector B on Columns, "
                       "Average of Correlation in Values."),
        ("By Sector", "Sector x sector average correlation, already built."),
        ("By Region", "Region x region average correlation, already built."),
        ("Holdings", "Every holding with its sector, region, weight and fund."),
        ("", ""),
        ("How to read it", "Correlation runs -1 to +1 on daily returns. Near 0 "
                           "means the two move independently. Green = "
                           "diversifying, red = moving together."),
        ("Note", "Pairs appear in both directions (A-B and B-A) so a PivotTable "
                 "returns a full symmetric grid rather than a triangle. This "
                 "does not change any average."),
        ("Note", "The diagonal of the built grids excludes each holding's "
                 "correlation with itself, so it reads as genuine internal "
                 "cohesion rather than being pulled toward 1.00."),
        ("Note", "Asia-vs-Americas correlations look low partly because the "
                 "exchanges do not trade at the same hours; daily closes "
                 "understate the true linkage."),
    ]
    for i, (k, v) in enumerate(rows, start=3):
        rm.cell(row=i, column=1, value=k).font = Font(name=FONT, bold=True, size=10)
        c = rm.cell(row=i, column=2, value=v)
        c.font = BODY
        c.alignment = Alignment(wrap_text=True, vertical="top")
    rm.cell(row=8, column=2).number_format = "0.000"
    rm.cell(row=9, column=2).number_format = "0.000"
    rm.cell(row=10, column=2).number_format = "0.000"
    rm.column_dimensions["A"].width = 32
    rm.column_dimensions["B"].width = 88
    for i in range(3, len(rows) + 3):
        rm.row_dimensions[i].height = 30

    # openpyxl writes formulas with no cached value, so a reader that trusts
    # the cache sees blanks. This makes Excel recalculate the whole book on
    # open, so the grids are populated the moment the file is opened.
    wb.calculation.fullCalcOnLoad = True
    wb.save(path)
    return path
