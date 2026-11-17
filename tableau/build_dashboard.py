#!/usr/bin/env python3
"""Add an 'Overview' dashboard to the DL4DR Tableau workbook.

Usage:
    python build_dashboard.py DL4DR_dashboard_v2.twbx DL4DR_dashboard_v3.twbx

Reads a packaged workbook (.twbx) that already contains the two worksheets
  - 'Same models, same records: the winner depends on aggregation'
  - 'Paired comparison'
and writes a new .twbx with:
  1. an 'Overview' dashboard (1400 x 800): title, key-finding text,
     the two charts side by side, Favours colour legend, data footnote;
  2. the 'Paired comparison' sheet cleaned up: Measure Names removed from
     colour (one colour per category) and the reference line set to 0.
The input file is never modified.
"""
import re
import sys
import uuid
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

SLOPE = 'Same models, same records: the winner depends on aggregation'
FOREST = 'Paired comparison'
DASH = 'Overview'

TITLE = 'DL4DR Drug-Response Benchmark: 623 Unseen Cancer Cell Lines'
FINDING = ('Same 6 models, same 177,836 held-out records, two aggregations. '
           'DL4DR ranks 1st when pooled (it places unseen cell lines on the right '
           'potency scale) but 4th per cell line, where gradient-boosted trees '
           'order compounds better.')
FOOTER = ('Data: GDSC1/GDSC2 and ChEMBL (drug response), DepMap (expression, mutation). '
          '5-fold leave-cell-line-out cross-validation; every cell line is scored by a '
          'model that never saw it. tCNNS excluded (collapsed to a constant predictor). '
          'Code: github.com/bayjuan5/DL4DR')


def uid():
    return '{' + str(uuid.uuid4()).upper() + '}'


def text_zone(zid, y, h, text, size, color, bold=False, fixed_px=None):
    b = " bold='true'" if bold else ''
    fx = f" fixed-size='{fixed_px}' is-fixed='true'" if fixed_px else ''
    return (f"            <zone h='{h}' id='{zid}' type-v2='text' w='98400' x='800' y='{y}'{fx}>\n"
            f"              <formatted-text>\n"
            f"                <run fontcolor='{color}' fontsize='{size}'{b}>{escape(text)}</run>\n"
            f"              </formatted-text>\n"
            f"            </zone>\n")


def build(twb: str) -> str:
    for name in (SLOPE, FOREST):
        if f"<worksheet name='{escape(name, {chr(39): '&apos;'})}'>" not in twb and f"<worksheet name='{name}'>" not in twb:
            sys.exit(f'Worksheet not found: {name}')
    if f"<dashboard name='{DASH}'" in twb:
        sys.exit(f"Workbook already has a dashboard named '{DASH}'.")

    # datasource id of the forest plot (for the legend zone)
    m = re.search(r"<worksheet name='Paired comparison'>.*?<datasource caption='02_paired_vs_dl4dr' name='([^']+)'", twb, re.S)
    ds = m.group(1)

    # --- clean up the forest plot -------------------------------------------
    start = twb.index("<worksheet name='Paired comparison'>")
    end = twb.index('</worksheet>', start)
    ws = twb[start:end]
    ws = re.sub(r"\s*<color column='\[[^\]]+\]\.\[:Measure Names\]' />", '', ws)
    ws = re.sub(r"(<reference-line [^>]*formula='constant'[^>]*value=')[^']*(')", r"\g<1>0\g<2>", ws)
    # thinner CI bars: <style> must be the LAST child of <pane> (schema order)
    ws = re.sub(r"(<mark class='GanttBar' />.*?)(\s*</pane>)",
                lambda m: m.group(1) + "\n            <style>\n              <style-rule element='mark'>\n"
                "                <format attr='size' value='0.35' />\n              </style-rule>\n            </style>" + m.group(2),
                ws, count=1, flags=re.S)
    ws = re.sub(r"(<reference-line [^>]*)label-type='[^']*'", r"\g<1>label-type='none'", ws)
    # 2. median dots: black, larger, drawn on top of the CI bars
    ws = re.sub(r"(<pane id='1' [^>]*sum:median_delta:qk\]'>.*?<mark class='Circle' />)\s*<encodings>.*?</encodings>",
                r"\g<1>\n            <style>\n              <style-rule element='mark'>\n"
                r"                <format attr='mark-color' value='#000000' />\n"
                r"                <format attr='size' value='1.6' />\n"
                r"              </style-rule>\n            </style>", ws, flags=re.S)
    twb = twb[:start] + ws + twb[end:]

    # forest plot: descriptive title
    title = ("<layout-options>\n        <title>\n          <formatted-text>\n"
             "            <run fontcolor='#000000'>Per-cell-line \u0394R\u00b2 vs. DL4DR (623 paired cell lines)</run>\n"
             "          </formatted-text>\n        </title>\n      </layout-options>\n      ")
    if not re.search(r"<worksheet name='Paired comparison'>\r?\n\s*<layout-options>", twb):
        twb = re.sub(r"(<worksheet name='Paired comparison'>\r?\n\s*)(<table>)",
                     lambda m: m.group(1) + title + m.group(2), twb, count=1)

    # legend: shorter label for the 'Tie (not significant)' category
    twb = re.sub(r"<column caption='Favours' datatype='string' name='\[favours\]' role='dimension' type='nominal' />",
                 "<column caption='Favours' datatype='string' name='[favours]' role='dimension' type='nominal'>"
                 "<aliases><alias key='&quot;Tie (not significant)&quot;' value='Not significant' /></aliases></column>",
                 twb, count=1)

    # ranking chart: hide the continuous Rank axis (it showed 1.5, 2.5 ...);
    # vertical order already shows the rank, top = 1st
    twb = twb.replace("<style-rule element='worksheet'>\n            <format attr='display-field-labels' scope='cols' value='false' />",
                      "<style-rule element='worksheet'>\n            <format attr='display-field-labels' scope='cols' value='false' />", 1)
    twb = re.sub(r"(<encoding attr='space' class='0' domain-expand='false' field='(\[[^\]]+\])\.\[sum:rank:qk\]'[^>]*/>)",
                 r"\g<1>\n            <format attr='display' class='0' field='\g<2>.[sum:rank:qk]' scope='rows' value='false' />", twb, count=1)
    twb = twb.replace(f"param='[{ds}].[none:favours:nk]&#10;[{ds}].[:Measure Names]'",
                      f"param='[{ds}].[none:favours:nk]'")

    # --- dashboard -----------------------------------------------------------
    # coordinates are in 1/100000 of the dashboard; vertical stack:
    # title 7% | finding 9% | charts 70% (+ legend) | footer 8%
    dash = (
        "  <dashboards>\n"
        f"    <dashboard name='{DASH}'>\n"
        "      <layout-options>\n"
        "        <title>\n"
        "          <formatted-text>\n"
        f"            <run bold='true' fontcolor='#1f2d3d' fontsize='20'>{escape(TITLE)}</run>\n"
        "          </formatted-text>\n"
        "        </title>\n"
        "      </layout-options>\n"
        "      <style />\n"
        "      <size maxheight='800' maxwidth='1400' minheight='800' minwidth='1400' />\n"
        "      <zones>\n"
        "        <zone h='100000' id='1' type-v2='layout-basic' w='100000' x='0' y='0'>\n"
        "          <zone h='98000' id='2' param='vert' type-v2='layout-flow' w='98400' x='800' y='1000'>\n"
        "            <zone h='7000' id='3' type-v2='title' w='98400' x='800' y='1000' fixed-size='52' is-fixed='true' />\n"
        + text_zone(4, 8000, 9000, FINDING, 12, '#333333', fixed_px=70) +
        "            <zone h='72000' id='5' param='horz' type-v2='layout-flow' w='98400' x='800' y='17000'>\n"
        f"              <zone h='72000' id='6' name='{escape(SLOPE)}' w='46000' x='800' y='17000' fixed-size='620' is-fixed='true' />\n"
        "              <zone h='72000' id='7' param='vert' type-v2='layout-flow' w='52400' x='46800' y='17000'>\n"
        f"                <zone h='62000' id='8' name='{FOREST}' w='52400' x='46800' y='17000' />\n"
        f"                <zone h='10000' id='9' name='{FOREST}' pane-specification-id='2' param='[{ds}].[none:favours:nk]' type-v2='color' w='52400' x='46800' y='79000' fixed-size='80' is-fixed='true' />\n"
        "              </zone>\n"
        "            </zone>\n"
        + text_zone(10, 89000, 10000, FOOTER, 9, '#7f7f7f', fixed_px=50) +
        "          </zone>\n"
        "        </zone>\n"
        "      </zones>\n"
        f"      <simple-id uuid='{uid()}' />\n"
        "    </dashboard>\n"
        "  </dashboards>\n"
    )
    twb = twb.replace('  <windows', dash + '  <windows', 1)

    # --- dashboard window (open on the dashboard) ----------------------------
    twb = twb.replace(" maximized='true'", '')
    win = (
        f"    <window class='dashboard' maximized='true' name='{DASH}'>\n"
        "      <viewpoints>\n"
        f"        <viewpoint name='{FOREST}'>\n"
        "          <zoom type='entire-view' />\n"
        "        </viewpoint>\n"
        f"        <viewpoint name='{escape(SLOPE)}'>\n"
        "          <zoom type='entire-view' />\n"
        "        </viewpoint>\n"
        "      </viewpoints>\n"
        "      <active id='-1' />\n"
        f"      <simple-id uuid='{uid()}' />\n"
        "    </window>\n"
    )
    twb = twb.replace('  </windows>', win + '  </windows>', 1)

    ET.fromstring(twb.encode('utf-8'))  # sanity check: still well-formed XML
    return twb


def main(src, dst):
    with zipfile.ZipFile(src) as zin:
        twb_name = next(n for n in zin.namelist() if n.endswith('.twb') and '/' not in n)
        twb = zin.read(twb_name).decode('utf-8')
        new = build(twb)
        out_twb = dst.rsplit('/', 1)[-1].rsplit('\\', 1)[-1].replace('.twbx', '.twb')
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == twb_name:
                    zout.writestr(out_twb, new.encode('utf-8'))
                else:
                    zout.writestr(item, zin.read(item.filename))
    print(f'Wrote {dst} (dashboard "{DASH}" added)')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
