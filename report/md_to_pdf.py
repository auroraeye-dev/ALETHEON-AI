"""
report/md_to_pdf.py
===================
Markdown-aware flowable renderer for the Aletheon PDF exporter.

WHY THIS EXISTS:
The original export_pdf.py was a hand-written line-by-line markdown parser.
It rendered ## headings and bullets fine, but didn't understand:
  - h3 headings (### subheading) — they showed as literal '###' in the PDF
  - markdown tables (the | col1 | col2 | syntax + |---|---| separator) — they
    showed as raw pipe characters

We now use the `markdown` Python library (already in the venv) to parse markdown
into HTML once, then walk the resulting block elements and emit ReportLab
flowables for each. ReportLab's Paragraph already speaks a subset of HTML for
inline elements (<b>, <i>, <a>, <font>) — so the inline rendering is free.
For block elements (h1-h6, p, ul/ol, table) we dispatch to small helpers.

Public entry:
  render_markdown_body(md_text, styles) -> list[Flowable]

styles is the same dict that export_pdf._styles() builds. We add a few extra
styles in get_extra_styles() that this module needs (h3, table cell, etc).
"""

import re
from xml.etree import ElementTree as ET

import markdown as md_lib
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether

# Brand palette — keep in sync with export_pdf.py.
NAVY = colors.HexColor("#0d4e78")
NAVY_LIGHT = colors.HexColor("#e9f1f7")
NAVY_VLIGHT = colors.HexColor("#f4f8fb")
INK = colors.HexColor("#1a2228")
MUTED = colors.HexColor("#5b6b78")
TABLE_BORDER = colors.HexColor("#c8d2dc")


def get_extra_styles() -> dict:
    """ParagraphStyles this module needs that aren't in export_pdf._styles()."""
    ss = getSampleStyleSheet()
    return {
        "h3": ParagraphStyle(
            "h3", parent=ss["Heading3"], fontName="Helvetica-Bold",
            fontSize=11.5, leading=15, textColor=NAVY,
            spaceBefore=10, spaceAfter=3),
        "table_header": ParagraphStyle(
            "table_header", parent=ss["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10, textColor=colors.white, alignment=0),
        "table_cell": ParagraphStyle(
            "table_cell", parent=ss["Normal"], fontName="Helvetica",
            fontSize=8, leading=10, textColor=INK, alignment=0),
        "table_caption": ParagraphStyle(
            "table_caption", parent=ss["Normal"], fontName="Helvetica-Oblique",
            fontSize=8.5, leading=11, textColor=MUTED,
            spaceBefore=2, spaceAfter=6),
    }


# ---- Inline conversion: HTML inline tags -> ReportLab Paragraph markup ----
# ReportLab Paragraph understands a subset of HTML/XML inline tags. Most map
# directly; we just need to convert <code>, drop unknown attributes, and clean
# the text so it parses cleanly.

_INLINE_CLEAN_PATTERNS = [
    # markdown library emits <code>...</code> which ReportLab doesn't style by
    # default — render as a different font.
    (re.compile(r"<code>(.*?)</code>", re.S),
     r'<font face="Courier" size="8">\1</font>'),
    # ReportLab doesn't handle <em>; it does handle <i>.
    (re.compile(r"<em>(.*?)</em>", re.S), r"<i>\1</i>"),
    (re.compile(r"</em>"), "</i>"),
    (re.compile(r"<em>"), "<i>"),
    # <strong> -> <b>
    (re.compile(r"<strong>(.*?)</strong>", re.S), r"<b>\1</b>"),
    (re.compile(r"</strong>"), "</b>"),
    (re.compile(r"<strong>"), "<b>"),
    # ReportLab requires self-closing tags to use XML form. The markdown library
    # emits HTML form <br>, which ReportLab's paraparser rejects with
    # "No content allowed in br tag" because it then tries to read what
    # follows as br's body. Normalize before render.
    (re.compile(r"<br\s*/?>", re.I), "<br/>"),
    (re.compile(r"<hr\s*/?>", re.I), "<hr/>"),
]


def _inner_html(elem) -> str:
    """Return the inner HTML of an ElementTree element as a string, preserving
    inline tags so ReportLab can style them.

    Note: ET.tostring(child) already includes child.tail in the serialization
    — we must NOT append child.tail again or content following inline elements
    gets duplicated (this caused the Sources-list double-rendering bug)."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        # ET.tostring INCLUDES the child's tail by default — don't double-append.
        s = ET.tostring(child, encoding="unicode", method="html")
        parts.append(s)
    raw = "".join(parts)
    for pat, repl in _INLINE_CLEAN_PATTERNS:
        raw = pat.sub(repl, raw)
    # Strip class= attrs etc. that ReportLab doesn't recognize and might trip on
    raw = re.sub(r' class="[^"]*"', "", raw)
    return raw


# ---- Table rendering ----

def _render_table(table_elem, styles: dict, max_width_inches: float = 6.5) -> Table:
    """Convert an HTML <table> element into a styled ReportLab Table flowable."""
    # Collect rows. The markdown library emits <thead><tr><th>... and <tbody><tr><td>...
    thead = table_elem.find("thead")
    tbody = table_elem.find("tbody")
    header_cells = []
    if thead is not None:
        tr = thead.find("tr")
        if tr is not None:
            for th in tr.findall("th"):
                header_cells.append(_inner_html(th))
    body_rows = []
    if tbody is not None:
        for tr in tbody.findall("tr"):
            row = []
            for td in tr.findall("td"):
                row.append(_inner_html(td))
            body_rows.append(row)
    if not header_cells and not body_rows:
        return None

    # Normalize column count.
    n_cols = max(len(header_cells), max((len(r) for r in body_rows), default=0))
    if n_cols == 0:
        return None
    while len(header_cells) < n_cols:
        header_cells.append("")
    for r in body_rows:
        while len(r) < n_cols:
            r.append("")

    # Wrap every cell in a Paragraph so it word-wraps inside the column.
    header_para = [Paragraph(c, styles["table_header"]) for c in header_cells]
    body_para = [[Paragraph(c, styles["table_cell"]) for c in r] for r in body_rows]
    data = [header_para] + body_para

    # Column widths: give the longer columns more room. Heuristic: weight each
    # column by the max content length in it (rough proxy).
    def _strip_html(s):
        return re.sub(r"<[^>]+>", "", s)
    widths_chars = []
    for col in range(n_cols):
        lengths = [len(_strip_html(header_cells[col]))]
        for r in body_rows:
            lengths.append(len(_strip_html(r[col])))
        widths_chars.append(max(8, min(80, max(lengths))))
    total_chars = sum(widths_chars) or 1
    total_width = max_width_inches * inch
    col_widths = [(w / total_chars) * total_width for w in widths_chars]

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header row: navy background, white text.
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        # Body rows: light alt-shading for readability.
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, NAVY_VLIGHT]),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        # Subtle grid lines.
        ("GRID", (0, 0), (-1, -1), 0.25, TABLE_BORDER),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, NAVY),
    ]))
    return tbl


# ---- Block dispatch ----

def render_markdown_body(md_text: str, styles: dict, callbacks: dict | None = None) -> list:
    """Parse markdown and return a list of ReportLab flowables.

    `styles` is the export_pdf style dict, EXTENDED with the extras from
    get_extra_styles() (the caller is expected to merge them).

    `callbacks` (optional) lets the caller hook block events — currently
    supports {'on_h2': fn(heading_text, story_list)} so export_pdf can apply
    its sources-detection logic.
    """
    callbacks = callbacks or {}
    extras = get_extra_styles()
    # Merge extras without clobbering caller-provided ones.
    for k, v in extras.items():
        styles.setdefault(k, v)

    # 'tables' extension enables pipe-table parsing. 'fenced_code' allows ``` blocks.
    html = md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
    # Wrap in a single root so ElementTree can parse it as one tree.
    try:
        root = ET.fromstring(f"<root>{html}</root>")
    except ET.ParseError:
        # If the markdown produced something ET can't parse, fall back to a
        # single paragraph with the raw text. (Should be rare in practice.)
        return [Paragraph(md_text, styles["body"])]

    story = []
    in_sources = False
    for elem in list(root):
        tag = elem.tag.lower()
        if tag == "h1":
            # Aletheon report top-level title is handled by the header block;
            # we skip h1 here so we don't print it twice.
            continue
        elif tag == "h2":
            heading = (elem.text or "").strip() + _inner_html(elem).replace(
                elem.text or "", "")  # in case of inline formatting
            heading = re.sub(r"<[^>]+>", "", _inner_html(elem)) or (elem.text or "").strip()
            cb = callbacks.get("on_h2")
            if cb:
                cb(heading, story)
            else:
                story.append(Paragraph(heading, styles["h2"]))
                story.append(HRFlowable(width="100%", thickness=0.4,
                                        color=NAVY_LIGHT,
                                        spaceBefore=0, spaceAfter=4))
            in_sources = heading.lower().startswith("sources")
            # Pass the in_sources state out via callbacks if needed.
            if "on_in_sources_change" in callbacks:
                callbacks["on_in_sources_change"](in_sources)
            continue
        elif tag == "h3":
            heading = re.sub(r"<[^>]+>", "", _inner_html(elem))
            story.append(Paragraph(heading, styles["h3"]))
            continue
        elif tag == "h4":
            heading = re.sub(r"<[^>]+>", "", _inner_html(elem))
            story.append(Paragraph(f"<b>{heading}</b>", styles["body"]))
            continue
        elif tag == "p":
            html_inner = _inner_html(elem)
            if not html_inner.strip():
                continue
            # If the caller wants to special-case source lines (badge rendering),
            # they can hook on_paragraph.
            cb = callbacks.get("on_paragraph")
            if cb:
                handled = cb(html_inner, story, in_sources=in_sources)
                if handled:
                    continue
            story.append(Paragraph(html_inner, styles["body"]))
            continue
        elif tag == "ul":
            for li in elem.findall("li"):
                content = _inner_html(li)
                # In the Sources section, markdown bullets carry the same source-line
                # shape as paragraphs do elsewhere. Route through on_paragraph so
                # the caller can render them as badge+tag rows. If we don't do
                # this, source bullets get the default body style AND duplicate
                # (since markdown sometimes wraps a soft-break URL into the same
                # bullet, our caller appends both, and the URL line crashes
                # ReportLab on <br>).
                if in_sources and callbacks.get("on_paragraph"):
                    handled = callbacks["on_paragraph"](
                        content, story, in_sources=True)
                    if handled:
                        continue
                story.append(Paragraph(content, styles["bullet"], bulletText="•"))
            continue
        elif tag == "ol":
            for i, li in enumerate(elem.findall("li"), 1):
                content = _inner_html(li)
                story.append(Paragraph(content, styles["bullet"],
                                       bulletText=f"{i}."))
            continue
        elif tag == "table":
            tbl = _render_table(elem, styles)
            if tbl is not None:
                story.append(Spacer(1, 4))
                story.append(tbl)
                story.append(Spacer(1, 6))
            continue
        elif tag == "hr":
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=NAVY_LIGHT,
                                    spaceBefore=4, spaceAfter=4))
            continue
        elif tag == "blockquote":
            # render quotes as italic indented paragraphs
            for sub in elem:
                content = _inner_html(sub)
                if content.strip():
                    story.append(Paragraph(f"<i>{content}</i>", styles["body"]))
            continue
        elif tag == "pre":
            # Fenced code block. Special case: if this contains the PRISMA flow
            # marker (the first line is "PRISMA-Style Evidence Flow" or any
            # heading containing "PRISMA"), substitute the native ReportLab
            # Drawing flowable for crisp output instead of monospace ASCII.
            code_text = "".join(elem.itertext()) or ""
            is_prisma = ("PRISMA" in code_text.split("\n")[0] if code_text else False) \
                or "PRISMA" in code_text[:120]
            if is_prisma:
                try:
                    from report.prisma import build_prisma_drawing
                    import core.combine as _combine
                    from report.prisma import PrismaCounts
                    pc = _combine._PRISMA_COUNTS
                    counts = PrismaCounts(
                        per_source=dict(pc.get("per_source", {})),
                        duplicates_removed=pc.get("duplicates_removed", 0),
                        off_drug_removed=pc.get("off_drug_removed", 0),
                        records_screened=pc.get("records_screened", 0),
                        reports_assessed=pc.get("reports_assessed", 0),
                        reports_excluded=dict(pc.get("reports_excluded", {})),
                        reports_sought=pc.get("records_screened", 0),
                        reports_retrieved=pc.get("records_screened", 0),
                        studies_included=pc.get("studies_included", 0),
                        reports_included=pc.get("reports_included", 0),
                        chunks_in_report=pc.get("chunks_in_report", 0),
                        sections_with_evidence=pc.get("sections_with_evidence", 0),
                    )
                    drawing = build_prisma_drawing(counts)
                    if drawing is not None:
                        # If we're near the end of a page, ReportLab will push
                        # the Drawing entirely to the next page, leaving a big
                        # empty gap. CondPageBreak triggers a clean page break
                        # ONLY if the Drawing can't fit in the remaining space.
                        # The threshold = drawing height + a bit of margin.
                        from reportlab.platypus import CondPageBreak, KeepTogether
                        story.append(CondPageBreak(drawing.height + 18))
                        story.append(Spacer(1, 4))
                        story.append(KeepTogether(drawing))
                        story.append(Spacer(1, 6))
                        continue
                except Exception as e:
                    # Fall back to monospace rendering if anything goes wrong
                    pass
            # Default: render the code text as a monospace paragraph block.
            # Escape special chars for ReportLab.
            escaped = (code_text.replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;")
                                .replace("\n", "<br/>"))
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f'<font face="Courier" size="7">{escaped}</font>',
                styles["body"]))
            story.append(Spacer(1, 6))
            continue
        else:
            # Unknown block — fall back to plain paragraph.
            html_inner = _inner_html(elem)
            if html_inner.strip():
                story.append(Paragraph(html_inner, styles["body"]))
    return story