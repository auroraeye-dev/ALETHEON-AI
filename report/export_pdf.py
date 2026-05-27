"""
report/export_pdf.py
====================
B3 — branded PDF export.

Renders an Aletheon report (markdown) into a professional, branded PDF:
  - Title block: ALETHEON wordmark + tagline + thin swoosh rule with star
  - Drug name + generation metadata
  - Styled section headings, color-coded tier badges in the Sources list
  - Footer on every page: disclaimer + page number
  - Aletheon navy (#0d4e78) as the brand accent throughout

Uses reportlab Platypus. The wordmark is drawn as styled text (matching the
logo's wide-tracked serif look); if a real logo image exists at
assets/aletheon_logo.png it is used instead.
"""

import os
import re
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
                                Table, TableStyle, Image, KeepTogether)

from core.config import config
from core.logging_setup import log

# ---- Brand palette (sampled from the Aletheon logo) ----
NAVY = colors.HexColor("#0d4e78")
NAVY_DARK = colors.HexColor("#093a5a")
NAVY_LIGHT = colors.HexColor("#e9f1f7")
INK = colors.HexColor("#1a2228")
MUTED = colors.HexColor("#5b6b78")

TAGLINE = "Designing the future of medical intelligence"

# Tier badge colors (kept tasteful, within the navy family + accents).
TIER_COLORS = {
    "regulatory":    colors.HexColor("#0d4e78"),   # navy — strongest
    "peer_reviewed": colors.HexColor("#1f7a5a"),   # green
    "real_world":    colors.HexColor("#9a6a16"),   # amber
    "preprint":      colors.HexColor("#7a5cb0"),   # purple — weakest
}


def _styles():
    ss = getSampleStyleSheet()
    out = {}
    out["wordmark"] = ParagraphStyle("wordmark", parent=ss["Title"], fontName="Times-Roman",
                                     fontSize=30, leading=34, textColor=NAVY,
                                     alignment=TA_CENTER, spaceAfter=2, tracking=8)
    out["tagline"] = ParagraphStyle("tagline", parent=ss["Normal"], fontName="Times-Italic",
                                    fontSize=10.5, textColor=MUTED, alignment=TA_CENTER,
                                    spaceBefore=2, spaceAfter=2)
    out["doctitle"] = ParagraphStyle("doctitle", parent=ss["Title"], fontName="Helvetica-Bold",
                                     fontSize=20, leading=24, textColor=INK, spaceBefore=14, spaceAfter=2)
    out["meta"] = ParagraphStyle("meta", parent=ss["Normal"], fontName="Helvetica",
                                 fontSize=8.5, textColor=MUTED, spaceAfter=10)
    out["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                               fontSize=13, leading=16, textColor=NAVY, spaceBefore=14, spaceAfter=5)
    out["body"] = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica",
                                 fontSize=10, leading=15, textColor=INK, spaceAfter=5, alignment=TA_LEFT)
    out["bullet"] = ParagraphStyle("bullet", parent=out["body"], leftIndent=14, bulletIndent=4,
                                   spaceAfter=4)
    out["source"] = ParagraphStyle("source", parent=ss["Normal"], fontName="Helvetica",
                                   fontSize=8.5, leading=12, textColor=INK, leftIndent=6, spaceAfter=4)
    out["disclaimer"] = ParagraphStyle("disclaimer", parent=ss["Normal"], fontName="Helvetica",
                                       fontSize=7.5, textColor=MUTED, alignment=TA_CENTER)
    return out


def _inline_md(text: str) -> str:
    """Convert minimal inline markdown to reportlab markup; escape stray &<>."""
    # First, neutralize any literal HTML tags that leaked in from source titles
    # (e.g. a paper title containing "<sub>12</sub>"). Unescape entities first,
    # then strip the tags, so we don't render raw "&lt;sub&gt;".
    text = (text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
    text = re.sub(r"</?(?:sub|sup|i|b|br|font|span)[^>]*>", "", text)

    # Map Unicode subscript/superscript digits to ASCII (the PDF base fonts
    # render them as black boxes otherwise — see PDF skill warning).
    subs = "₀₁₂₃₄₅₆₇₈₉"; sups = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    for i, (lo, hi) in enumerate(zip(subs, sups)):
        text = text.replace(lo, str(i)).replace(hi, str(i))
    # A few other common offenders -> safe text
    text = (text.replace("≥", ">=").replace("≤", "<=").replace("–", "-")
                .replace("™", "(TM)").replace("\u2009", " "))

    # Now re-escape for reportlab, then apply our own markup.
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # bold **x**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # italic *x*
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    # citation tags [E#]/[A#]/[B#] -> small navy
    text = re.sub(r"\[(E\d+)\]", r'<font color="#0d4e78">[\1]</font>', text)
    text = re.sub(r"\[([AB]\d+)\]", r'<font color="#0d4e78">[\1]</font>', text)
    return text


def _badge(tier: str):
    """A small colored tier badge as a mini-table cell."""
    label = tier.replace("_", " ")
    c = TIER_COLORS.get(tier, MUTED)
    t = Table([[label]], colWidths=[1.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), c),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _header_block(story, st, drug, meta_line):
    """Build the branded title block: logo/wordmark + tagline + swoosh."""
    logo_path = os.path.join("assets", "aletheon_logo.png")
    if os.path.exists(logo_path):
        try:
            img = Image(logo_path)
            # scale to ~2.6 inch wide, keep aspect
            iw, ih = img.imageWidth, img.imageHeight
            target_w = 2.8 * inch
            img.drawWidth = target_w
            img.drawHeight = target_w * ih / iw
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Spacer(1, 4))
        except Exception:
            story.append(Paragraph("ALETHEON", st["wordmark"]))
    else:
        # Styled text wordmark (wide-tracked serif, matching the logo feel)
        story.append(Paragraph("A L E T H E O N", st["wordmark"]))
        # thin swoosh rule with a centered star/diamond
        story.append(Spacer(1, 1))
        story.append(HRFlowable(width="55%", thickness=0.8, color=NAVY,
                                lineCap="round", spaceBefore=1, spaceAfter=1,
                                hAlign="CENTER"))
    story.append(Paragraph(TAGLINE, st["tagline"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceBefore=4, spaceAfter=2))
    # Document title + meta
    story.append(Paragraph(f"Drug Intelligence Report: {drug.title()}", st["doctitle"]))
    story.append(Paragraph(meta_line, st["meta"]))


def _footer(canvas, doc):
    canvas.saveState()
    w, h = letter
    # top-of-footer rule
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.5)
    canvas.line(0.9 * inch, 0.72 * inch, w - 0.9 * inch, 0.72 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    disclaimer = ("For research and informational use only — not medical advice. "
                  "Generated by Aletheon.")
    canvas.drawCentredString(w / 2, 0.56 * inch, disclaimer)
    canvas.drawRightString(w - 0.9 * inch, 0.56 * inch, f"Page {doc.page}")
    canvas.drawString(0.9 * inch, 0.56 * inch, "ALETHEON")
    canvas.restoreState()


# Sources lines look like:  - **[E1][E2]** (tier) Title — source:id
_SRC_RE = re.compile(r"^-\s*\*\*(.+?)\*\*\s*\((\w+)\)\s*(.+?)\s*—\s*(.+?)\s*$")


def markdown_to_pdf(report_md: str, drug: str, out_path: str) -> str:
    st = _styles()
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                            topMargin=0.8 * inch, bottomMargin=0.95 * inch,
                            leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            title=f"Aletheon Report: {drug}", author="Aletheon")
    story = []

    # Pull the generated-meta line out of the markdown header if present.
    meta_line = ""
    m = re.search(r"_Generated (.+?)_", report_md)
    if m:
        meta_line = m.group(1).strip()
    else:
        meta_line = datetime.now().strftime("%Y-%m-%d %H:%M")

    _header_block(story, st, drug, meta_line)

    in_sources = False
    lines = report_md.splitlines()
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        # skip the top markdown title and the meta italics (already in header)
        if line.startswith("# "):
            continue
        if line.startswith("_Generated"):
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            in_sources = heading.lower().startswith("sources")
            story.append(Paragraph(heading, st["h2"]))
            story.append(HRFlowable(width="100%", thickness=0.4, color=NAVY_LIGHT,
                                    spaceBefore=0, spaceAfter=4))
            continue
        if in_sources:
            sm = _SRC_RE.match(line)
            if sm:
                tags, tier, title, srcid = sm.groups()
                badge = _badge(tier)
                txt = Paragraph(f'<b><font color="#0d4e78">{tags}</font></b> '
                                f'{_inline_md(title)}  '
                                f'<font color="#5b6b78" size="7">{_inline_md(srcid)}</font>',
                                st["source"])
                row = Table([[badge, txt]], colWidths=[1.05 * inch, 5.0 * inch])
                row.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (0, 0), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                story.append(row)
            else:
                # Skip raw URL continuation lines — they clutter the print
                # layout and the source id already identifies the document.
                if line.strip().lower().startswith("http"):
                    continue
                story.append(Paragraph(_inline_md(line.strip()), st["source"]))
            continue
        # body bullets vs paragraphs
        if line.lstrip().startswith(("- ", "* ")):
            content = line.lstrip()[2:]
            story.append(Paragraph(_inline_md(content), st["bullet"], bulletText="•"))
        else:
            story.append(Paragraph(_inline_md(line), st["body"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    log.info(f"[export] PDF written -> {out_path}")
    return out_path
