"""
report/prisma.py
================
PRISMA 2020 v2 flow-diagram support for Aletheon reports.

PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses)
is the standard methodology-reporting framework for systematic reviews. The
2020 update added a v2 flow diagram template that explicitly tracks evidence
from THREE source arms:
  - Databases (literature databases)
  - Registers (trial registers)
  - Other methods (citation chasing, hand searching, regulatory sources)

This module collects per-stage counts during the pipeline and renders the
flow as a markdown text-art diagram (faithful to PRISMA box labels, suitable
for both .md and PDF). A native ReportLab graphics version is in a separate
step.

Aletheon-to-PRISMA mapping:
  Databases  = europepmc, semanticscholar, pubchem, chembl, pharmgkb
  Registers  = clinicaltrials
  Other      = fda (OpenFDA labels), dailymed (SPLs), faers

This mapping is opinionated but defensible: trial registers and regulatory
sources are genuinely different evidence types than peer-reviewed databases,
and PRISMA reviewers expect to see them separated.
"""
from dataclasses import dataclass, field


# Source classification — used to bucket each fetched source into the right
# PRISMA arm. If you add a new data source, add it here too.
SOURCE_ARMS: dict[str, str] = {
    # Literature databases
    "europepmc":       "databases",
    "semanticscholar": "databases",
    "pubchem":         "databases",   # chemistry DB
    "chembl":          "databases",   # bioactivity DB
    "pharmgkb":        "databases",   # pharmacogenomic DB
    # Trial registers
    "clinicaltrials":  "registers",
    # Other / regulatory
    "fda":             "other",       # OpenFDA labels
    "dailymed":        "other",       # FDA SPL labels
    "faers":           "other",       # adverse-event reports
}


@dataclass
class PrismaCounts:
    """All counts tracked for a PRISMA 2020 v2 flow diagram, organized by arm.

    Filled in by core/graph.py nodes as the pipeline runs. Read by
    render_markdown / build_drawing for the report output.
    """
    # ---- Stage 1: Identification ----
    # Records identified per source (used to fill the per-arm 'identified' boxes)
    per_source: dict[str, int] = field(default_factory=dict)
    # Duplicates / off-drug removed BEFORE screening (the precision-filter drop)
    duplicates_removed: int = 0
    off_drug_removed:   int = 0
    # ---- Stage 2: Screening ----
    # Records screened = identified - (duplicates + off_drug). Set on combine.
    records_screened:   int = 0
    records_excluded_at_screening: int = 0  # 0 for now — we don't run a separate screen
    # ---- Stage 3: Eligibility / Retrieval ----
    # In Aletheon, every screened record is a 'report sought' and we retrieve
    # the abstract/body inline. So these two are equal in practice.
    reports_sought:     int = 0
    reports_retrieved:  int = 0
    reports_not_retrieved: int = 0
    # ---- Stage 4: Assessed for eligibility (extraction step) ----
    reports_assessed:   int = 0
    reports_excluded: dict[str, int] = field(default_factory=dict)
    # ---- Stage 5: Included in synthesis ----
    studies_included:   int = 0   # unique papers whose chunks appear in report
    reports_included:   int = 0   # same number for us (no study-vs-report distinction)
    chunks_in_report:   int = 0
    sections_with_evidence: int = 0

    # ---- Derived helpers ----
    def identified_in_arm(self, arm: str) -> int:
        """Total identified from sources in this arm ('databases', 'registers', 'other')."""
        return sum(n for src, n in self.per_source.items()
                   if SOURCE_ARMS.get(src) == arm)

    def sources_in_arm(self, arm: str) -> list[tuple[str, int]]:
        """Per-source breakdown for one arm, sorted by count desc."""
        return sorted(
            [(src, n) for src, n in self.per_source.items()
             if SOURCE_ARMS.get(src) == arm],
            key=lambda x: -x[1])

    @property
    def total_identified(self) -> int:
        return sum(self.per_source.values())

    @property
    def total_excluded_at_extraction(self) -> int:
        return sum(self.reports_excluded.values())


# ============================================================================
#  MARKDOWN RENDERING
# ============================================================================
#
# We render PRISMA as text-art inside a fenced code block. Reasoning:
#   - Stays readable in the raw .md file (a reviewer can grep / paste it)
#   - Renders correctly in the PDF inside a monospace block (no image deps)
#   - Layout is deterministic and testable — no fragile drawing code
#   - PDF-native graphics version comes later, AS A LAYER ON TOP

def _box(width: int, lines: list[str]) -> list[str]:
    """Render an ASCII box of `width` chars wrapping the given lines.

    Lines longer than `width-4` are wrapped at word boundaries. Returns the
    list of output strings (including the top/bottom borders)."""
    inner_w = width - 4  # 2 chars for borders on each side
    wrapped: list[str] = []
    for raw in lines:
        if not raw:
            wrapped.append("")
            continue
        # Manual word-wrap
        words = raw.split()
        cur = ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > inner_w:
                wrapped.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip() if cur else w
        if cur:
            wrapped.append(cur)
    out = ["┌" + "─" * (width - 2) + "┐"]
    for line in wrapped:
        out.append("│ " + line.ljust(inner_w) + " │")
    out.append("└" + "─" * (width - 2) + "┘")
    return out


def _side_by_side(left: list[str], right: list[str], gap: int = 2) -> list[str]:
    """Render two box-stacks side by side, aligned at the top."""
    max_left = max((len(line) for line in left), default=0)
    h = max(len(left), len(right))
    while len(left) < h:
        left.append("")
    while len(right) < h:
        right.append("")
    return [l.ljust(max_left) + " " * gap + r for l, r in zip(left, right)]


def _three_columns(a: list[str], b: list[str], c: list[str], gap: int = 2) -> list[str]:
    """Render three box-stacks side by side."""
    return _side_by_side(_side_by_side(a, b, gap), c, gap)


def _arrow_down(width: int, indent: int = 0) -> list[str]:
    """A 3-line down-arrow centered in a column of `width` chars."""
    pre = " " * indent
    mid = width // 2
    return [pre + " " * mid + "│",
            pre + " " * mid + "│",
            pre + " " * (mid - 1) + "▼"]


def render_markdown(counts: PrismaCounts) -> str:
    """Return a fenced markdown code block containing the PRISMA 2020 v2 flow.

    Structure (top to bottom):
      Identification:  [Databases]  [Registers]  [Other methods]
                         │             │             │
                         ▼             ▼             ▼
      Screening:       [Records screened ← duplicates / off-drug removed]
                                       │
                                       ▼
      Eligibility:     [Reports assessed ← excluded with reasons]
                                       │
                                       ▼
      Included:        [Studies included in synthesis]
    """
    # Build the three identification columns
    col_w = 32

    def _arm_lines(label: str, arm: str) -> list[str]:
        n = counts.identified_in_arm(arm)
        body = [
            f"Identification from {label}",
            "",
            f"Records identified (n = {n})",
        ]
        for src, src_n in counts.sources_in_arm(arm):
            body.append(f"  · {src}: {src_n}")
        if not counts.sources_in_arm(arm):
            body.append("  · (no sources in this arm)")
        return _box(col_w, body)

    db_box   = _arm_lines("databases", "databases")
    reg_box  = _arm_lines("registers", "registers")
    oth_box  = _arm_lines("other methods", "other")

    top_row = _three_columns(db_box, reg_box, oth_box)

    # Three down arrows under the columns
    arrow_row = []
    arrow_height = 3
    for i in range(arrow_height):
        line = ""
        for col in range(3):
            mid = col_w // 2
            offset = col * (col_w + 2)
            indent = offset + mid
            line = line.ljust(indent)
            line += "▼" if i == arrow_height - 1 else "│"
        arrow_row.append(line)

    # Stage 2: Screening (full-width box)
    full_w = col_w * 3 + 4  # three columns + two gaps
    screening_box = _box(full_w, [
        "SCREENING",
        "",
        f"Records screened          (n = {counts.records_screened})",
        f"Duplicates removed         (n = {counts.duplicates_removed})",
        f"Off-drug records removed   (n = {counts.off_drug_removed})",
        f"Records excluded at screen (n = {counts.records_excluded_at_screening})",
    ])

    # Stage 3: Eligibility
    elig_lines = [
        "ELIGIBILITY",
        "",
        f"Reports assessed for eligibility (n = {counts.reports_assessed})",
        "",
        "Reports excluded with reasons:",
    ]
    if counts.reports_excluded:
        for reason, n in sorted(counts.reports_excluded.items(), key=lambda x: -x[1]):
            elig_lines.append(f"  · {reason} (n = {n})")
    else:
        elig_lines.append("  · (none)")
    eligibility_box = _box(full_w, elig_lines)

    # Stage 4: Included
    included_box = _box(full_w, [
        "INCLUDED IN SYNTHESIS",
        "",
        f"Studies included in review        (n = {counts.studies_included})",
        f"Reports of included studies       (n = {counts.reports_included})",
        f"Unique evidence chunks cited      (n = {counts.chunks_in_report})",
        f"Report sections with evidence     (n = {counts.sections_with_evidence})",
    ])

    # Centered down-arrow between full-width boxes
    def _center_arrow(width: int) -> list[str]:
        mid = width // 2
        return [" " * mid + "│",
                " " * mid + "│",
                " " * (mid - 1) + "▼"]

    all_lines = []
    # First line is a marker the PDF renderer keys off of to substitute the
    # native graphics version. The marker is also human-readable in raw .md.
    all_lines.append("PRISMA 2020 v2 Flow Diagram")
    all_lines.append("=" * 78)
    all_lines.extend(top_row)
    all_lines.extend(arrow_row)
    all_lines.extend(screening_box)
    all_lines.extend(_center_arrow(full_w))
    all_lines.extend(eligibility_box)
    all_lines.extend(_center_arrow(full_w))
    all_lines.extend(included_box)

    return "```\n" + "\n".join(all_lines) + "\n```"


def build_prisma_block(counts: PrismaCounts) -> str:
    """Full markdown section, including a heading and explanatory blurb,
    ready to splice into the report."""
    return f"""## Methodology — PRISMA 2020 Flow Diagram

_This diagram follows the PRISMA 2020 v2 reporting standard for new
systematic reviews including searches of databases, registers, and other
sources. It shows how evidence flowed from initial source retrieval through
screening, eligibility assessment, and final inclusion in the synthesized
report. Numbers in parentheses indicate paper / record counts at each
stage._

{render_markdown(counts)}
"""


# ============================================================================
#  PDF RENDERING — native ReportLab graphics
# ============================================================================
#
# Called from md_to_pdf.py when the renderer hits the PRISMA marker.
# Returns a Drawing flowable that will be inserted in place of the monospace
# code block that would otherwise render. Crisper than the ASCII fallback.

def build_prisma_drawing(counts: PrismaCounts):
    """Build a ReportLab Drawing of the PRISMA 2020 v2 flow.

    Layout (top to bottom):
      Row 1: three identification boxes (databases | registers | other)
             with per-source breakdowns
      Row 2: full-width SCREENING box (records screened, dups removed, etc)
      Row 3: full-width ELIGIBILITY box (reports assessed, exclusions w/ reasons)
      Row 4: full-width INCLUDED box (studies + chunks + sections)

    Returns None if reportlab.graphics isn't importable (caller should fall
    back to the ASCII markdown block).
    """
    try:
        from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
        from reportlab.lib import colors
    except ImportError:
        return None

    # ---- Geometry ----
    W = 540            # full diagram width — fits letter page minus margins
    col_w = 170        # identification-column width
    col_gap = 15
    box_h = 95         # identification-box height
    full_w = W         # full-width screening/eligibility/included boxes
    arrow_h = 14       # shorter arrows so the whole diagram fits on one page
    title_h = 14       # tighter title strip

    # Eligibility height: each exclusion reason becomes 1 or 2 lines (wraps if
    # the reason text is long). We measure how many lines we actually need
    # before sizing the box, so the bottom row doesn't fall outside the frame.
    max_chars_per_line = 76   # what fits comfortably at fontSize=7.5
    elig_reasons = sorted(counts.reports_excluded.items(),
                          key=lambda x: -x[1]) if counts.reports_excluded else []
    elig_reason_lines = []   # pre-wrapped (reason, n) → list of strings
    for reason, n in elig_reasons:
        full_text = f"  · {reason} (n = {n})"
        if len(full_text) <= max_chars_per_line:
            elig_reason_lines.append(full_text)
        else:
            # wrap onto a continuation line — indent the continuation
            split_at = max_chars_per_line - 4
            elig_reason_lines.append(full_text[:split_at])
            elig_reason_lines.append("    " + full_text[split_at:])
    if not elig_reason_lines:
        elig_reason_lines.append("  · (none recorded)")
    # eligibility body has 4 fixed lines + the wrapped reasons
    elig_body_lines = 4 + len(elig_reason_lines)
    elig_h = title_h + 6 + (elig_body_lines * 10) + 6

    # Stage heights
    screening_h = 70
    included_h = 70

    # Total height = identification + arrow + screening + arrow + eligibility + arrow + included + small padding
    H = box_h + arrow_h + screening_h + arrow_h + elig_h + arrow_h + included_h + 6

    d = Drawing(W, H)

    # ---- Colors ----
    BLUE      = colors.HexColor("#1e40af")
    BLUE_BG   = colors.HexColor("#eff6ff")
    BORDER    = colors.HexColor("#cbd5e1")
    GRAY      = colors.HexColor("#374151")
    GRAY_DIM  = colors.HexColor("#6b7280")

    def _box(x, y, w, h, title, count_str, body_lines, title_color=BLUE):
        d.add(Rect(x, y, w, h,
                   fillColor=BLUE_BG, strokeColor=BORDER, strokeWidth=0.6))
        d.add(Rect(x, y + h - title_h, w, title_h,
                   fillColor=title_color, strokeColor=title_color, strokeWidth=0))
        d.add(String(x + 6, y + h - title_h + 4, title,
                     fillColor=colors.white,
                     fontSize=8, fontName="Helvetica-Bold"))
        if count_str:
            d.add(String(x + w - 6, y + h - title_h + 4, count_str,
                         fillColor=colors.white,
                         fontSize=8, fontName="Helvetica-Bold",
                         textAnchor="end"))
        # body lines, top-down from just under the title
        for i, line in enumerate(body_lines):
            d.add(String(x + 6, y + h - title_h - 10 - i * 10, line,
                         fillColor=GRAY, fontSize=7.5, fontName="Helvetica"))

    def _down_arrow(x_center, y_top, height):
        d.add(Line(x_center, y_top, x_center, y_top - height + 5,
                   strokeColor=GRAY_DIM, strokeWidth=0.8))
        d.add(Polygon([x_center - 3.5, y_top - height + 7,
                       x_center + 3.5, y_top - height + 7,
                       x_center, y_top - height + 1],
                      fillColor=GRAY_DIM, strokeColor=GRAY_DIM,
                      strokeWidth=0.5))

    # ---- Row 1: three identification columns ----
    y_top = H - 2
    x1 = (W - 3 * col_w - 2 * col_gap) / 2
    x2 = x1 + col_w + col_gap
    x3 = x2 + col_w + col_gap

    def _arm_body(arm: str) -> list[str]:
        sources = counts.sources_in_arm(arm)
        lines = []
        for src, n in sources[:6]:
            lines.append(f"  · {src}: {n}")
        if not sources:
            lines.append("  · (no sources in this arm)")
        if len(sources) > 6:
            lines.append(f"  · (+{len(sources) - 6} more)")
        return lines

    # Titles fit in 170px wide boxes at fontSize=8 — keep them short.
    _box(x1, y_top - box_h, col_w, box_h,
         "Databases",
         f"n = {counts.identified_in_arm('databases')}",
         _arm_body("databases"))
    _box(x2, y_top - box_h, col_w, box_h,
         "Registers",
         f"n = {counts.identified_in_arm('registers')}",
         _arm_body("registers"))
    _box(x3, y_top - box_h, col_w, box_h,
         "Other Methods",
         f"n = {counts.identified_in_arm('other')}",
         _arm_body("other"))

    arrows_top_y = y_top - box_h
    _down_arrow(x1 + col_w / 2, arrows_top_y, arrow_h)
    _down_arrow(x2 + col_w / 2, arrows_top_y, arrow_h)
    _down_arrow(x3 + col_w / 2, arrows_top_y, arrow_h)

    # ---- Row 2: screening (full width) ----
    y_screening = arrows_top_y - arrow_h - screening_h
    _box(0, y_screening, full_w, screening_h,
         "SCREENING",
         f"n = {counts.records_screened}",
         [f"Records identified across all sources    (n = {counts.total_identified})",
          f"Duplicates removed                        (n = {counts.duplicates_removed})",
          f"Off-drug records removed                  (n = {counts.off_drug_removed})",
          f"Records screened                          (n = {counts.records_screened})"])
    _down_arrow(full_w / 2, y_screening, arrow_h)

    # ---- Row 3: eligibility (variable height, wraps long exclusion lines) ----
    y_elig = y_screening - arrow_h - elig_h
    body = [f"Reports sought for retrieval              (n = {counts.reports_sought})",
            f"Reports assessed for eligibility          (n = {counts.reports_assessed})",
            "",
            "Reports excluded with reasons:"]
    body.extend(elig_reason_lines)
    _box(0, y_elig, full_w, elig_h,
         "ELIGIBILITY (extraction)",
         f"n = {counts.reports_assessed}",
         body)
    _down_arrow(full_w / 2, y_elig, arrow_h)

    # ---- Row 4: included (full width) ----
    y_inc = y_elig - arrow_h - included_h
    _box(0, y_inc, full_w, included_h,
         "INCLUDED IN SYNTHESIS",
         f"n = {counts.studies_included}",
         [f"Studies included in review                (n = {counts.studies_included})",
          f"Reports of included studies               (n = {counts.reports_included})",
          f"Unique evidence chunks cited              (n = {counts.chunks_in_report})",
          f"Report sections with evidence             (n = {counts.sections_with_evidence})"])

    return d