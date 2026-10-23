"""Render the work-plan markdown to a formatted PDF (reportlab, base-14 fonts only)."""
import re, sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether, Preformatted)

SRC, OUT, DOCTITLE = sys.argv[1], sys.argv[2], sys.argv[3]
INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#6b6b6b")
RULE = colors.HexColor("#d9d9d9")
BAND = colors.HexColor("#f2f4f5")

S = {
 "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, leading=25,
                      textColor=INK, spaceAfter=2),
 "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=9, leading=13,
                       textColor=MUTED, spaceAfter=14),
 "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13.5, leading=17,
                      textColor=INK, spaceBefore=16, spaceAfter=6),
 "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                      textColor=INK, spaceBefore=10, spaceAfter=4),
 "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=14,
                        textColor=INK, spaceAfter=6, alignment=TA_LEFT),
 "quote": ParagraphStyle("quote", fontName="Helvetica-Oblique", fontSize=9, leading=13,
                         textColor=MUTED, leftIndent=8, spaceAfter=8,
                         borderPadding=(6, 6, 6, 8)),
 "li": ParagraphStyle("li", fontName="Helvetica", fontSize=9.5, leading=13.5,
                      textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=2.5),
 "li2": ParagraphStyle("li2", fontName="Helvetica", fontSize=9.5, leading=13.5,
                       textColor=INK, leftIndent=26, bulletIndent=16, spaceAfter=2.5),
 "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.8, leading=10.5,
                        textColor=INK),
 "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=7.8, leading=10.5,
                         textColor=INK),
 "code": ParagraphStyle("code", fontName="Courier", fontSize=7.2, leading=9.2,
                        textColor=INK),
}

def inline(t):
    t = (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r'<font face="Courier" size="8.6">\1</font>', t)
    return t

lines = open(SRC).read().split("\n")
flow, i = [], 0
# strip the H1 + byline, rendered as a title block instead
title = lines[0].lstrip("# ").strip()
flow += [Paragraph(inline(title), S["h1"])]

while i < len(lines):
    ln = lines[i]
    st = ln.strip()

    if st.startswith("```"):                                    # fenced block
        i += 1; buf = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            buf.append(lines[i]); i += 1
        i += 1
        flow.append(Table([[Preformatted("\n".join(buf), S["code"])]],
                          colWidths=[165*mm],
                          style=TableStyle([("BACKGROUND", (0,0), (-1,-1), BAND),
                                            ("BOX", (0,0), (-1,-1), 0.4, RULE),
                                            ("LEFTPADDING", (0,0), (-1,-1), 8),
                                            ("TOPPADDING", (0,0), (-1,-1), 7),
                                            ("BOTTOMPADDING", (0,0), (-1,-1), 7)])))
        flow.append(Spacer(1, 7)); continue

    if st.startswith("|"):                                      # table
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(set(c) <= set(":- ") for c in cells):
                rows.append(cells)
            i += 1
        head, body = rows[0], rows[1:]
        n = len(head)
        w = [40*mm, 52*mm, 58*mm, 25*mm] if n == 4 else [165*mm/n]*n
        data = [[Paragraph(inline(c), S["cellh"]) for c in head]] + \
               [[Paragraph(inline(c), S["cell"]) for c in r] for r in body]
        tbl = Table(data, colWidths=w, repeatRows=1, style=TableStyle([
            ("BACKGROUND", (0,0), (-1,0), BAND),
            ("LINEBELOW", (0,0), (-1,0), 0.6, MUTED),
            ("INNERGRID", (0,1), (-1,-1), 0.3, RULE),
            ("BOX", (0,0), (-1,-1), 0.4, RULE),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4)]))
        # keep short tables whole so a single row never orphans onto the next page
        flow.append(KeepTogether(tbl) if len(data) <= 5 else tbl)
        flow.append(Spacer(1, 8)); continue

    if st.startswith("### "):
        flow.append(Paragraph(inline(st[4:]), S["h3"]))
    elif st.startswith("## "):
        flow.append(Paragraph(inline(st[3:]), S["h2"]))
        flow.append(Table([[""]], colWidths=[165*mm], rowHeights=[0.6],
                          style=TableStyle([("BACKGROUND", (0,0), (-1,-1), RULE)])))
        flow.append(Spacer(1, 7))
    elif st.startswith("# "):
        pass
    elif st.startswith("> "):
        body = [st[2:]]
        while i+1 < len(lines) and lines[i+1].strip().startswith(">"):
            i += 1; body.append(lines[i].strip().lstrip(">").strip())
        flow.append(Table([[Paragraph(inline(" ".join(body)), S["quote"])]],
                          colWidths=[165*mm],
                          style=TableStyle([("BACKGROUND", (0,0), (-1,-1), BAND),
                                            ("LINEBEFORE", (0,0), (0,-1), 2, MUTED),
                                            ("LEFTPADDING", (0,0), (-1,-1), 8),
                                            ("TOPPADDING", (0,0), (-1,-1), 6),
                                            ("BOTTOMPADDING", (0,0), (-1,-1), 6)])))
        flow.append(Spacer(1, 8))
    elif re.match(r"^\s*-\s", ln):
        indent = len(ln) - len(ln.lstrip())
        txt = st[2:]
        while i+1 < len(lines) and lines[i+1].strip() and \
              not re.match(r"^\s*-\s", lines[i+1]) and \
              not lines[i+1].strip().startswith(("#", "|", ">", "```")) and \
              (len(lines[i+1]) - len(lines[i+1].lstrip())) > indent:
            i += 1; txt += " " + lines[i].strip()
        flow.append(Paragraph(inline(txt), S["li2" if indent >= 2 else "li"],
                              bulletText="\u2013" if indent >= 2 else "\u2022"))
    elif st == "":
        pass
    else:                                                        # paragraph
        buf = [st]
        while i+1 < len(lines) and lines[i+1].strip() and \
              not lines[i+1].strip().startswith(("#", "|", "-", ">", "```")):
            i += 1; buf.append(lines[i].strip())
        para = " ".join(buf)
        if flow and isinstance(flow[-1], Paragraph) and flow[-1].style is S["h1"]:
            flow.append(Paragraph(inline(para), S["sub"]))
        else:
            flow.append(Paragraph(inline(para), S["body"]))
    i += 1

def decorate(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 7.5); canv.setFillColor(MUTED)
    canv.drawString(22*mm, 12*mm, DOCTITLE)
    canv.drawRightString(A4[0]-22*mm, 12*mm, f"Page {canv.getPageNumber()}")
    canv.setStrokeColor(RULE); canv.setLineWidth(0.4)
    canv.line(22*mm, 15.5*mm, A4[0]-22*mm, 15.5*mm)
    canv.restoreState()

doc = BaseDocTemplate(OUT, pagesize=A4, title=DOCTITLE,
                      author="Water Quality Prediction team",
                      leftMargin=22*mm, rightMargin=22*mm,
                      topMargin=18*mm, bottomMargin=20*mm)
doc.addPageTemplates([PageTemplate(id="main",
    frames=[Frame(22*mm, 20*mm, 165*mm, A4[1]-38*mm, id="f")], onPage=decorate)])
doc.build(flow)
print("wrote", OUT)

# Usage (from the repository root):
#
#   pip install reportlab
#   python docs/render_workplan.py \
#       docs/phase1_work_plan.md \
#       "Water Quality Prediction Phase 1 Work Plan.pdf" \
#       "Water Quality Prediction: Phase 1 Work Plan (rev. DD Mon YYYY)"
#
# Edit docs/phase1_work_plan.md, regenerate, and share the PDF. Never edit the PDF
# directly - the markdown is the source of truth.
