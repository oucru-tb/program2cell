"""Talk slides: PALM as a graphical model (two-group design, and a single group such as TB patients only).

Left: the generative model PALM fits (it gives the cell mix pi). Right: the calibrated estimator that gives the
reported per-cell change (reallocation -> calibration on atlas donors -> unmixing across patients), as in the
frozen method (docs/model_dag.md). Writes docs/figures/talk/2_2c_graphical_model.svg and
2_2d_graphical_model_tb_only.svg. Usage: python talk_graphical_models.py
"""
import os

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
OUT = os.path.join(ROOT, "docs", "figures", "talk")

BLUE, BLUE_F = "#2563eb", "#dbeafe"
ORANGE, ORANGE_F = "#ea580c", "#ffedd5"
GREY, GREY_F = "#6b7280", "#f1f3f5"
GREEN, GREEN_F = "#16a34a", "#dcfce7"
PURPLE, PURPLE_F = "#7c3aed", "#f5f3ff"
DARK = "#2b2d42"
MUTED = "#5c6670"


def text(x, y, s, size=16, fill=DARK, weight="normal", anchor="start", style=""):
    return ('<text x="%g" y="%g" font-size="%g" fill="%s" font-weight="%s" text-anchor="%s"%s>%s</text>'
            % (x, y, size, fill, weight, anchor, (' font-style="%s"' % style) if style else "", s))


def circle(x, y, r, fill, stroke, label, sub=None, sub2=None, color=None, lab_size=30):
    color = color or stroke
    out = ['<circle cx="%g" cy="%g" r="%g" fill="%s" stroke="%s" stroke-width="3"/>' % (x, y, r, fill, stroke),
           text(x, y + lab_size * 0.33, label, lab_size, color if fill != DARK else "#ffffff", "bold", "middle")]
    if sub:
        out.append(text(x, y + r + 22, sub, 15, color, "bold", "middle"))
    if sub2:
        out.append(text(x, y + r + 40, sub2, 13, color, "normal", "middle"))
    return "\n".join(out)


def box(x, y, w, h, fill, stroke, label, sub=None, sub2=None, color=None, dash=None, size=24):
    color = color or stroke
    d = ' stroke-dasharray="%s"' % dash if dash else ""
    out = ['<rect x="%g" y="%g" width="%g" height="%g" rx="10" fill="%s" stroke="%s" stroke-width="2.5"%s/>'
           % (x, y, w, h, fill, stroke, d), text(x + w / 2, y + h / 2 - (8 if sub else -8), label, size, color, "bold", "middle")]
    if sub:
        out.append(text(x + w / 2, y + h / 2 + 16, sub, 13.5, color, "normal", "middle"))
    if sub2:
        out.append(text(x + w / 2, y + h / 2 + 33, sub2, 12.5, GREY, "normal", "middle"))
    return "\n".join(out)


def hexagon(x, y, r, label, sub):
    pts = " ".join("%g,%g" % (x + r * c, y + r * s) for c, s in
                   ((1, 0), (0.5, 0.866), (-0.5, 0.866), (-1, 0), (-0.5, -0.866), (0.5, -0.866)))
    return "\n".join(['<polygon points="%s" fill="#fff7e0" stroke="#b07d00" stroke-width="2.5"/>' % pts,
                      text(x, y + 9, label, 26, "#8a6100", "bold", "middle"),
                      text(x - r - 10, y + 6, sub, 14, "#8a6100", "bold", "end")])


def plate(x, y, w, h, label, lx=None, ly=None, dash="6 4", color="#c4c8cc", width=2):
    return ('<rect x="%g" y="%g" width="%g" height="%g" rx="14" fill="none" stroke="%s" stroke-width="%g" '
            'stroke-dasharray="%s"/>' % (x, y, w, h, color, width, dash) +
            text(lx if lx is not None else x + w - 12, ly if ly is not None else y + 24, label, 15, MUTED, "bold", "end"))


def arrow(d, color="#4b5563", dash=None, marker="arr", width=2.5):
    return '<path d="%s" fill="none" stroke="%s" stroke-width="%g"%s marker-end="url(#%s)"/>' % (
        d, color, width, (' stroke-dasharray="%s"' % dash) if dash else "", marker)


def slide(tb_only):
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" width="1600" height="900" '
         'font-family="Helvetica, Arial, sans-serif">', "<defs>"]
    for name, col in (("arr", "#4b5563"), ("arrO", ORANGE), ("arrP", PURPLE), ("arrG", "#9aa0a6")):
        s.append('<marker id="%s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
                 'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>' % (name, col))
    s.append("</defs>")
    s.append('<rect width="1600" height="900" fill="#ffffff"/>')
    if tb_only:
        s.append(text(40, 56, "PALM with TB patients only (no comparison group)", 32, DARK, "bold"))
        s.append(text(40, 88, "Same model of the mix; the change is estimated along a clinical measure z instead of TB vs healthy", 18, MUTED))
    else:
        s.append(text(40, 56, "PALM as a graphical model", 32, DARK, "bold"))
        s.append(text(40, 88, "A model of the cell mix (left) and a calibrated estimate of each cell type's change (right)", 18, MUTED))

    # ---------------- left: generative model ----------------
    s.append(text(40, 140, "1  The model PALM fits", 20, DARK, "bold"))
    s.append(text(40, 162, "gives each patient's cell mix π, with uncertainty", 14.5, MUTED))
    s.append(text(48, 222, "shared by", 13.5, MUTED, "bold"))
    s.append(text(48, 239, "all patients", 13.5, MUTED, "bold"))
    if tb_only:
        s.append(circle(115, 320, 40, BLUE_F, BLUE, "a", "average", "cell mix", lab_size=28))
    else:
        s.append(circle(115, 320, 40, BLUE_F, BLUE, "γ", "disease", "shifts the mix", lab_size=28))
    s.append(circle(115, 500, 40, ORANGE_F, ORANGE, "κ", "average program", "activity", lab_size=28))

    s.append(plate(220, 190, 670, 520, "each patient i", dash="0", color="#9aa0a6", width=2.5))
    s.append(plate(370, 255, 410, 445, "each cell type c", ly=279))
    s.append(plate(565, 290, 205, 175, "each program p", ly=311, color="#d9dcdf", dash="3 4"))
    s.append(plate(395, 520, 485, 350, "each gene g", ly=860))

    if tb_only:
        s.append(box(240, 215, 125, 66, PURPLE_F, PURPLE, "z", "clinical measure", dash="7 4", size=22))
    else:
        s.append(box(240, 215, 125, 66, "#ffffff", DARK, "D", "TB or healthy", color=DARK, size=22))
    s.append(circle(460, 375, 46, BLUE_F, BLUE, "π", "fraction", "how many cells"))
    s.append(circle(640, 375, 42, ORANGE_F, ORANGE, "δ", None))
    s.append(text(690, 372, "activity", 15, ORANGE, "bold"))
    s.append(text(690, 390, "helps the fit", 13, ORANGE))
    s.append(hexagon(560, 600, 40, "E", "cell profile"))
    s.append(circle(820, 600, 42, DARK, DARK, "Y", None))
    s.append(text(820, 668, "bulk counts", 15, DARK, "bold", "middle"))
    s.append(text(820, 686, "(measured)", 13, MUTED, "normal", "middle"))
    s.append(box(420, 765, 160, 80, GREY_F, GREY, "B<tspan font-size=\"14\" dy=\"5\">cg</tspan>", "reference profile",
                 "atlas average", color="#4b5563"))
    s.append(box(615, 765, 160, 80, GREEN_F, GREEN, "G<tspan font-size=\"14\" dy=\"5\">pg</tspan>", "program genes",
                 "PROGENy", color=GREEN))

    a = []
    if tb_only:
        a.append(arrow("M155,325 L412,368"))
    else:
        a.append(arrow("M155,325 L412,368"))
        a.append(arrow("M330,281 L422,340"))
        a.append(arrow("M350,232 Q560,200 630,336"))
    a.append(arrow("M155,500 L596,386"))
    a.append(arrow("M628,416 L582,564"))          # delta -> E
    a.append(arrow("M500,780 L540,640"))          # B -> E
    a.append(arrow("M680,765 L585,640"))          # G -> E
    a.append(arrow("M600,600 L776,600"))          # E -> Y
    a.append(arrow("M488,412 Q640,530 778,588"))  # pi -> Y
    s += a
    s.append(text(40, 620, "Y ≈ Σ<tspan font-size=\"12\" dy=\"4\">c</tspan><tspan dy=\"-4\"> π · E</tspan>", 17, DARK))
    s.append(text(40, 643, "E = B · exp(δ × G)", 17, DARK))
    s.append(text(40, 668, "noisy counts (neg. binomial)", 13, MUTED))

    # ---------------- right: calibrated estimator ----------------
    X0, W = 935, 625
    s.append(text(X0, 140, "2  Estimating each cell type's change", 20, DARK, "bold"))
    s.append(text(X0, 162, "the reported answer comes from here, not from δ in the model", 14.5, MUTED))
    steps = [("Split each patient's reads by π", "→ program activity per cell type (still has spillover)"),
             ("Calibrate on atlas donors with a known answer", "corrects the bias, sets honest 90% ranges"),
             ("Unmix across patients" + (" along z" if tb_only else ""),
              "spillover map from π and B; " + ("z = clinical measure as a continuous contrast" if tb_only
                                                 else "contrast = TB vs healthy"))]
    ys = [185, 275, 365]
    for k, ((t1, t2), y) in enumerate(zip(steps, ys)):
        fill = GREY_F if k < 2 else ORANGE_F
        s.append('<rect x="%g" y="%g" width="%g" height="70" rx="10" fill="%s"/>' % (X0, y, W, fill))
        s.append(text(X0 + 18, y + 29, "%d  %s" % (k + 1, t1), 17, ORANGE if k == 2 else DARK, "bold"))
        s.append(text(X0 + 18, y + 54, t2, 14.5, "#333"))
        if k < 2:
            s.append(arrow("M%g,%g L%g,%g" % (X0 + W / 2, y + 70, X0 + W / 2, y + 88), "#9aa0a6", marker="arrG", width=2))
    # inputs into step 1: pi and Y
    s.append(arrow("M478,330 C520,120 860,110 930,205", BLUE, dash="7 5", marker="arr", width=2.2))
    s.append(text(700, 128, "π (posterior draws)", 14, BLUE, "bold", "middle"))
    s.append(arrow("M858,585 C895,520 892,300 930,236", "#4b5563", dash="7 5", width=2.2))
    if tb_only:
        s.append(arrow("M365,250 H 905 Q 918,250 918,280 V 388 Q 918,400 930,400", PURPLE, dash="7 5", marker="arrP", width=2.2))
    else:
        s.append(arrow("M365,250 H 905 Q 918,250 918,280 V 388 Q 918,400 930,400", "#4b5563", dash="7 5", width=2.2))
    # output
    s.append(arrow("M%g,435 L%g,470" % (X0 + 170, X0 + 170), ORANGE, marker="arrO"))
    if tb_only:
        s.append(circle(X0 + 170, 522, 48, ORANGE_F, ORANGE, "Δ", None, lab_size=30))
        s.append(text(X0 + 235, 510, "change over the z range", 17, ORANGE, "bold"))
        s.append(text(X0 + 235, 533, "per cell type and program, ± 90% range", 14.5, "#333"))
    else:
        s.append(circle(X0 + 170, 522, 48, ORANGE_F, ORANGE, "β", None, lab_size=30))
        s.append(text(X0 + 235, 510, "per-cell change", 17, ORANGE, "bold"))
        s.append(text(X0 + 235, 533, "per cell type and program, ± 90% range", 14.5, "#333"))
        s.append('<rect x="%g" y="588" width="%g" height="56" rx="10" fill="none" stroke="%s" stroke-width="2" '
                 'stroke-dasharray="6 4"/>' % (X0 + 60, W - 60, GREY))
        s.append(text(X0 + 80, 612, "contributions (from π and β)", 15.5, DARK, "bold"))
        s.append(text(X0 + 80, 633, "each cell type's share of the bulk change: more cells / cells changed", 13.5, "#333"))
        s.append(arrow("M%g,570 L%g,586" % (X0 + 170, X0 + 170), "#9aa0a6", marker="arrG", width=2))
    # atlas
    ay = 700 if not tb_only else 610
    s.append('<rect x="%g" y="%g" width="270" height="62" rx="10" fill="%s" stroke="%s" stroke-width="2.5"/>'
             % (X0 + 330, ay, GREY_F, GREY))
    s.append(text(X0 + 465, ay + 26, "single-cell atlas", 16, "#4b5563", "bold", "middle"))
    s.append(text(X0 + 465, ay + 47, "healthy donors, known answer", 13, GREY, "normal", "middle"))
    if not tb_only:                         # atlas -> B (the TB slide states it in the label)
        s.append(arrow("M%g,%g C%g,%g %g,%g 500,848" % (X0 + 330, ay + 45, X0 + 150, 905, 640, 905), "#9aa0a6",
                       dash="6 4", marker="arrG", width=2))
    s.append(arrow("M%g,%g C%g,%g %g,%g %g,347" % (X0 + 590, ay, X0 + 660, ay - 150, X0 + 660, 320, X0 + W - 2),
                   "#9aa0a6", dash="6 4", marker="arrG", width=2))
    s.append(text(X0 + 340, ay - 12, "→ B, and → calibration (step 2)", 13, GREY))

    if tb_only:
        X0 += 60; W -= 60
        s.append('<rect x="%g" y="700" width="%g" height="150" rx="12" fill="#fff7ed" stroke="%s" stroke-width="2"/>'
                 % (X0, W, ORANGE))
        s.append(text(X0 + 18, 730, "Not yet validated (benchmarking.md 0.8)", 17, "#c2410c", "bold"))
        s.append(text(X0 + 18, 757, "• correlating each patient's activity with z: spillover gives false", 14.5, "#333"))
        s.append(text(X0 + 32, 777, "associations in non-responding cell types (63% in the first test)", 14.5, "#333"))
        s.append(text(X0 + 18, 803, "• unmixing along z cuts this to 21% (target ≤ 10%); 10-split run queued", 14.5, "#333"))
        s.append(text(X0 + 18, 829, "• z is not part of the model: it enters only in step 3", 14.5, "#333"))
    else:
        pass
    lx = 40
    s.append(text(lx, 728, "Legend", 15, DARK, "bold"))
    s.append('<circle cx="%g" cy="750" r="10" fill="%s" stroke="%s" stroke-width="2"/>' % (lx + 10, BLUE_F, BLUE))
    s.append('<circle cx="%g" cy="750" r="10" fill="%s" stroke="%s" stroke-width="2"/>' % (lx + 34, ORANGE_F, ORANGE))
    s.append(text(lx + 52, 755, "estimated: cell mix / program activity", 13.5, "#333"))
    s.append('<rect x="%g" y="767" width="20" height="20" rx="4" fill="%s" stroke="%s" stroke-width="2"/>' % (lx, GREY_F, GREY))
    s.append(text(lx + 28, 782, "known input", 13.5, "#333"))
    s.append('<circle cx="%g" cy="804" r="10" fill="%s"/>' % (lx + 10, DARK))
    s.append(text(lx + 28, 809, "measured", 13.5, "#333"))
    s.append('<polygon points="%s" fill="#fff7e0" stroke="#b07d00" stroke-width="2"/>' % " ".join(
        "%g,%g" % (lx + 10 + 10 * c, 831 + 10 * sn) for c, sn in ((1, 0), (0.5, 0.866), (-0.5, 0.866), (-1, 0), (-0.5, -0.866), (0.5, -0.866))))
    s.append(text(lx + 28, 836, "computed", 13.5, "#333"))
    s.append(text(lx, 862, "box = repeated for each", 12.5, MUTED))
    s.append(text(lx, 878, "patient / cell type / program / gene", 12.5, MUTED))
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    for name, tb in (("2_2c_graphical_model", False), ("2_2d_graphical_model_tb_only", True)):
        with open(os.path.join(OUT, name + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(slide(tb))
        print("wrote", name)
