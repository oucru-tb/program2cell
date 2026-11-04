"""Bayes-style plate diagram of PALM's generative model (stage A, freeze v2) -> docs/figures/talk/2_2e_graphical_model_bayes.svg

Conventions (Bishop / Koller): open circle = latent random variable, shaded = observed, double circle = deterministic,
small filled dot = fixed input, rectangle = plate (repeat over the index in its corner). Overlapping plates mean the
node carries both indices. Non-centred parameterisation and covariates X are left out for readability (method_spec).
"""
import math, os

OUT = os.path.join("C:/Users/duy.pham/Documents/dev/program2cell", "docs", "figures", "talk")
DARK, MUTED, BLUE, BLUE_F, ORANGE, ORANGE_F = "#2b2d42", "#5c6670", "#2563eb", "#dbeafe", "#ea580c", "#ffedd5"
OBS_F, GREY = "#cfd4da", "#4b5563"
R = 31
s = []


def text(x, y, t, size=14, color=DARK, weight="normal", anchor="middle", italic=False):
    st = ' font-style="italic"' if italic else ""
    s.append('<text x="%g" y="%g" font-size="%g" fill="%s" font-weight="%s" text-anchor="%s"%s>%s</text>'
             % (x, y, size, color, weight, anchor, st, t))


def sym(x, y, base, sub, color, size=22):
    s.append('<text x="%g" y="%g" font-size="%g" font-weight="bold" fill="%s" text-anchor="middle">%s'
             '<tspan font-size="%g" dy="5">%s</tspan></text>' % (x, y + 7, size, color, base, size * 0.6, sub))


def node(x, y, base, sub, kind, color, fill, label=None, label2=None):
    if kind == "obs":
        fill = OBS_F
    s.append('<circle cx="%g" cy="%g" r="%g" fill="%s" stroke="%s" stroke-width="2.5"/>' % (x, y, R, fill, color))
    if kind == "det":
        s.append('<circle cx="%g" cy="%g" r="%g" fill="none" stroke="%s" stroke-width="1.8"/>' % (x, y, R - 5, color))
    sym(x, y, base, sub, color)
    if label:
        text(x, y + R + 17, label, 12.5, color)
    if label2:
        text(x, y + R + 32, label2, 12.5, color)
    return (x, y, R)


def dot(x, y, base, sub, label, color=GREY, below=True):
    s.append('<circle cx="%g" cy="%g" r="6" fill="%s"/>' % (x, y, color))
    dy = 26 if below else -14
    if below:
        s.append('<text x="%g" y="%g" font-size="18" font-weight="bold" fill="%s" text-anchor="middle">%s'
                 '<tspan font-size="11" dy="4">%s</tspan></text>' % (x, y + dy, color, base, sub))
    if label:
        text(x, y + dy + 17, label, 12, MUTED)
    return (x, y, 7)


def plate(x, y, w, h, label, lx, ly, anchor="end", color="#8a9099"):
    s.append('<rect x="%g" y="%g" width="%g" height="%g" rx="10" fill="none" stroke="%s" stroke-width="2"/>'
             % (x, y, w, h, color))
    text(lx, ly, label, 14, MUTED, "bold", anchor)


def edge(a, b, bend=None):
    (x1, y1, r1), (x2, y2, r2) = a, b
    if bend is None:
        d = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        s.append('<path d="M%g,%g L%g,%g"/>' % (x1 + ux * r1, y1 + uy * r1, x2 - ux * (r2 + 3), y2 - uy * (r2 + 3)))
    else:
        cx, cy = bend
        d1 = math.hypot(cx - x1, cy - y1); d2 = math.hypot(x2 - cx, y2 - cy)
        sx, sy = x1 + (cx - x1) / d1 * r1, y1 + (cy - y1) / d1 * r1
        ex, ey = x2 - (x2 - cx) / d2 * (r2 + 3), y2 - (y2 - cy) / d2 * (r2 + 3)
        s.append('<path d="M%g,%g Q%g,%g %g,%g"/>' % (sx, sy, cx, cy, ex, ey))


s.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" width="1600" height="900" '
         'font-family="Helvetica, Arial, sans-serif">')
s.append('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker></defs>' % GREY)
s.append('<rect width="1600" height="900" fill="#ffffff"/>')
text(40, 52, "PALM as a Bayesian network", 32, DARK, "bold", "start")
text(40, 84, "The generative model fitted in stage A, in plate notation: each node is drawn given its parents",
     18, MUTED, anchor="start")

# plates: C (cell types), P (programs), N (patients), G (genes)
plate(140, 200, 660, 410, "C cell types", 152, 222, "start")
plate(140, 450, 840, 270, "P programs", 970, 710)
plate(410, 160, 740, 585, "N patients", 420, 739, "start")
plate(990, 330, 190, 500, "G genes", 1170, 820)

# nodes
s_eta = node(270, 125, "s", "η", "lat", BLUE, BLUE_F, None)
text(318, 120, "mix noise", 12.5, BLUE, anchor="start")
tau = node(78, 530, "τ", "", "lat", ORANGE, ORANGE_F, "sparsity", "(horseshoe)")
a = node(210, 270, "a", "c", "lat", BLUE, BLUE_F, "baseline mix")
gam = node(330, 340, "γ", "c", "lat", BLUE, BLUE_F, "disease shifts")
sig = node(210, 395, "σ", "c", "lat", ORANGE, ORANGE_F, None)
text(250, 400, "spread", 12.5, ORANGE, anchor="start")
kap = node(210, 505, "κ", "cp", "lat", ORANGE, ORANGE_F, None)
text(210, 556, "baseline", 12.5, ORANGE)
bet = node(330, 565, "β", "cp", "lat", ORANGE, ORANGE_F, None)
text(370, 586, "disease effect", 12.5, ORANGE, anchor="start")
lam = node(300, 668, "λ", "p", "lat", ORANGE, ORANGE_F, None)
text(300, 713, "shared-factor scale", 12.5, ORANGE)
eta = node(500, 300, "η", "ic", "lat", BLUE, BLUE_F, "mix logit")
pi = node(660, 300, "π", "ic", "det", BLUE, BLUE_F, "cell mix")
D = node(880, 250, "D", "i", "obs", DARK, OBS_F, None)
text(918, 255, "TB or healthy", 12.5, DARK, anchor="start")
delta = node(600, 530, "δ", "icp", "lat", ORANGE, ORANGE_F, "program activity")
u = node(890, 580, "u", "ip", "lat", ORANGE, ORANGE_F, "patient factor", "(all types)")
mu = node(1085, 430, "μ", "ig", "det", DARK, "#ffffff", None)
text(1093, 476, "expected", 12, DARK, anchor="start"); text(1093, 490, "count", 12, DARK, anchor="start")
Y = node(1085, 620, "Y", "ig", "obs", DARK, OBS_F, None)
text(1093, 666, "bulk", 12, DARK, anchor="start"); text(1093, 680, "counts", 12, DARK, anchor="start")
theta = node(1085, 778, "θ", "g", "lat", GREY, "#f1f3f5", None)
text(1050, 772, "dispersion", 11.5, GREY, anchor="end"); text(1050, 787, "(fixed)", 11.5, GREY, anchor="end")
B = dot(990, 125, "B", "cg", None, below=False)
Gp = dot(1080, 125, "G", "pg", None, below=False)
s.append('<text x="990" y="108" font-size="18" font-weight="bold" fill="%s" text-anchor="middle">B<tspan font-size="11" dy="4">cg</tspan></text>' % GREY)
s.append('<text x="1080" y="108" font-size="18" font-weight="bold" fill="%s" text-anchor="middle">G<tspan font-size="11" dy="4">pg</tspan></text>' % GREY)
text(935, 128, "atlas profiles", 12, MUTED, anchor="end"); text(1095, 130, "program weights", 12, MUTED, anchor="start")

s.append('<g stroke="%s" stroke-width="2.2" fill="none" marker-end="url(#arr)">' % GREY)
edge(s_eta, eta)
edge(a, eta)
edge(gam, eta)
edge(D, eta, bend=(690, 205))
edge(eta, pi)
edge(tau, kap)
edge(tau, bet, bend=(170, 610))
edge(kap, delta)
edge(bet, delta)
edge(sig, delta)
edge(lam, delta)
edge(u, delta)
edge(D, delta)
edge(pi, mu)
edge(delta, mu)
edge(B, mu)
edge(Gp, mu)
edge(mu, Y)
edge(theta, Y)
s.append("</g>")

# right panel: equations and notation
X0 = 1215
s.append('<rect x="%d" y="110" width="360" height="430" rx="12" fill="#f8fafc" stroke="#e1e4e8" stroke-width="2"/>' % (X0 - 10))
text(X0 + 5, 140, "Generative model", 18, DARK, "bold", "start")
eqs = [
    ("Composition", BLUE),
    ("η<tspan font-size='11' dy='4'>ic</tspan><tspan dy='-4'> ~ N(a</tspan><tspan font-size='11' dy='4'>c</tspan>"
     "<tspan dy='-4'> + γ</tspan><tspan font-size='11' dy='4'>c</tspan><tspan dy='-4'> D</tspan>"
     "<tspan font-size='11' dy='4'>i</tspan><tspan dy='-4'>, s</tspan><tspan font-size='11' dy='4'>η</tspan>"
     "<tspan dy='-4'>²)</tspan>", DARK),
    ("π<tspan font-size='11' dy='4'>i</tspan><tspan dy='-4'> = softmax(η</tspan><tspan font-size='11' dy='4'>i</tspan>"
     "<tspan dy='-4'>)</tspan>", DARK),
    ("Program activity", ORANGE),
    ("δ<tspan font-size='11' dy='4'>icp</tspan><tspan dy='-4'> ~ N(κ</tspan><tspan font-size='11' dy='4'>cp</tspan>"
     "<tspan dy='-4'> + β</tspan><tspan font-size='11' dy='4'>cp</tspan><tspan dy='-4'> D</tspan>"
     "<tspan font-size='11' dy='4'>i</tspan><tspan dy='-4'> + λ</tspan><tspan font-size='11' dy='4'>p</tspan>"
     "<tspan dy='-4'> u</tspan><tspan font-size='11' dy='4'>ip</tspan><tspan dy='-4'>, σ</tspan>"
     "<tspan font-size='11' dy='4'>c</tspan><tspan dy='-4'>²)</tspan>", DARK),
    ("κ, β ~ horseshoe(τ);  u<tspan font-size='11' dy='4'>ip</tspan><tspan dy='-4'> ~ N(0, 1)</tspan>", DARK),
    ("Counts", DARK),
    ("μ<tspan font-size='11' dy='4'>ig</tspan><tspan dy='-4'> = L</tspan><tspan font-size='11' dy='4'>i</tspan>"
     "<tspan dy='-4'> Σ</tspan><tspan font-size='11' dy='4'>c</tspan><tspan dy='-4'> π</tspan>"
     "<tspan font-size='11' dy='4'>ic</tspan><tspan dy='-4'> E</tspan><tspan font-size='11' dy='4'>icg</tspan>", DARK),
    ("E<tspan font-size='11' dy='4'>icg</tspan><tspan dy='-4'> = softmax</tspan><tspan font-size='11' dy='4'>g</tspan>"
     "<tspan dy='-4'>(log B</tspan><tspan font-size='11' dy='4'>cg</tspan><tspan dy='-4'> + Σ</tspan>"
     "<tspan font-size='11' dy='4'>p</tspan><tspan dy='-4'> δ</tspan><tspan font-size='11' dy='4'>icp</tspan>"
     "<tspan dy='-4'> G</tspan><tspan font-size='11' dy='4'>pg</tspan><tspan dy='-4'>)</tspan>", DARK),
    ("L<tspan font-size='11' dy='4'>i</tspan><tspan dy='-4'>: library size (fixed)</tspan>", DARK),
    ("Y<tspan font-size='11' dy='4'>ig</tspan><tspan dy='-4'> ~ NegBin(μ</tspan><tspan font-size='11' dy='4'>ig</tspan>"
     "<tspan dy='-4'>, θ</tspan><tspan font-size='11' dy='4'>g</tspan><tspan dy='-4'>)</tspan>", DARK),
]
y = 172
for t, c in eqs:
    head = c != DARK or t in ("Counts",)
    if head:
        y += 8
        s.append('<text x="%d" y="%d" font-size="14" font-weight="bold" fill="%s">%s</text>' % (X0 + 5, y, c, t))
    else:
        s.append('<text x="%d" y="%d" font-size="16.5" fill="%s">%s</text>' % (X0 + 12, y, DARK, t))
    y += 30 if not head else 26

# notation legend
ly = 575
text(X0 + 5, ly, "Notation", 16, DARK, "bold", "start")
items = [("lat", "latent (estimated)"), ("obs", "observed"), ("det", "deterministic"), ("dot", "fixed input"),
         ("plate", "plate: repeated over the index")]
for k, (kind, lab) in enumerate(items):
    yy = ly + 28 + k * 30
    cx = X0 + 18
    if kind == "dot":
        s.append('<circle cx="%d" cy="%d" r="5" fill="%s"/>' % (cx, yy - 5, GREY))
    elif kind == "plate":
        s.append('<rect x="%d" y="%d" width="26" height="18" rx="4" fill="none" stroke="#8a9099" stroke-width="2"/>'
                 % (cx - 13, yy - 14))
    else:
        s.append('<circle cx="%d" cy="%d" r="10" fill="%s" stroke="%s" stroke-width="2"/>'
                 % (cx, yy - 5, OBS_F if kind == "obs" else "#ffffff", DARK))
        if kind == "det":
            s.append('<circle cx="%d" cy="%d" r="6.5" fill="none" stroke="%s" stroke-width="1.5"/>' % (cx, yy - 5, DARK))
    text(cx + 22, yy, lab, 14, "#333", anchor="start")
text(X0 + 5, ly + 190, "blue = cell mix, orange = program activity", 13, MUTED, anchor="start")

# footnote
s.append('<rect x="%d" y="790" width="360" height="72" rx="10" fill="#fff7ed" stroke="%s" stroke-width="1.5"/>'
         % (X0 - 10, ORANGE))
text(X0 + 5, 814, "Reported per-cell change β̂ is not this β:", 13.5, ORANGE, "bold", "start")
text(X0 + 5, 834, "π's posterior feeds the calibrated unmixing", 13, "#333", anchor="start")
text(X0 + 5, 852, "(modular, or 'cut', Bayesian inference)", 13, "#333", anchor="start")

s.append("</svg>")
with open(os.path.join(OUT, "2_2e_graphical_model_bayes.svg"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(s))
print("wrote 2_2e_graphical_model_bayes")
