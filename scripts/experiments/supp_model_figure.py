"""Supplementary figure: the full PALM model and how it is fitted and used (freeze v2).
-> docs/figures/supp/figS_palm_model.svg (+ .png at 2x via headless Edge; caption in figS_palm_model.md)

Panel a: plate diagram of the generative model as implemented (bayes/model.py, bayes/twostage.py stage 2), every
prior included: regularised horseshoe local scales, covariates on composition and activity (K plates, drawn where
each covariate weight lives), optional platform scale m_g (off in freeze v2, dashed), plug-in dispersion theta_g,
library size L_i (in the equations). Lineage grouping (kappa, beta, sigma, D per lineage l(c)) is in the table.
Panel b: two-stage fit, the cut, and the downstream stages B-E that produce the reported effects.
"""
import math, os

OUT = os.path.join("C:/Users/duy.pham/Documents/dev/program2cell", "docs", "figures", "supp")
os.makedirs(OUT, exist_ok=True)
W, H = 1800, 1300
DARK, MUTED, GREY = "#1f2937", "#6b7280", "#4b5563"
BLUE, BLUE_F, ORANGE, ORANGE_F = "#1d4ed8", "#e0eaff", "#c2410c", "#fff1e6"
OBS_F, PLATE = "#d1d5db", "#9ca3af"
R = 25
s = []


def text(x, y, t, size=13, color=DARK, weight="normal", anchor="middle", style=""):
    s.append('<text x="%g" y="%g" font-size="%g" fill="%s" font-weight="%s" text-anchor="%s"%s>%s</text>'
             % (x, y, size, color, weight, anchor, (' font-style="%s"' % style) if style else "", t))


def sub(base, idx, sup=None, size=19):
    out = '%s<tspan font-size="%g" dy="5">%s</tspan>' % (base, size * 0.58, idx)
    if sup:
        out += '<tspan font-size="%g" dy="-13">%s</tspan>' % (size * 0.58, sup)
    return out


def node(x, y, base, idx, kind="lat", color=DARK, fill="#ffffff", sup=None, dashed=False, label=None,
         lpos="below"):
    if kind == "obs":
        fill = OBS_F
    da = ' stroke-dasharray="5 3"' if dashed else ""
    s.append('<circle cx="%g" cy="%g" r="%g" fill="%s" stroke="%s" stroke-width="2"%s/>' % (x, y, R, fill, color, da))
    if kind == "det":
        s.append('<circle cx="%g" cy="%g" r="%g" fill="none" stroke="%s" stroke-width="1.4"/>' % (x, y, R - 4, color))
    s.append('<text x="%g" y="%g" font-size="19" font-weight="bold" fill="%s" text-anchor="middle">%s</text>'
             % (x - (3 if sup else 0), y + 5, color, sub(base, idx, sup)))
    if label:
        if lpos == "below":
            text(x, y + R + 15, label, 11.5, MUTED)
        elif lpos == "right":
            text(x + R + 6, y + 4, label, 11.5, MUTED, anchor="start")
        else:
            text(x - R - 6, y + 4, label, 11.5, MUTED, anchor="end")
    return (x, y, R)


def dot(x, y, base, idx, label=None, lpos="above"):
    s.append('<circle cx="%g" cy="%g" r="5.5" fill="%s"/>' % (x, y, GREY))
    if lpos == "above":
        s.append('<text x="%g" y="%g" font-size="17" font-weight="bold" fill="%s" text-anchor="middle">%s</text>'
                 % (x, y - 12, GREY, sub(base, idx, size=17)))
        if label:
            text(x, y - 34, label, 11.5, MUTED)
    else:
        s.append('<text x="%g" y="%g" font-size="17" font-weight="bold" fill="%s" text-anchor="end">%s</text>'
                 % (x - 10, y + 6, GREY, sub(base, idx, size=17)))
        if label:
            text(x + 10, y + 5, label, 11.5, MUTED, anchor="start")
    return (x, y, 6)


def plate(x, y, w, h, label, lx=None, ly=None, anchor="end", small=False):
    s.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="none" stroke="%s" stroke-width="%g"/>'
             % (x, y, w, h, 6 if small else 9, PLATE, 1.4 if small else 1.8))
    text(x + w - 6 if lx is None else lx, y + h - 7 if ly is None else ly, label, 11.5 if small else 13, MUTED,
         "bold", anchor)


def edge(a, b, bend=None, dashed=False):
    (x1, y1, r1), (x2, y2, r2) = a, b
    da = ' stroke-dasharray="5 3"' if dashed else ""
    if bend is None:
        d = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / d, (y2 - y1) / d
        s.append('<path d="M%g,%g L%g,%g"%s/>' % (x1 + ux * r1, y1 + uy * r1, x2 - ux * (r2 + 2), y2 - uy * (r2 + 2), da))
    else:
        cx, cy = bend
        d1 = math.hypot(cx - x1, cy - y1); d2 = math.hypot(x2 - cx, y2 - cy)
        s.append('<path d="M%g,%g Q%g,%g %g,%g"%s/>' % (x1 + (cx - x1) / d1 * r1, y1 + (cy - y1) / d1 * r1, cx, cy,
                                                        x2 - (x2 - cx) / d2 * (r2 + 2), y2 - (y2 - cy) / d2 * (r2 + 2), da))


s.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
         'font-family="Helvetica, Arial, sans-serif">' % (W, H, W, H))
s.append('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker></defs>' % GREY)
s.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))

def fmt(t):
    return (t.replace("<sub>", '<tspan baseline-shift="sub" font-size="9.5">').replace("</sub>", "</tspan>")
             .replace("<sup>", '<tspan baseline-shift="super" font-size="9.5">').replace("</sup>", "</tspan>"))


# ============================== panel a ==============================
text(20, 38, "a", 30, DARK, "bold", "start")
text(52, 36, "Generative model of PALM (stage A), as implemented", 18, DARK, "bold", "start")

plate(150, 150, 650, 500, "C cell types", 160, 170, "start")
plate(150, 400, 810, 345, "P programs", 950, 737)
plate(480, 110, 630, 670, "N patients", 490, 772, "start")
plate(990, 300, 175, 565, "G genes", 1157, 857)

# composition (blue)
s_eta = node(310, 90, "s", "η", color=BLUE, fill=BLUE_F, label="mix noise", lpos="right")
a = node(205, 210, "a", "c", color=BLUE, fill=BLUE_F, label="baseline mix")
gam = node(305, 210, "γ", "c", color=BLUE, fill=BLUE_F, label="group shift")
plate(250, 280, 105, 85, "K", small=True)
A = node(300, 318, "A", "kc", color=BLUE, fill=BLUE_F)
eta = node(560, 300, "η", "ic", color=BLUE, fill=BLUE_F, label="logit")
pi = node(700, 300, "π", "ic", "det", color=BLUE, fill=BLUE_F, label="read fraction")

# activity (orange)
sig = node(430, 345, "σ", "c", color=ORANGE, fill=ORANGE_F, label="residual sd")
tk = node(55, 440, "τ", "κ", color=ORANGE, fill=ORANGE_F, label="global scale")
tb = node(55, 610, "τ", "β", color=ORANGE, fill=ORANGE_F, label="global scale")
lk = node(200, 450, "ℓ", "cp", sup="κ", color=ORANGE, fill=ORANGE_F)
kap = node(300, 450, "κ", "cp", color=ORANGE, fill=ORANGE_F)
text(300, 490, "baseline activity", 11.5, MUTED)
lb = node(200, 560, "ℓ", "cp", sup="β", color=ORANGE, fill=ORANGE_F)
text(200, 600, "local scales", 11.5, MUTED)
bet = node(300, 560, "β", "cp", color=ORANGE, fill=ORANGE_F)
text(300, 600, "group effect", 11.5, MUTED)
plate(362, 575, 100, 70, "K", small=True)
Dk = node(410, 607, "D", "kcp", color=ORANGE, fill=ORANGE_F)
lam = node(230, 690, "λ", "p", color=ORANGE, fill=ORANGE_F, label="factor scale")
delta = node(640, 480, "δ", "icp", color=ORANGE, fill=ORANGE_F)
text(640, 524, "activity", 11.5, MUTED)
u = node(890, 600, "u", "ip", color=ORANGE, fill=ORANGE_F, label="patient factor (all types)")

# patient-level observed / fixed
T = node(960, 190, "T", "i", "obs")
text(960, 155, "group", 11.5, MUTED)
plate(840, 230, 90, 80, "K", small=True)
X = node(880, 265, "X", "ik", "obs")
text(880, 320, "", 1)

# counts
mu = node(1050, 420, "μ", "ig", "det", label=None)
text(1080, 455, "mean", 11.5, MUTED, anchor="start")
Y = node(1050, 610, "Y", "ig", "obs")
text(1080, 645, "counts", 11.5, MUTED, anchor="start")
theta = dot(1040, 835, "θ", "g", None, lpos="left")
text(1048, 840, "plug-in", 11, MUTED, anchor="start")
m = node(1120, 808, "m", "g", dashed=True, color=GREY)
sm = node(930, 805, "σ", "m", dashed=True, color=GREY)
text(930, 845, "platform scale", 11.5, MUTED); text(930, 860, "(optional, off in v2)", 11.5, MUTED)
B = dot(1000, 72, "B", "cg", "atlas profiles")
G = dot(1100, 72, "G", "pg", "program weights")

s.append('<g stroke="%s" stroke-width="1.8" fill="none" marker-end="url(#arr)">' % GREY)
for e in [(s_eta, eta), (a, eta), (gam, eta), (A, eta), (eta, pi), (lk, kap), (lb, bet), (kap, delta),
          (bet, delta), (Dk, delta), (u, delta), (pi, mu), (delta, mu), (B, mu), (G, mu), (mu, Y), (theta, Y)]:
    edge(*e)
edge(sig, delta)
edge(tk, kap, bend=(170, 395))
edge(tb, bet, bend=(170, 625))
edge(lam, delta, bend=(450, 690))
edge(T, eta, bend=(760, 175))
edge(T, delta, bend=(985, 470))
edge(X, eta, bend=(735, 238))
edge(X, delta)
edge(m, mu, bend=(1185, 600), dashed=True)
edge(sm, m, dashed=True)
s.append("</g>")

# ============================== specification table ==============================
X0 = 1215
s.append('<rect x="%d" y="55" width="565" height="790" rx="10" fill="#f9fafb" stroke="#e5e7eb" stroke-width="1.5"/>'
         % (X0 - 15))
text(X0, 82, "Specification (freeze v2: activity = 'common', use_m = False)", 14.5, DARK, "bold", "start")
rows = [
    ("Composition", BLUE),
    ("a ~ ZeroSumNormal(1.5²) over c;  γ<sub>c</sub> ~ N(0, 1);  A<sub>kc</sub> ~ N(0, 1)", None),
    ("s<sub>η</sub> ~ HalfNormal(0.5);  ε<sub>ic</sub> ~ N(0, 1)", None),
    ("η<sub>ic</sub> = a<sub>c</sub> + γ<sub>c</sub>T<sub>i</sub> + Σ<sub>k</sub> X<sub>ik</sub>A<sub>kc</sub> + s<sub>η</sub>ε<sub>ic</sub>", None),
    ("π<sub>i</sub> = softmax(η<sub>i</sub>)   (read fractions)", None),
    ("Program activity (per lineage l(c); l(c) = c by default)", ORANGE),
    ("κ<sub>lp</sub> = τ<sub>κ</sub> ℓ̃<sup>κ</sup><sub>lp</sub> z<sup>κ</sup><sub>lp</sub>,  "
     "τ<sub>κ</sub> ~ HalfCauchy(0.3),  ℓ<sup>κ</sup><sub>lp</sub> ~ HalfCauchy(1)", None),
    ("ℓ̃² = c²ℓ² / (c² + τ²ℓ²),  c² = 1 (regularised horseshoe);  β likewise", None),
    ("λ<sub>p</sub> ~ HalfNormal(0.3);  u<sub>ip</sub> ~ N(0, 1);  σ<sub>l</sub> ~ HalfNormal(0.3)", None),
    ("D<sub>klp</sub> ~ N(0, 0.3²);  z<sub>ilp</sub> ~ N(0, 1)", None),
    ("δ<sub>ilp</sub> = κ<sub>lp</sub> + β<sub>lp</sub>T<sub>i</sub> + Σ<sub>k</sub> X<sub>ik</sub>D<sub>klp</sub>"
     " + λ<sub>p</sub>u<sub>ip</sub> + σ<sub>l</sub>z<sub>ilp</sub>", None),
    ("Counts", DARK),
    ("E<sub>icg</sub> = softmax<sub>g</sub>( log B<sub>cg</sub> + Σ<sub>p</sub> δ<sub>i,l(c),p</sub> G<sub>pg</sub> )", None),
    ("μ<sub>ig</sub> = L<sub>i</sub> m<sub>g</sub> Σ<sub>c</sub> π<sub>ic</sub>E<sub>icg</sub>", None),
    ("m<sub>g</sub> = 1 in v2 (else exp(σ<sub>m</sub>z<sub>g</sub>), σ<sub>m</sub> ~ HalfNormal(0.3))", None),
    ("Y<sub>ig</sub> ~ NegBin(mean μ<sub>ig</sub>, dispersion θ<sub>g</sub>)", None),
    ("Fixed inputs", GREY),
    ("B<sub>cg</sub>: atlas reference profiles (pseudo-bulk per donor, averaged)", None),
    ("G<sub>pg</sub>: PROGENy weights, rescaled to median |w| = 1 per program", None),
    ("L<sub>i</sub>: library size;  θ<sub>g</sub>: moment estimate from the bulk,", None),
    ("    shrunk to the median on the log scale (plug-in)", None),
    ("T<sub>i</sub>: group (0/1);  X<sub>ik</sub>: covariates", None),
    ("Cell fractions", GREY),
    ("cell fraction ∝ π<sub>ic</sub> / r<sub>c</sub>;  r<sub>c</sub> = mean UMI per cell in the atlas", None),
]
y = 112
for t, head in rows:
    t = fmt(t)
    if head:
        y += 10
        s.append('<text x="%d" y="%d" font-size="13.5" font-weight="bold" fill="%s">%s</text>' % (X0, y, head, t))
        y += 25
    else:
        s.append('<text x="%d" y="%d" font-size="13.5" fill="%s" xml:space="preserve">%s</text>' % (X0 + 8, y, DARK, t))
        y += 25

# notation
ny = y + 12
text(X0, ny, "Notation", 13.5, DARK, "bold", "start")
leg = [("lat", "latent"), ("obs", "observed"), ("det", "deterministic"), ("dot", "fixed / plug-in"),
       ("dash", "optional, off in v2")]
for k, (kind, lab) in enumerate(leg):
    cx, cy = X0 + 12 + (k % 3) * 180, ny + 26 + (k // 3) * 28
    if kind == "dot":
        s.append('<circle cx="%d" cy="%d" r="5" fill="%s"/>' % (cx, cy - 4, GREY))
    else:
        s.append('<circle cx="%d" cy="%d" r="9" fill="%s" stroke="%s" stroke-width="1.6"%s/>'
                 % (cx, cy - 4, OBS_F if kind == "obs" else "#ffffff", DARK,
                    ' stroke-dasharray="4 2"' if kind == "dash" else ""))
        if kind == "det":
            s.append('<circle cx="%d" cy="%d" r="5.5" fill="none" stroke="%s" stroke-width="1.2"/>' % (cx, cy - 4, DARK))
    text(cx + 16, cy, lab, 12.5, DARK, anchor="start")

# ============================== panel b ==============================
PY = 950
text(20, PY - 18, "b", 30, DARK, "bold", "start")
text(52, PY - 20, "How the model is fitted and used: a modular (cut) pipeline", 18, DARK, "bold", "start")
boxes = [
    ("0  Platform offsets", "(optional, real bulk)", ["per-gene log2 offset o<sub>g</sub>", "from the study bulk;", "Y<sub>ig</sub> ← Y<sub>ig</sub> / 2<sup>o<sub>g</sub></sup>"], "#f3f4f6", GREY),
    ("A1  Per-patient summary", "(stage 1)", ["mode of (η<sub>i</sub>, δ<sub>i</sub>) by Fisher", "scoring; likelihood ≈ Gaussian", "(Λ<sub>i</sub>, h<sub>i</sub>); θ<sub>g</sub> plugged in"], BLUE_F, BLUE),
    ("A2  Population fit", "(stage 2, NUTS)", ["NUTS on a, γ, s<sub>η</sub>, κ, β, λ, σ, …;", "η<sub>i</sub>, δ<sub>i</sub> drawn from their", "exact Gaussian conditional"], BLUE_F, BLUE),
    ("B  Reallocation", "", ["reads split by π<sub>ic</sub>B<sub>cg</sub>;", "δ̂<sub>icp</sub> by estimate_delta", "(attenuated, leaks)"], "#f3f4f6", GREY),
    ("C  Calibration", "(reference donors)", ["thinned donors with known", "truth: ridge map δ̂ → δ,", "conformal intervals"], "#f3f4f6", GREY),
    ("D  Effects router", "", ["unmix (leakage operator,", "REML, bootstrap) vs calibrated;", "width rule + guard → β̂<sub>cp</sub>"], ORANGE_F, ORANGE),
    ("E  Readouts", "", ["β̂<sub>cp</sub> ± 90%, calls; contributions", "(Shapley); shared response;", "robustness; design limit"], "#ecfdf5", "#15803d"),
]
bw, gap, x = 228, 22, 22
centers = []
for k, (title, subt, lines, fill, col) in enumerate(boxes):
    s.append('<rect x="%d" y="%d" width="%d" height="150" rx="9" fill="%s" stroke="%s" stroke-width="1.6"/>'
             % (x, PY, bw, fill, col))
    text(x + 12, PY + 26, title, 14, col, "bold", "start")
    if subt:
        text(x + 12, PY + 45, subt, 12, MUTED, anchor="start")
    for j, ln in enumerate(lines):
        s.append('<text x="%d" y="%d" font-size="12.5" fill="%s">%s</text>' % (x + 12, PY + 74 + j * 21, DARK, fmt(ln)))
    centers.append(x + bw / 2)
    if k < len(boxes) - 1:
        s.append('<path d="M%d,%d L%d,%d" stroke="%s" stroke-width="1.8" marker-end="url(#arr)"/>'
                 % (x + bw + 2, PY + 75, x + bw + gap - 3, PY + 75, GREY))
    x += bw + gap
# the cut between A2 and B
cx = 22 + 3 * (bw + gap) - gap / 2
s.append('<path d="M%g,%d L%g,%d" stroke="#b91c1c" stroke-width="2.2" stroke-dasharray="7 4"/>' % (cx, PY - 12, cx, PY + 168))
text(cx, PY + 188, "cut: π's posterior draws pass forward; nothing feeds back into the model", 12.5, "#b91c1c", "bold")
text(cx, PY + 206, "(the model's own δ and β are not reported)", 12, "#b91c1c")
# inputs and side inputs
text(22, PY + 300, "Inputs: bulk counts Y, atlas (reference specimens of other donors), programs G, group T, covariates X",
     12.5, MUTED, anchor="start")
s.append('<path d="M%g,%d L%g,%d" stroke="%s" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#arr)"/>'
         % (centers[4], PY + 240, centers[4], PY + 153, GREY))
s.append('<path d="M%g,%d L%g,%d" stroke="%s" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#arr)"/>'
         % (centers[5], PY + 240, centers[5], PY + 153, GREY))
text(centers[4], PY + 258, "atlas donors not used for B", 12, MUTED)
text(centers[5], PY + 258, "design check: ~10 simulated cohorts", 12, MUTED)
text(centers[5], PY + 274, "with this study's fractions and design", 12, MUTED)
text(centers[1], PY + 236, "Bayesian fit (A1–A2) for McFarland, Kang, Zhang;", 12, MUTED)
text(centers[1], PY + 252, "PALM-lite (EM, no posterior) for Tahoe, Parse, COMBAT", 12, MUTED)

s.append("</svg>")
with open(os.path.join(OUT, "figS_palm_model.svg"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(s))
print("wrote figS_palm_model.svg")
