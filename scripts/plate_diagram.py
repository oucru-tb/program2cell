"""Plate diagram of program2cell (freeze v2) in the style of cell2location (Kleshchevnikov et al. 2022), Fig. 1b:
circles = latent, shaded = observed, small filled dots = fixed inputs, double circles = deterministic.
Module 1 (PALM, fractions) on the left, module 2 (effects) on the right, joined by a one-way cut.
Output: docs/figures/methods/palm_plate.{png,pdf}.  Usage: python scripts/plate_diagram.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures", "methods")
INK, BLUE, GREY, MUTED = "#222222", "#1f5fa8", "#d9d9d9", "#9a9a9a"
R = 0.42

fig, ax = plt.subplots(figsize=(15.6, 8.0))
ax.set_xlim(0, 26.2); ax.set_ylim(-0.5, 12.4); ax.set_aspect("equal"); ax.axis("off")
pos = {}


def node(key, x, y, label, kind="latent", muted=False, fs=12):
    pos[key] = (x, y)
    ec = MUTED if muted else (BLUE if kind in ("latent", "det") else INK)
    if kind == "fixed":
        ax.add_patch(Circle((x, y), 0.09, color=INK, zorder=3))
        ax.text(x, y + 0.28, label, ha="center", va="bottom", fontsize=fs, zorder=4)
        return
    fc = GREY if kind == "obs" else "white"
    ax.add_patch(Circle((x, y), R, fc=fc, ec=ec, lw=1.6, zorder=3, ls="--" if muted else "-"))
    if kind == "det":
        ax.add_patch(Circle((x, y), R - 0.07, fc="none", ec=ec, lw=1.0, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4, color=MUTED if muted else INK)


def edge(a, b, color=INK, ls="-", rad=0.0):
    (x0, y0), (x1, y1) = pos[a], pos[b]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, ls=ls, shrinkA=13, shrinkB=13,
                                mutation_scale=12, connectionstyle="arc3,rad=%s" % rad))


def plate(x0, y0, x1, y1, label, corner="br"):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=INK, lw=1.0, zorder=1))
    tx = x1 - 0.12 if corner.endswith("r") else x0 + 0.12
    top = corner.startswith("t")
    ax.text(tx, y1 - 0.12 if top else y0 + 0.12, label, ha="right" if corner.endswith("r") else "left",
            va="top" if top else "bottom", fontsize=10.5)


# ---------------- module 1: PALM generative model (fractions) ----------------
ax.text(0.3, 12.0, "Module 1  PALM generative model (reported: fractions π)", fontsize=13, weight="bold",
        va="center")
# population parameters (outside the sample plate)
node("a", 2.4, 10.4, r"$a_c$"); node("gam", 3.7, 10.4, r"$\gamma_c$"); node("seta", 5.0, 10.4, r"$s_\eta$")
node("kb", 7.6, 10.4, r"$\kappa_{cp}$", muted=True); node("beta", 8.9, 10.4, r"$\beta_{cp}$", muted=True)
node("lsd", 10.2, 10.4, r"$\lambda,\sigma$", muted=True, fs=11)
node("B", 12.6, 10.6, r"$B_{cg}$", "fixed"); node("G", 13.8, 10.6, r"$G_{pg}$", "fixed")
node("th", 13.95, 2.0, r"$\theta_g$", "fixed")
# sample-level
node("T", 1.2, 7.4, r"$T_i, X_i$", "fixed")
node("eta", 3.0, 7.0, r"$\eta_{ic}$"); node("pi", 4.8, 7.0, r"$\pi_{ic}$", "det")
node("delta", 8.9, 7.0, r"$\delta_{icp}$", muted=True)
node("E", 12.6, 6.1, r"$E_{icg}$", "det")
node("f", 12.6, 4.1, r"$f_{ig}$", "det")
node("Y", 12.6, 2.0, r"$Y_{ig}$", "obs")
node("L", 10.6, 2.0, r"$L_i$", "fixed")
# plates
plate(0.5, 0.9, 14.35, 8.7, r"$i = 1..N$ samples", corner="bl")
plate(2.3, 5.25, 13.4, 8.25, r"$c = 1..C$ types", corner="bl")
plate(8.0, 6.15, 9.8, 7.95, r"$p$")
plate(11.8, 1.2, 13.4, 6.95, r"$g$", corner="tl")
for a in ("a", "gam", "seta"):
    edge(a, "eta")
edge("T", "eta")
edge("eta", "pi")
for a in ("kb", "beta", "lsd"):
    edge(a, "delta", color=MUTED, ls="--")
edge("T", "delta", color=MUTED, ls="--", rad=-0.25)
edge("delta", "E", color=MUTED, ls="--")
edge("B", "E"); edge("G", "E", rad=0.15)
edge("pi", "f", rad=0.25)
edge("E", "f"); edge("f", "Y"); edge("L", "Y"); edge("th", "Y")

# ---------------- the cut ----------------
py = 9.15
ax.plot([5.45, 5.45, 15.3], [7.35, py, py], color=INK, lw=1.8, zorder=2, solid_joinstyle="round")
ax.annotate("", xy=(16.75, py), xytext=(15.2, py), zorder=2,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8, mutation_scale=14, shrinkA=0, shrinkB=0))
vx, vy = 15.55, py                      # one-way valve (diode) on the cut arrow
ax.add_patch(Polygon([(vx - 0.27, vy + 0.24), (vx - 0.27, vy - 0.24), (vx + 0.15, vy)], closed=True,
                     fc=INK, ec=INK, zorder=5))
ax.plot([vx + 0.17, vx + 0.17], [vy - 0.26, vy + 0.26], color=INK, lw=2.4, zorder=5)
ax.text(vx, vy + 0.38, "cut", ha="center", va="bottom", fontsize=11, weight="bold")
ax.text(vx, vy - 0.4, "draws of π only,\nno feedback", ha="center", va="top", fontsize=9.5, style="italic")

# ---------------- module 2: effect module ----------------
ax.text(16.75, 12.0, "Module 2  effects (calibrated estimator)", fontsize=13, weight="bold", va="center")


def box(y, title, sub, h=1.25, fc="white", key=None):
    ax.add_patch(FancyBboxPatch((16.75, y - h / 2), 9.3, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                                fc=fc, ec=BLUE, lw=1.4, zorder=3))
    ax.text(17.0, y + 0.18, title, fontsize=11.5, weight="bold", va="center", zorder=4)
    ax.text(17.0, y - 0.3, sub, fontsize=9.8, va="center", zorder=4, color="#333333")
    if key:
        pos[key] = (21.4, y)


box(9.4, r"$\pi^{(s)}_{ic}$, $s = 1..50$ posterior draws", "fraction uncertainty carried as multiple imputation",
    key="m0")
box(7.65, r"B  reallocation  $\hat\delta^{(s)}_{icp}$", r"bulk residual given $\pi^{(s)}$, split to types via $B_{cg}$ and $G_{pg}$", key="m1")
box(5.9, "C  calibration on reference donors", "conformal quantiles on held-out donors; simulated leakage", key="m2")
box(4.15, "D  unmixing across samples, router, guard", "pre-registered selection rule, outside the probability model", key="m3")
box(2.3, r"E  readouts  $\hat\beta_{cp}$, 90% interval, calls", "contributions, shared response, design limit",
    fc="#eef3fa", key="m4", h=1.35)
for a, b in (("m0", "m1"), ("m1", "m2"), ("m2", "m3"), ("m3", "m4")):
    (x0, y0), (x1, y1) = pos[a], pos[b]
    ax.annotate("", xy=(x1, y1 + 0.66), xytext=(x0, y0 - 0.66),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=12))
ax.text(16.75, 0.95, "Reference donors and the study's own fractions and design\nenter C and D; nothing flows back "
        "to module 1.", fontsize=9.8, va="center", color="#333333")

# legend
ly, r0 = -0.15, 0.22
for i, (kind, lab) in enumerate((("latent", "latent"), ("det", "deterministic"), ("obs", "observed"),
                                 ("fixed", "fixed input"))):
    x = 0.8 + i * 2.9
    if kind == "fixed":
        ax.add_patch(Circle((x, ly), 0.09, color=INK))
    else:
        ax.add_patch(Circle((x, ly), r0, fc=GREY if kind == "obs" else "white", ec=INK if kind == "obs" else BLUE,
                            lw=1.4))
        if kind == "det":
            ax.add_patch(Circle((x, ly), r0 - 0.06, fc="none", ec=BLUE, lw=0.9))
    ax.text(x + 0.38, ly, lab, va="center", fontsize=10)
ax.plot([12.4, 13.0], [ly, ly], color=MUTED, ls="--", lw=1.4)
ax.text(13.1, ly, "in the model, not reported", va="center", fontsize=10, color="#555555")

os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "palm_plate." + ext), dpi=200, bbox_inches="tight")
print("wrote", os.path.join(OUT, "palm_plate.png"))
