"""PTB program set with response-based IFN-beta and TNF -> data/immune_response6.csv (program,gene,weight).

User request 2026-11-11 (after the Reactome member-set run gave no calls): IFN-beta and TNF from MSigDB Hallmark
response sets (genes that change when the pathway is active; data/hallmark_immune.csv), the other four kept from
Reactome member sets (data/reactome_immune6.csv; no standard response sets exist for them):
  IFNb  Hallmark INTERFERON_ALPHA_RESPONSE (97 genes; type I IFN: IFN-beta and IFN-alpha share IFNAR)
  TNF   Hallmark TNFA_SIGNALING_VIA_NFKB (200 genes)
  TLR, NLR, BCR, TCR   Reactome (build_reactome_programs.py)
Weight 1 for every member; overlaps printed and kept. PTB application only, not benchmarked.
Second set (user request 2026-11-12) -> data/immune_response7.csv: TLR, NLR, TNF, IFNI, IFNg, BCR, TCR, with
  IFNI  Hallmark INTERFERON_ALPHA_RESPONSE (the same 97 genes as IFNb above, renamed: type I IFN, alpha and beta share
        IFNAR and induce nearly the same genes, so they are one program; user decision)
  IFNg  Hallmark INTERFERON_GAMMA_RESPONSE (200 genes)
Third set (user choice 2026-11-12, option 1) -> data/immune_response8.csv: as immune_response7 with the IFN genes
split by membership, so the two IFN programs no longer share genes:
  IFNI  Type I IFN response genes not in IFN-gamma (Hallmark alpha minus gamma)
  IFNg  IFN-gamma response genes not in type I (Hallmark gamma minus alpha)
  ISG   the genes in both (interferon-stimulated genes induced by either)
Fourth set (user choice 2026-11-12) -> data/immune_reactome_ifn8.csv: as immune_response8 with the IFN sets from
Reactome instead of Hallmark, split the same way:
  IFNI  Interferon alpha/beta signaling (R-HSA-909733) members not in IFN-gamma signaling
  IFNg  Interferon gamma signaling (R-HSA-877300) members not in alpha/beta
  ISG   members of both
Source: data/reactome/ReactomePathways.gmt (Reactome download, CC0).
Fifth set (user choice 2026-11-12; Interferome was offline) -> data/immune_reactome_ifn7.csv: as immune_response7
(TLR, NLR, TNF, IFNI, IFNg, BCR, TCR) with the two IFN programs as the WHOLE Reactome sets, not split:
  IFNI  Interferon alpha/beta signaling (R-HSA-909733), 78 genes
  IFNg  Interferon gamma signaling (R-HSA-877300), 99 genes (28 shared with IFNI, kept in both)
Usage: python build_immune_response_programs.py [7|8|r8|r7]
"""
import itertools, os, sys
import pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
OUT = os.path.join(ROOT, "data", "immune_response6.csv")
HALLMARK = {"IFNb": "INTERFERON_ALPHA_RESPONSE", "TNF": "TNFA_SIGNALING_VIA_NFKB"}
REACTOME = ["TLR", "NLR", "BCR", "TCR"]


def main():
    h = pd.read_csv(os.path.join(ROOT, "data", "hallmark_immune.csv"))
    r = pd.read_csv(os.path.join(ROOT, "data", "reactome_immune6.csv"))
    parts = [h[h.program == v].assign(program=k, weight=1.0) for k, v in HALLMARK.items()]
    parts.append(r[r.program.isin(REACTOME)])
    out = pd.concat(parts)[["program", "gene", "weight"]]
    order = ["TLR", "NLR", "IFNb", "TNF", "BCR", "TCR"]
    out = out.set_index("program").loc[order].reset_index()
    out.to_csv(OUT, index=False)
    print("wrote", OUT)
    g = {p: set(out.gene[out.program == p]) for p in order}
    for p in order:
        print("  %-4s %3d genes (%s)" % (p, len(g[p]), "Hallmark " + HALLMARK[p] if p in HALLMARK else "Reactome"))
    for a, b in itertools.combinations(order, 2):
        n = len(g[a] & g[b])
        if n:
            print("  %-4s %-4s %3d shared (%.0f%% of the smaller)" % (a, b, n, 100 * n / min(len(g[a]), len(g[b]))))


def main7():
    h = pd.read_csv(os.path.join(ROOT, "data", "hallmark_immune.csv"))
    r = pd.read_csv(os.path.join(ROOT, "data", "reactome_immune6.csv"))
    hm = {"TNF": "TNFA_SIGNALING_VIA_NFKB", "IFNI": "INTERFERON_ALPHA_RESPONSE", "IFNg": "INTERFERON_GAMMA_RESPONSE"}
    parts = [h[h.program == v].assign(program=k, weight=1.0) for k, v in hm.items()]
    parts.append(r[r.program.isin(REACTOME)])
    out = pd.concat(parts)[["program", "gene", "weight"]]
    order = ["TLR", "NLR", "TNF", "IFNI", "IFNg", "BCR", "TCR"]
    out = out.set_index("program").loc[order].reset_index()
    f = os.path.join(ROOT, "data", "immune_response7.csv")
    out.to_csv(f, index=False)
    print("wrote", f)
    g = {p: set(out.gene[out.program == p]) for p in order}
    for p in order:
        print("  %-4s %3d genes" % (p, len(g[p])))
    for a, b in itertools.combinations(order, 2):
        n = len(g[a] & g[b])
        if n:
            print("  %-4s %-4s %3d shared (%.0f%% of the smaller)" % (a, b, n, 100 * n / min(len(g[a]), len(g[b]))))


def main8():
    d = pd.read_csv(os.path.join(ROOT, "data", "immune_response7.csv"))
    a, g = set(d.gene[d.program == "IFNI"]), set(d.gene[d.program == "IFNg"])
    rest = d[~d.program.isin(["IFNI", "IFNg"])]
    ifn = [pd.DataFrame(dict(program=k, gene=sorted(v), weight=1.0))
           for k, v in (("IFNI", a - g), ("IFNg", g - a), ("ISG", a & g))]
    out = pd.concat([rest] + ifn)
    order = ["TLR", "NLR", "TNF", "IFNI", "IFNg", "ISG", "BCR", "TCR"]
    out = out.set_index("program").loc[order].reset_index()[["program", "gene", "weight"]]
    f = os.path.join(ROOT, "data", "immune_response8.csv")
    out.to_csv(f, index=False)
    print("wrote", f, out.groupby("program", sort=False).size().to_dict())


def main_r8():
    sets = {}
    for line in open(os.path.join(ROOT, "data", "reactome", "ReactomePathways.gmt"), encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) > 2 and f[1] in ("R-HSA-909733", "R-HSA-877300"):
            sets[f[1]] = (f[0], set(g for g in f[2:] if g))
    for rid, (name, g) in sets.items():
        print("  %s %s: %d genes" % (rid, name, len(g)))
    a, g = sets["R-HSA-909733"][1], sets["R-HSA-877300"][1]
    d = pd.read_csv(os.path.join(ROOT, "data", "immune_response8.csv"))
    rest = d[~d.program.isin(["IFNI", "IFNg", "ISG"])]
    ifn = [pd.DataFrame(dict(program=k, gene=sorted(v), weight=1.0))
           for k, v in (("IFNI", a - g), ("IFNg", g - a), ("ISG", a & g))]
    out = pd.concat([rest] + ifn)
    order = ["TLR", "NLR", "TNF", "IFNI", "IFNg", "ISG", "BCR", "TCR"]
    out = out.set_index("program").loc[order].reset_index()[["program", "gene", "weight"]]
    f = os.path.join(ROOT, "data", "immune_reactome_ifn8.csv")
    out.to_csv(f, index=False)
    print("wrote", f, out.groupby("program", sort=False).size().to_dict())


def main_r7():
    sets = {}
    for line in open(os.path.join(ROOT, "data", "reactome", "ReactomePathways.gmt"), encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) > 2 and f[1] in ("R-HSA-909733", "R-HSA-877300"):
            sets[f[1]] = sorted(set(g for g in f[2:] if g))
    d = pd.read_csv(os.path.join(ROOT, "data", "immune_response7.csv"))
    rest = d[~d.program.isin(["IFNI", "IFNg"])]
    ifn = [pd.DataFrame(dict(program=k, gene=sets[r], weight=1.0))
           for k, r in (("IFNI", "R-HSA-909733"), ("IFNg", "R-HSA-877300"))]
    out = pd.concat([rest] + ifn)
    order = ["TLR", "NLR", "TNF", "IFNI", "IFNg", "BCR", "TCR"]
    out = out.set_index("program").loc[order].reset_index()[["program", "gene", "weight"]]
    f = os.path.join(ROOT, "data", "immune_reactome_ifn7.csv")
    out.to_csv(f, index=False)
    print("wrote", f, out.groupby("program", sort=False).size().to_dict(),
          "IFNI & IFNg shared:", len(set(sets["R-HSA-909733"]) & set(sets["R-HSA-877300"])))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "r7":
        main_r7()
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "r8":
        main_r8()
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "8":
        main8()
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "7":
        main7()
        sys.exit()
    main()
