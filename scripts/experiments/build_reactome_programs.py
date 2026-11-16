"""Six Reactome immune signalling pathways -> data/reactome_immune6.csv (program,gene,weight; pb.load_programs format).

User choice 2026-11-10, for the PTB application only (not benchmarked). Source: ReactomePathways.gmt
(https://reactome.org/download/current/ReactomePathways.gmt.zip, CC0), passed as argument.
  TLR   R-HSA-168898  Toll-like Receptor Cascades
  NLR   R-HSA-168643  Nucleotide-binding domain, leucine rich repeat containing receptor (NLR) signaling pathways
  IFNb  R-HSA-909733  Interferon alpha/beta signaling (type I IFN: IFN-beta and IFN-alpha share IFNAR)
  TNF   R-HSA-75893   TNF signaling
  BCR   R-HSA-983705  Signaling by the B Cell Receptor (BCR)
  TCR   R-HSA-202403  TCR signaling
These are member sets (the pathway's proteins), not response footprints like PROGENy, and they overlap (shared
NF-kB, MAPK and proteasome genes). Weight 1 for every member; overlaps are printed and kept.
Usage: python build_reactome_programs.py PATH/TO/ReactomePathways.gmt
"""
import itertools, os, sys
import pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
OUT = os.path.join(ROOT, "data", "reactome_immune6.csv")
IDS = {"TLR": "R-HSA-168898", "NLR": "R-HSA-168643", "IFNb": "R-HSA-909733", "TNF": "R-HSA-75893",
       "BCR": "R-HSA-983705", "TCR": "R-HSA-202403"}


def main(gmt):
    sets = {}
    for line in open(gmt, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) > 2 and f[1] in IDS.values():
            sets[f[1]] = (f[0], sorted(set(g for g in f[2:] if g)))
    missing = [k for k, v in IDS.items() if v not in sets]
    if missing:
        raise SystemExit("not in this Reactome release: %s" % missing)
    rows = [dict(program=k, gene=g, weight=1.0) for k, rid in IDS.items() for g in sets[rid][1]]
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT)
    for k, rid in IDS.items():
        print("  %-4s %-13s %3d genes  %s" % (k, rid, len(sets[rid][1]), sets[rid][0]))
    print("shared genes (pairs):")
    g = {k: set(sets[r][1]) for k, r in IDS.items()}
    for a, b in itertools.combinations(IDS, 2):
        n = len(g[a] & g[b])
        if n:
            print("  %-4s %-4s %3d  (%.0f%% of the smaller)" % (a, b, n, 100 * n / min(len(g[a]), len(g[b]))))


if __name__ == "__main__":
    main(sys.argv[1])
