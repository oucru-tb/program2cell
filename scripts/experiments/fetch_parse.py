"""Parse Biosciences 10M PBMC cytokine dataset (12 donors, 90 cytokines + PBS; CC BY-NC 4.0): fetch a subset of the
published pseudobulk files (one small h5ad per cell type x cytokine, rows = donors, summed raw counts) by HTTP range
requests on the 6.2 GB zip, without downloading it whole.

Source: https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/

Stages
  list   the zip's central directory -> data/parse/listing.tsv (name, method, offset, compressed size)
  get    the entries whose cytokine is in CYTOKINES (or PBS) -> data/parse/pseudobulk/<name>
Usage: python fetch_parse.py list|get
"""
import os, sys, struct, zlib, time
import urllib.request

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
D = os.path.join(ROOT, "data", "parse")
URL = "https://parse-wget.s3.us-west-2.amazonaws.com/10m/pseudobulks_20260203.zip"
CYTOKINES = ["PBS", "IFN-gamma", "IFN-beta", "IFN-alpha1", "TNF-alpha", "IL-1-beta", "IL-6", "IL-10", "GM-CSF",
             "IL-4"]


def _range(a, b, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(URL, headers={"Range": "bytes=%d-%d" % (a, b)})
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def _size():
    req = urllib.request.Request(URL, method="HEAD")
    return int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])


def _entries():
    size = _size()
    tail = _range(size - 200_000, size - 1)
    i = tail.rfind(b"PK\x06\x06")                                   # zip64 end of central directory
    cd_size, cd_off = struct.unpack("<QQ", tail[i + 40:i + 56])
    cd = _range(cd_off, cd_off + cd_size - 1)
    p, out = 0, []
    while cd[p:p + 4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", cd[p + 10:p + 12])
        comp, unc = struct.unpack("<II", cd[p + 20:p + 28])
        n, e, c = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + n].decode()
        extra, k = cd[p + 46 + n:p + 46 + n + e], 0
        while k < len(extra):                                       # zip64 extra field: 64-bit sizes / offset
            hid, hl = struct.unpack("<HH", extra[k:k + 4])
            if hid == 1:
                vals, vi = struct.unpack("<" + "Q" * (hl // 8), extra[k + 4:k + 4 + hl]), 0
                if unc == 0xFFFFFFFF:
                    unc, vi = vals[vi], vi + 1
                if comp == 0xFFFFFFFF:
                    comp, vi = vals[vi], vi + 1
                if off == 0xFFFFFFFF:
                    off, vi = vals[vi], vi + 1
            k += 4 + hl
        out.append((name, meth, off, comp, unc))
        p += 46 + n + e + c
    return out


def list_():
    os.makedirs(D, exist_ok=True)
    ent = _entries()
    with open(os.path.join(D, "listing.tsv"), "w", encoding="utf-8") as fh:
        fh.write("name\tmethod\toffset\tcomp\tunc\n")
        for row in ent:
            fh.write("\t".join(map(str, row)) + "\n")
    print(len(ent), "entries;", sum(r[3] for r in ent) / 1e9, "GB compressed")


def _cytokine(name):
    base = os.path.basename(name)
    if not base.startswith("pbulk_") or not base.endswith(".h5ad"):
        return None
    return base[:-5].rsplit("_", 1)[-1]


def get():
    import csv
    out = os.path.join(D, "pseudobulk")
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(D, "listing.tsv"), encoding="utf-8"), delimiter="\t"))
    want = [r for r in rows if _cytokine(r["name"]) in CYTOKINES and "/PBS_by_well/" not in "/" + r["name"]]
    print(len(want), "files,", sum(int(r["comp"]) for r in want) / 1e9, "GB", flush=True)
    for r in want:
        dst = os.path.join(out, os.path.basename(r["name"]))
        if os.path.exists(dst):
            continue
        off, comp, meth = int(r["offset"]), int(r["comp"]), int(r["method"])
        lh = _range(off, off + 29)
        ln, le = struct.unpack("<HH", lh[26:30])
        start = off + 30 + ln + le
        d = _range(start, start + comp - 1)
        d = d if meth == 0 else zlib.decompress(d, -15)
        with open(dst + ".part", "wb") as fh:
            fh.write(d)
        os.replace(dst + ".part", dst)
        print("got", os.path.basename(dst), len(d), flush=True)


if __name__ == "__main__":
    dict(list=list_, get=get)[sys.argv[1]]()
