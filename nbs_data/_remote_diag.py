#!/usr/bin/env python
"""READ-ONLY diagnostics: fix HDF5 char decode + resolve file-count vs numOfSub."""
import sys, glob, os, re
import numpy as np
import h5py


def inspect_name_refs(path, k=3):
    print(f"\n=== raw name-ref inspection: {path} ===")
    with h5py.File(path, "r") as h:
        files = h["files"]
        namerefs = np.array(files["name"]).ravel()
        print(" n name refs:", namerefs.size, "ref dtype:", namerefs.dtype)
        for i, r in enumerate(namerefs[:k]):
            t = h[r]
            a = np.array(t)
            print(f" [{i}] target shape={a.shape} dtype={a.dtype} min={a.min()} max={a.max()}")
            # try several decodings
            flat_c = a.ravel(order="C")
            flat_f = a.ravel(order="F")
            sC = "".join(chr(int(c)) for c in flat_c if 0 < int(c) < 0x110000)
            sF = "".join(chr(int(c)) for c in flat_f if 0 < int(c) < 0x110000)
            print("     C-order:", repr(sC[:120]))
            print("     F-order:", repr(sF[:120]))
            # if 2D, maybe each ROW is a char path (axis0=chars). try per-column join
            if a.ndim == 2:
                # GIFT: char matrix rows=files cols=chars OR transposed. Show row0 & col0.
                row0 = "".join(chr(int(c)) for c in a[0].ravel() if 0 < int(c) < 0x110000)
                col0 = "".join(chr(int(c)) for c in a[:, 0].ravel() if 0 < int(c) < 0x110000)
                print("     row0:", repr(row0[:120]))
                print("     col0:", repr(col0[:120]))


def resolve_counts(icadir, pattern):
    files = glob.glob(os.path.join(icadir, pattern))
    idxs = []
    for f in files:
        m = re.search(r"_ica_br(\d+)\.mat$", f)
        if m:
            idxs.append(int(m.group(1)))
    idxs.sort()
    print(f"\n=== {icadir} pattern={pattern} ===")
    print(" n files:", len(files), "| br index min/max:", (idxs[0], idxs[-1]) if idxs else None,
          "| n distinct idx:", len(set(idxs)))
    if idxs:
        full = set(range(idxs[0], idxs[-1] + 1))
        missing = sorted(full - set(idxs))
        print(" contiguous?", len(missing) == 0, "| n missing in range:", len(missing),
              "| first missing:", missing[:5])


def main():
    what = sys.argv[1]
    if what == "names":
        inspect_name_refs("/data/qneuromark/Results/ICA/JH/JHSubject.mat")
        inspect_name_refs("/data/qneuromark/Results/ICA/BSNIP/PC/BSNIPSubject.mat")
    elif what == "counts":
        resolve_counts("/data/qneuromark/Results/ICA/JH", "JH_ica_br*.mat")
        resolve_counts("/data/qneuromark/Results/ICA/BSNIP/PC", "BSNIP_ica_br*.mat")
        resolve_counts("/data/qneuromark/Results/ICA/BSNIP/allPR", "BSNIP_allPR_ica_br*.mat")
        resolve_counts("/data/qneuromark/Results/ICA/BSNIP2", "BSNIP2_ica_br*.mat")


if __name__ == "__main__":
    main()
