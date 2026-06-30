#!/usr/bin/env python
"""READ-ONLY remote probe for psychosis cohorts (BSNIP/BSNIP2/JH/PANStudy).

Decodes GIFT subject-order .mat files (to recover the ICA-file <-> subject-ID join),
inspects compSet.tc shapes, and dumps candidate demographics. No writes anywhere.

Usage: python _remote_probe_psychosis.py <what>
  what in {bsnip_subj, jh_meta, pan_meta, bsnip2_meta}
"""
import sys, os, json
import numpy as np


def load_mat_any(path):
    """Return dict of top-level vars, handling both old and v7.3 (HDF5) mat."""
    import scipy.io as sio
    try:
        return ("scipy", sio.loadmat(path, squeeze_me=False, struct_as_record=False))
    except NotImplementedError:
        import h5py
        return ("h5py", h5py.File(path, "r"))


def h5_str(h5file, ref_or_ds):
    """Decode a MATLAB char array stored in HDF5 into a python str."""
    import h5py
    obj = h5file[ref_or_ds] if isinstance(ref_or_ds, (bytes, h5py.h5r.Reference)) else ref_or_ds
    arr = np.array(obj).ravel()
    try:
        return "".join(chr(int(c)) for c in arr if int(c) != 0)
    except Exception:
        return repr(arr[:10])


def dump_subject_mat(path):
    print(f"\n=== subject-order mat: {path} ===")
    kind, m = load_mat_any(path)
    print("loader:", kind)
    if kind == "scipy":
        for k, v in m.items():
            if k.startswith("__"):
                continue
            print(" var", k, type(v))
            # Common GIFT: 'files' cell array of paths, or struct array w/ .name
            try:
                arr = np.array(v)
                flat = arr.ravel()
                print("   shape", arr.shape, "n", flat.size)
                for e in flat[:5]:
                    # unwrap matlab cell / char
                    val = e
                    if isinstance(val, np.ndarray):
                        val = val.ravel()
                        if val.size == 1:
                            val = val[0]
                    print("     >", str(val)[:160])
            except Exception as ex:
                print("   (decode err)", repr(ex))
    else:
        import h5py
        with m as h:
            def walk(g, p=""):
                for k in g:
                    it = g[k]
                    if isinstance(it, h5py.Group):
                        walk(it, p + k + "/")
                    else:
                        print(" ds", p + k, it.shape, it.dtype)
                        # try string decode + first refs
                        try:
                            data = it[()]
                            flat = np.array(data).ravel()
                            if it.dtype == object:  # refs
                                for r in flat[:5]:
                                    print("    str>", h5_str(h, r)[:160])
                            elif np.issubdtype(np.array(flat[:1]).dtype, np.number) and flat.size < 4000:
                                s = "".join(chr(int(c)) for c in flat if 9 < int(c) < 127)
                                if len(s) > 3:
                                    print("    asstr>", s[:200])
                        except Exception:
                            pass
            walk(h)


def _h5_char(h, obj):
    a = np.array(obj).ravel()
    return "".join(chr(int(c)) for c in a if 0 < int(c) < 0x110000).replace("\x00", "").strip()


def decode_subject_paths(path, n_show=8):
    """GIFT *Subject.mat: var 'files' is a struct array w/ 'name' field = subject file path.

    Handles HDF5 (v7.3) and old scipy format. Returns subject-order path strings
    (one representative path per subject -> reveals subject ID + group folder).
    """
    print(f"\n=== decode subject paths: {path} ===")
    import scipy.io as sio
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return _decode_scipy(m, n_show)
    except NotImplementedError:
        pass
    import h5py
    out = []
    with h5py.File(path, "r") as h:
        top = [k for k in h.keys() if k != "#refs#"]
        print(" top vars:", top)
        if "numOfSub" in h:
            print(" numOfSub =", np.array(h["numOfSub"]).ravel())
        if "files" not in h:
            print(" NO 'files' var"); return out
        files = h["files"]
        print(" files type:", type(files), "keys:", list(files.keys()) if hasattr(files, "keys") else "(ds)")
        # files is a group representing struct array; field 'name' holds refs
        if hasattr(files, "keys") and "name" in files:
            namerefs = np.array(files["name"]).ravel()
            print(" #name entries:", namerefs.size)
            for r in namerefs:
                try:
                    out.append(_h5_char(h, h[r]))
                except Exception:
                    pass
        else:
            # fallback: follow every ref under files
            def collect(g):
                for k in g:
                    it = g[k]
                    if isinstance(it, h5py.Group):
                        collect(it)
                    else:
                        for r in np.array(it).ravel():
                            try:
                                if isinstance(r, h5py.h5r.Reference):
                                    out.append(_h5_char(h, h[r]))
                            except Exception:
                                pass
            collect(files)
    _report(out, n_show)
    return out


def _decode_scipy(m, n_show):
    """Old-format GIFT Subject.mat via scipy: m['files'] is struct array w/ .name"""
    out = []
    print(" loader: scipy(old)")
    if "numOfSub" in m:
        print(" numOfSub =", m["numOfSub"])
    files = m.get("files")
    if files is None:
        print(" NO 'files'"); return out
    files = np.atleast_1d(files)
    for f in files.ravel():
        nm = getattr(f, "name", None)
        if nm is None:
            continue
        nm = np.atleast_1d(nm).ravel()
        # name may itself be a char matrix (multi-session); take first row/string
        for row in nm:
            out.append(str(row))
    _report(out, n_show)
    return out


def _report(out, n_show):
    print(f" decoded {len(out)} strings; first {n_show}:")
    for s in out[:n_show]:
        print("   |", str(s)[:220])
    if out:
        print(" last:")
        print("   |", str(out[-1])[:220])


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "bsnip_subj"
    if what == "bsnip_subj":
        dump_subject_mat("/data/qneuromark/Results/ICA/BSNIP/PC/BSNIPSubject.mat")
    elif what == "bsnip_decode":
        decode_subject_paths("/data/qneuromark/Results/ICA/BSNIP/PC/BSNIPSubject.mat", n_show=12)
        decode_subject_paths("/data/qneuromark/Results/ICA/BSNIP/allPR/BSNIP_allPRSubject.mat", n_show=12)
    elif what == "jh_decode":
        decode_subject_paths("/data/qneuromark/Results/ICA/JH/JHSubject.mat", n_show=12)
    elif what == "bsnip2_decode":
        import glob
        cands = glob.glob("/data/qneuromark/Results/ICA/BSNIP2/*ubject*.mat")
        print("BSNIP2 subject-order candidates:", cands)
        for c in cands:
            decode_subject_paths(c, n_show=12)
    else:
        print("unknown what:", what)


if __name__ == "__main__":
    main()
