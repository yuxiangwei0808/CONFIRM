#!/usr/bin/env python
"""READ-ONLY probe: BSNIP1 subject_information_BSNIP.mat (demographics/dx source)."""
import numpy as np

P = "/data/mialab/users/zfu/Matlab/GSU/Neuromark/Results/Data_info/BSNIP/subject_information_BSNIP.mat"


def main():
    import scipy.io as sio
    try:
        m = sio.loadmat(P, squeeze_me=True, struct_as_record=False)
        print("loader: scipy")
    except NotImplementedError:
        import h5py
        with h5py.File(P, "r") as h:
            print("loader: h5py; vars:", [k for k in h if k != "#refs#"])
        return
    for k, v in m.items():
        if k.startswith("__"):
            continue
        a = np.atleast_1d(np.asarray(v, dtype=object) if v is None else v)
        print(f" var {k}: type={type(v).__name__} shape={getattr(v,'shape',None)}")
    # SubjectID
    sid = np.atleast_1d(m["SubjectID"])
    print("\nSubjectID n:", sid.size, "| samples:", [str(x) for x in np.ravel(sid)[:5]])
    # diagnosis
    dx = m["diagnosis"]
    dx = np.asarray(dx, dtype=object)
    print("diagnosis shape:", dx.shape)
    print("diagnosis[:5]:", [list(map(str, np.ravel(r))) for r in dx[:5]])
    col2 = [str(np.ravel(r)[1]) if np.ravel(r).size > 1 else "" for r in dx]
    from collections import Counter
    print("diagnosis col2 counts:", dict(Counter(col2)))
    # site
    st = np.atleast_1d(m["site"])
    from collections import Counter as C2
    print("site counts:", dict(C2(str(x) for x in np.ravel(st))))
    # Num_scores
    ns = np.asarray(m["Num_scores"])
    print("Num_scores shape:", ns.shape, "| col2(age) sample:", ns[:5, 1].tolist() if ns.ndim == 2 else ns[:5])
    # sex? look for a gender var
    print("\nALL VARS:", [k for k in m if not k.startswith("__")])


if __name__ == "__main__":
    main()
