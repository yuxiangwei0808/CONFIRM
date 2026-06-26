"""Generate all paper figures and LaTeX tables from existing result artifacts."""

from __future__ import annotations

import gen_coverage_fcr
import gen_multillm
import gen_neuroclaw
import gen_tables


def main() -> int:
    paths = []
    paths.extend(gen_coverage_fcr.generate())
    paths.extend(gen_neuroclaw.generate())
    paths.extend(gen_multillm.generate())
    paths.extend(gen_tables.generate())
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
