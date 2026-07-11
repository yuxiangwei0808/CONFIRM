"""Small progress helpers for benchmark runners."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def iter_progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "progress",
    enabled: bool = True,
    unit: str = "item",
) -> Iterator[T]:
    """Yield items with a tqdm progress bar when available.

    The fallback avoids adding a hard dependency on tqdm while still producing
    visible progress in logs.
    """

    if not enabled:
        yield from iterable
        return

    try:
        from tqdm.auto import tqdm  # type: ignore

        yield from tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)
        return
    except Exception:
        pass

    for index, item in enumerate(iterable, start=1):
        yield item
        denominator = total if total is not None else "?"
        print(f"[progress] {desc} {index}/{denominator}", flush=True)
