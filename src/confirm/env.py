"""Environment loading for CONFIRM agent integrations."""

from __future__ import annotations

from pathlib import Path

_LOADED = False


def load_env(path: str | Path | None = None) -> None:
    """Load ``.env`` into ``os.environ`` once."""

    global _LOADED
    if _LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _LOADED = True
        return
    load_dotenv(dotenv_path=path, override=False)
    _LOADED = True
