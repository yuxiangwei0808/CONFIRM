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

        load_dotenv(dotenv_path=path, override=False)
    except Exception:
        # TODO(confirm): Surface dotenv import/load failures through structured
        # agent diagnostics if the CLI grows a diagnostic mode.
        pass
    _LOADED = True
