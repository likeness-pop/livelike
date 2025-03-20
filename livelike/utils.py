"""Utility functions for ``livelike``."""

import pathlib

__all__ = [
    "clear_acs_cache",
]


def _ensure_cache_folder_input(cache_folder: str | pathlib.Path) -> pathlib.Path:
    """Helper to prep cache directory."""
    if not isinstance(cache_folder, str | pathlib.Path):
        raise TypeError(
            "Check input type for ``cache_folder`` and consult documentation."
        )

    cache_folder = pathlib.Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)

    return cache_folder


def clear_acs_cache(
    cache_folder: None | str | pathlib.Path = "./livelike_acs_cache",
):
    """Clears the ``cache_folder`` directory.

    Parameters
    ----------
    cache_folder : str | pathlib.Path = './livelike_acs_cache'
        The cache folder to store queried ACS inputs. This default value
        creates the folder in the current working directory if not present.
    """

    acs_cache_dir = _ensure_cache_folder_input(cache_folder)
    cached_files = list(acs_cache_dir.iterdir())

    if len(cached_files) > 0:
        [f.unlink() for f in cached_files if f.name != "temp.txt"]
