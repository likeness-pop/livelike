import pathlib
import shutil

import pytest

from livelike.utils import (
    _ensure_cache_folder_input,
    clear_acs_cache,
)


class TestCacheUtils:
    def test_invalid_cache_input(self):
        with pytest.raises(
            TypeError,
            match="Check input type for ``cache_folder`` and consult documentation.",
        ):
            _ensure_cache_folder_input(["this_is_a_list"])

    def test_valid_cache_input(self):
        _temp_name = "temp_cache_for_test_1"
        known = pathlib.Path(_temp_name)
        observed = _ensure_cache_folder_input(_temp_name)

        assert observed.is_dir()
        assert observed == known
        shutil.rmtree(known)

    def test_clear_cache(self):
        directory = _ensure_cache_folder_input("temp_cache_for_test_2")
        test_file = directory / "temp.txt"
        test_file.touch()

        clear_acs_cache(directory)

        known = True
        observed = test_file.exists() and test_file.is_file()

        assert observed == known
        shutil.rmtree(directory)
