import pytest

from livelike import acs, multi

YEAR = 2019


class TestMultiMakePUMAs:
    def setup_method(self):
        self.puma_fips = [
            pytest.data_0100100["puma_fips"],
            pytest.data_0100200["puma_fips"],
        ]
        self.key = pytest.CENSUSAPIKEY

    @pytest.skip_if_no_censusapikey
    def test_defaults(self):
        known_keys = self.puma_fips

        observed = multi.make_pumas(self.puma_fips, censusapikey=self.key)

        for known_key in known_keys:
            assert known_key in observed
            assert isinstance(observed[known_key], acs.puma)

    @pytest.skip_if_no_censusapikey
    def test_cache_append_year(self):
        known_keys = [f"{pf}_{YEAR}" for pf in self.puma_fips]

        observed = multi.make_pumas(
            self.puma_fips,
            append_year=True,
            censusapikey=self.key,
            cache=True,
            cache_folder=pytest.testing_cache / "multi_make_puma",
        )

        for known_key in known_keys:
            assert known_key in observed
            assert isinstance(observed[known_key], acs.puma)
