#######################################################################
# Healthy, redundant testing of ``test_supplementary_api_wrappers.py``
#######################################################################


import numpy
import pandas
import pytest

from livelike import acs


@pytest.skip_if_no_censusapikey
def test_extract_from_census_microdata_api():
    year = 2019
    rq = acs.build_census_microdata_api_base_request("household", year=year)
    rq += acs.build_census_microdata_api_geo_request("0100100", year=year)

    observed = acs.extract_from_census_microdata_api(
        rq, censusapikey=pytest.CENSUSAPIKEY
    )

    assert isinstance(observed, pandas.DataFrame)
    pandas.testing.assert_index_equal(observed.columns, pandas.Index(["SERIALNO"]))


def test_extract_from_census_microdata_api_error():
    with pytest.raises(ValueError, match="Supported years are 2016 - 2019 and 2023+."):
        (
            acs.build_census_microdata_api_base_request("household", year=2015)
            + acs.build_census_microdata_api_geo_request("0100100", year=2015)
        )


class TestExtractPumsSegmentIds:
    def setup_method(self):
        self.puma_fips = pytest.data_0100100["puma_fips"]
        self.query = "ESR=1&NAICSP=6111&OCCP=2300:2320&WKHP=40:999"
        self.key = pytest.CENSUSAPIKEY

        self.known_esr = "1"
        self.known_naicsp = "6111"
        self.known_occp_low = "2300"
        self.known_occp_high = "2320"
        self.known_wkhp_low = "40"
        self.known_wkhp_high = "999"

    def test_level_invalid(self):
        with pytest.raises(
            AssertionError,
            match="Argument ``level`` must be one of: ``person``, ``household``.",
        ):
            acs.extract_pums_segment_ids("", "pets", "")

    @pytest.skip_if_no_censusapikey
    def test_person(self):
        observed = acs.extract_pums_segment_ids(
            self.puma_fips, "person", self.query, censusapikey=self.key
        )

        assert isinstance(observed, pandas.DataFrame)
        numpy.testing.assert_array_equal(
            observed.columns,
            ["SERIALNO", "SPORDER", "ESR", "NAICSP", "OCCP", "WKHP", "p_id"],
        )
        assert (observed["ESR"].unique() == self.known_esr).all()
        assert (observed["NAICSP"].unique() == self.known_naicsp).all()
        assert (
            self.known_occp_low
            <= observed["OCCP"].min()
            <= observed["OCCP"].max()
            <= self.known_occp_high
        )
        assert (
            self.known_wkhp_low
            <= observed["WKHP"].min()
            <= observed["WKHP"].max()
            <= self.known_wkhp_high
        )

    @pytest.skip_if_no_censusapikey
    def test_household(self):
        observed = acs.extract_pums_segment_ids(
            self.puma_fips, "household", self.query, censusapikey=self.key
        )

        assert isinstance(observed, pandas.DataFrame)
        numpy.testing.assert_array_equal(
            observed.columns,
            ["SERIALNO", "ESR", "NAICSP", "OCCP", "WKHP"],
        )
        assert (observed["ESR"].unique() == self.known_esr).all()
        assert (observed["NAICSP"].unique() == self.known_naicsp).all()
        assert (
            self.known_occp_low
            <= observed["OCCP"].min()
            <= observed["OCCP"].max()
            <= self.known_occp_high
        )
        assert (
            self.known_wkhp_low
            <= observed["WKHP"].min()
            <= observed["WKHP"].max()
            <= self.known_wkhp_high
        )


class TestExtractPumsDescriptors:
    def setup_method(self):
        self.puma_fips = pytest.data_0100100["puma_fips"]
        self.features_p = ["ESR"]
        self.features_h = numpy.array(["YBL", "HINCP"])
        self.key = pytest.CENSUSAPIKEY

    def test_level_invalid(self):
        with pytest.raises(
            AssertionError,
            match="Argument ``level`` must be one of: ``person``, ``household``.",
        ):
            acs.extract_pums_segment_ids("", "pets", "")

    @pytest.skip_if_no_censusapikey
    def test_person(self):
        observed = acs.extract_pums_descriptors(
            self.puma_fips, "person", self.features_p, censusapikey=self.key
        )

        assert isinstance(observed, pandas.DataFrame)
        numpy.testing.assert_array_equal(
            observed.columns,
            ["SERIALNO", "SPORDER", "ESR", "p_id"],
        )

    @pytest.skip_if_no_censusapikey
    def test_household(self):
        observed = acs.extract_pums_descriptors(
            self.puma_fips, "household", self.features_h, censusapikey=self.key
        )

        assert isinstance(observed, pandas.DataFrame)
        numpy.testing.assert_array_equal(
            observed.columns, ["SERIALNO", "YBL", "HINCP", "ADJINC"]
        )

    @pytest.skip_if_no_censusapikey
    def test_person_sample(self):
        observed = acs.extract_pums_descriptors(
            self.puma_fips,
            "person",
            self.features_p,
            sample_weights=True,
            censusapikey=self.key,
        )

        assert isinstance(observed, pandas.DataFrame)
        numpy.testing.assert_array_equal(
            observed.columns,
            ["SERIALNO", "SPORDER", "PWGTP", "ESR", "p_id"],
        )

    @pytest.skip_if_no_censusapikey
    def test_household_sample(self):
        observed = acs.extract_pums_descriptors(
            self.puma_fips,
            "household",
            self.features_h,
            sample_weights=True,
            censusapikey=self.key,
        )

        assert isinstance(observed, pandas.DataFrame)
        numpy.testing.assert_array_equal(
            observed.columns, ["SERIALNO", "WGTP", "YBL", "HINCP", "ADJINC"]
        )
