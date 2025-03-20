import numpy
import pandas
import pytest

from livelike import multi

YEAR = 2019


class TestMultiExtractPumsSegmentIds:
    def setup_method(self):
        self.puma_fips = [
            pytest.data_0100100["puma_fips"],
            pytest.data_0100200["puma_fips"],
        ]
        self.query = "ESR=1&NAICSP=6111&OCCP=2300:2320&WKHP=40:999"
        self.key = pytest.CENSUSAPIKEY

        self.known_esr = "1"
        self.known_naicsp = "6111"
        self.known_occp_low = "2300"
        self.known_occp_high = "2320"
        self.known_wkhp_low = "40"
        self.known_wkhp_high = "999"
        self.known_year = YEAR

    def test_level_invalid(self):
        with pytest.raises(
            AssertionError,
            match="Argument ``level`` must be one of: ``person``, ``household``.",
        ):
            multi.extract_pums_segment_ids_multi("", "pets", "")

    @pytest.skip_if_no_censusapikey
    def test_person(self):
        observed = multi.extract_pums_segment_ids_multi(
            self.puma_fips,
            "person",
            self.query,
            year=YEAR,
            append_year=False,
            censusapikey=self.key,
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
            <= observed["OCCP"].unique().min()
            <= observed["OCCP"].unique().max()
            <= self.known_occp_high
        )
        assert (
            self.known_wkhp_low
            <= observed["WKHP"].unique().min()
            <= observed["WKHP"].unique().max()
            <= self.known_wkhp_high
        )

    @pytest.skip_if_no_censusapikey
    def test_household(self):
        observed = multi.extract_pums_segment_ids_multi(
            self.puma_fips,
            "household",
            self.query,
            year=YEAR,
            append_year=True,
            censusapikey=self.key,
        )

        assert isinstance(observed, pandas.DataFrame)
        numpy.testing.assert_array_equal(
            observed.columns,
            ["SERIALNO", "ESR", "NAICSP", "OCCP", "WKHP", "year"],
        )
        assert (observed["ESR"].unique() == self.known_esr).all()
        assert (observed["NAICSP"].unique() == self.known_naicsp).all()
        assert (
            self.known_occp_low
            <= observed["OCCP"].unique().min()
            <= observed["OCCP"].unique().max()
            <= self.known_occp_high
        )
        assert (
            self.known_wkhp_low
            <= observed["WKHP"].unique().min()
            <= observed["WKHP"].unique().max()
            <= self.known_wkhp_high
        )
        assert (observed["year"].unique() == self.known_year).all()


class TestMultiExtractPumsDescriptors:
    def setup_method(self):
        self.puma_fips = [
            pytest.data_0100100["puma_fips"],
            pytest.data_0100200["puma_fips"],
        ]
        self.features_p = ["ESR"]
        self.features_h = numpy.array(["YBL"])
        self.key = pytest.CENSUSAPIKEY

        self.known_year = YEAR

    def test_level_invalid(self):
        with pytest.raises(
            AssertionError,
            match="Argument ``level`` must be one of: ``person``, ``household``.",
        ):
            multi.extract_pums_descriptors_multi("", "pets", "")

    @pytest.skip_if_no_censusapikey
    def test_person(self):
        observed = multi.extract_pums_descriptors_multi(
            self.puma_fips,
            "person",
            self.features_p,
            year=YEAR,
            append_year=False,
            censusapikey=self.key,
        )

        assert isinstance(observed, pandas.DataFrame)
        numpy.testing.assert_array_equal(
            observed.columns,
            ["SERIALNO", "SPORDER", "ESR", "p_id"],
        )

    @pytest.skip_if_no_censusapikey
    def test_household(self):
        observed = multi.extract_pums_descriptors_multi(
            self.puma_fips,
            "household",
            self.features_h,
            year=YEAR,
            append_year=True,
            censusapikey=self.key,
        )

        assert isinstance(observed, pandas.DataFrame)
        numpy.testing.assert_array_equal(observed.columns, ["SERIALNO", "YBL", "year"])
        assert (observed["year"].unique() == self.known_year).all()
