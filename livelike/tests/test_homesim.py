import re

import numpy
import pandas
import pytest

from livelike import homesim


class TestTRS:
    def test_frac_error(self):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "``fracs`` parameter type expected numpy.ndarray or pandas.Series. "
                "Consult ``livelike.homesim.trs()`` documentation. "
                "Input type: 'DataFrame'"
            ),
        ):
            homesim.trs(pandas.DataFrame())

    def test_subfrac_pd_null(self):
        known = 0
        observed = homesim._sum_rnd_sub_frc_remainders(pandas.Series([]))
        assert observed == known
        assert isinstance(observed, int)

    def test_subfrac_np_null(self):
        known = numpy.int64(0)
        observed = homesim._sum_rnd_sub_frc_remainders(numpy.array([]))
        assert observed == known
        assert isinstance(observed, numpy.int64)

    def test_subfrac_error(self):
        with pytest.raises(
            AttributeError,
            match="'str' object has no attribute 'sum'",
        ):
            homesim._sum_rnd_sub_frc_remainders("can't pass strings here.")


class TestSynthesize:
    def setup_method(self):
        self.st_fips = pytest.data_0100100["st_fips"]

        self.synthesize_args = (
            pytest.data_0100100["almat"],
            pytest.data_0100100["est_ind"],
            pytest.data_0100100["est_g2"],
            pytest.data_0100100["sporder"],
        )
        self.nsims = 2
        self.synthesize_kwargs = {"nsim": self.nsims, "random_state": 1}

    def test_no_hht_x_hhs_cols(self):
        _est_ind = pytest.data_0100100["est_ind"].copy()
        hht_x_hhsize_cols = list(_est_ind.filter(regex="^hht.*hhsize"))

        with pytest.raises(
            ValueError,
            match=(
                "Individual constraints must contain household "
                "type by household size columns."
            ),
        ):
            homesim.synthesize(
                pytest.data_0100100["almat"],
                _est_ind[_est_ind.columns.drop(hht_x_hhsize_cols)],
                pytest.data_0100100["est_g2"],
                pytest.data_0100100["sporder"],
            )

    def test_longform(self):
        known_sims = self.nsims
        known_sims_eq_count = True
        known_in_state = True

        observed = homesim.synthesize(
            pytest.data_0100100["almat"],
            pytest.data_0100100["est_ind"],
            pytest.data_0100100["est_g2"],
            pytest.data_0100100["sporder"],
            **self.synthesize_kwargs,
            longform=True,
        )

        assert isinstance(observed, pandas.DataFrame)
        observed_sims = observed["sim"].nunique()
        assert observed_sims == known_sims
        observed_sims_eq_count = observed.groupby("sim")["count"].sum().nunique() == 1
        assert observed_sims_eq_count == known_sims_eq_count
        observed_in_state = observed["geoid"].str.startswith(self.st_fips).all()
        assert observed_in_state == known_in_state

    def test_shortform(self):
        known_sims = self.nsims
        known_in_state = True

        observed = homesim.synthesize(
            pytest.data_0100100["almat"],
            pytest.data_0100100["est_ind"],
            pytest.data_0100100["est_g2"],
            pytest.data_0100100["sporder"],
            longform=False,
            **self.synthesize_kwargs,
        )

        assert isinstance(observed, list)

        for obs in observed:
            assert isinstance(obs, pandas.DataFrame)
            assert obs.shape == pytest.data_0100100["almat"].shape
            obs_in_state = obs.columns.str.startswith(self.st_fips).all()
            assert obs_in_state == known_in_state

        assert len(observed) == known_sims

        pandas.testing.assert_frame_equal(
            homesim._make_longform(observed),
            homesim.synthesize(
                pytest.data_0100100["almat"],
                pytest.data_0100100["est_ind"],
                pytest.data_0100100["est_g2"],
                pytest.data_0100100["sporder"],
                longform=True,
                **self.synthesize_kwargs,
            ),
        )


class TestGenerateRandomStates:
    def test_n2(self):
        n = 2
        known_shape = (n,)
        known_type = numpy.int64
        observed = homesim.generate_random_states(n=n)
        observed_shape = observed.shape
        observed_type = observed.dtype
        assert observed_shape == known_shape
        assert observed_type == known_type

    def test_n0(self):
        n = 0
        known_shape = (n,)
        known_type = numpy.int64
        observed = homesim.generate_random_states(n=n)
        observed_shape = observed.shape
        observed_type = observed.dtype
        assert observed_shape == known_shape
        assert observed_type == known_type

    def test_neg_n(self):
        with pytest.raises(ValueError, match="negative dimensions are not allowed"):
            homesim.generate_random_states(n=-1)

    def test_neg_seed(self):
        with pytest.raises(
            ValueError,
            match=re.escape("Seed must be between 0 and 2**32 - 1"),
        ):
            homesim.generate_random_states(seed=-1)
