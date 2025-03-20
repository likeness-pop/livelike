import io

import pandas
import pytest

from livelike import est, homesim


class TestTabulateBySerial:
    def setup_method(self):
        self.st_fips = (pytest.data_0100100["st_fips"],)

        self.synthesize_args = (
            pytest.data_0100100["almat"],
            pytest.data_0100100["est_ind"],
            pytest.data_0100100["est_g2"],
            pytest.data_0100100["sporder"],
        )
        self.nsims = 2
        self.random_state = 1
        self.synthesize_kwargs = {"nsim": self.nsims, "random_state": self.random_state}

        self.serialno = pytest.data_0100100["pums_seg_ids"]["SERIALNO"]

    def test_level_invalid(self):
        with pytest.raises(
            AssertionError,
            match="Argument ``level`` must be one of: ``person``, ``household``.",
        ):
            est.tabulate_by_serial(pandas.DataFrame(), pandas.DataFrame(), "pets")

    def test_serial_invalid(self):
        with pytest.raises(
            TypeError,
            match=(
                "``serial`` must be ``numpy.ndarray`` or ``pandas.Series``. Check type."
            ),
        ):
            est.tabulate_by_serial(pandas.DataFrame(), [1, 2, 3], "person")

    def test_person(self):
        hs = homesim.synthesize(*self.synthesize_args, **self.synthesize_kwargs)

        known_sims = self.nsims
        known_in_state = True
        observed = est.tabulate_by_serial(hs, self.serialno, "person")

        assert observed.index.names == ["sim", "geoid"]
        observed_sims = observed.index.get_level_values("sim").nunique()
        assert observed_sims == known_sims
        observed_in_state = (
            observed.index.get_level_values("geoid").str.startswith(self.st_fips).all()
        )
        assert observed_in_state == known_in_state

    def test_household(self):
        hs = homesim.synthesize(*self.synthesize_args, **self.synthesize_kwargs)
        hs["year"] = "2019"

        known_sims = self.nsims
        known_in_state = True
        observed = est.tabulate_by_serial(hs, self.serialno.to_numpy(), "household")

        assert observed.index.names == ["sim", "geoid", "year"]
        observed_sims = observed.index.get_level_values("sim").nunique()
        assert observed_sims == known_sims
        observed_in_state = (
            observed.index.get_level_values("geoid").str.startswith(self.st_fips).all()
        )
        assert observed_in_state == known_in_state


class TestTabulateByCount:
    def setup_method(self):
        self.st_fips = (pytest.data_0100100["st_fips"],)

        self.synthesize_args = (
            pytest.data_0100100["almat"],
            pytest.data_0100100["est_ind"],
            pytest.data_0100100["est_g2"],
            pytest.data_0100100["sporder"],
        )
        self.nsims = 2
        self.random_state = 1
        self.synthesize_kwargs = {"nsim": self.nsims, "random_state": self.random_state}

        self.population = pytest.data_0100100["est_ind"]["population"]
        no_name_pop = self.population.copy()
        no_name_pop.name = None
        self.no_name_pop = no_name_pop

    def test_default(self):
        hs = homesim.synthesize(*self.synthesize_args, **self.synthesize_kwargs)

        known_sims = self.nsims
        known_in_state = True
        known_name = "population"

        observed = est.tabulate_by_count(hs, self.population)

        assert observed.index.names == ["sim", "geoid"]
        observed_sims = observed.index.get_level_values("sim").nunique()
        assert observed_sims == known_sims
        observed_in_state = (
            observed.index.get_level_values("geoid").str.startswith(self.st_fips).all()
        )
        assert observed_in_state == known_in_state
        observed_name = observed.name
        assert observed_name == known_name

    def test_label(self):
        hs = homesim.synthesize(*self.synthesize_args, **self.synthesize_kwargs)

        known_sims = self.nsims
        known_in_state = True
        known_name = "density"

        observed = est.tabulate_by_count(hs, self.no_name_pop, label="density")

        assert observed.index.names == ["sim", "geoid"]
        observed_sims = observed.index.get_level_values("sim").nunique()
        assert observed_sims == known_sims
        observed_in_state = (
            observed.index.get_level_values("geoid").str.startswith(self.st_fips).all()
        )
        assert observed_in_state == known_in_state
        observed_name = observed.name
        assert observed_name == known_name

    def test_no_name(self):
        hs = homesim.synthesize(*self.synthesize_args, **self.synthesize_kwargs)

        known_sims = self.nsims
        known_in_state = True
        known_name = "count"
        hs["year"] = "2019"

        observed = est.tabulate_by_count(hs, self.no_name_pop, label=None)

        assert observed.index.names == ["sim", "geoid", "year"]
        observed_sims = observed.index.get_level_values("sim").nunique()
        assert observed_sims == known_sims
        observed_in_state = (
            observed.index.get_level_values("geoid").str.startswith(self.st_fips).all()
        )
        assert observed_in_state == known_in_state
        observed_name = observed.name
        assert observed_name == known_name


def test_to_prop():
    df1 = pandas.DataFrame([1, 1, 3])
    df2 = pandas.DataFrame([1, 2, 3, 4])

    known_shape = max(df1.shape[0], df2.shape[0])
    known_leq_1 = True

    observed = est.to_prop(df1[0], df2[0])

    observed_shape = observed.shape[0]
    assert observed_shape == known_shape
    observed_leq_1 = observed[~observed.isna()].all() <= 1
    assert observed_leq_1 == known_leq_1


class TestMonteCarloEstimate:
    def setup_method(self):
        self.df = pandas.DataFrame(
            {
                "val": [1, 1, 3, 2],
                "sim": [0, 1, 0, 1],
                "geoid": ["g1", "g1", "g2", "g2"],
                "year": [2019] * 4,
            }
        ).set_index(["sim", "geoid", "year"])

    def test_default(self):
        known = pandas.read_csv(
            io.StringIO("geoid,est,se\ng1,1.0,0.0\ng2,2.5,0.7071067811865476\n")
        ).set_index(["geoid"])
        observed = est.monte_carlo_estimate(self.df.droplevel("year"))

        pandas.testing.assert_frame_equal(observed, known)

    def test_year(self):
        known = pandas.read_csv(
            io.StringIO(
                "geoid,year,est,se\ng1,2019,1.0,0.0\ng2,2019,2.5,0.7071067811865476\n"
            )
        ).set_index(["geoid", "year"])
        observed = est.monte_carlo_estimate(self.df)

        pandas.testing.assert_frame_equal(observed, known)
