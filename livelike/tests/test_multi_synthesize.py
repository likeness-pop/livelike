import pandas
import pytest

from livelike import multi

####################################################################################
# -- Helper classes for pseudo instances


class Shell:
    def __init__(self, fips: str, year: None | int):
        self.fips = fips
        self.year = year
        self.path = pytest.TEST_DIR / f"buildup_{self.fips}"

    def read(self, f: str) -> pandas.DataFrame:
        return pandas.read_parquet(self.path / pytest.PARQ.format(f))


class ShellPUMA(Shell):
    def __init__(self, fips: str, year: int):
        super().__init__(fips, year)
        self.est_ind = self.read("est_ind")
        self.est_g2 = self.read("est_g2")
        self.sporder = self.read("sporder")


class ShellPMEDM(Shell):
    def __init__(self, fips: str):
        super().__init__(fips, None)
        self.almat = self.read("almat").to_numpy()


####################################################################################


class TestMultiSynthesize:
    def setup_method(self):
        self.st_fips = pytest.data_0100100["st_fips"]
        self.puma_1 = pytest.data_0100100["puma_fips"]
        self.puma_2 = pytest.data_0100200["puma_fips"]
        self.year = 2019
        self.mpu = {
            self.puma_1: ShellPUMA(self.puma_1, self.year),
            self.puma_2: ShellPUMA(self.puma_2, self.year),
        }
        self.pmds = {
            self.puma_1: ShellPMEDM(self.puma_1),
            self.puma_2: ShellPMEDM(self.puma_2),
        }
        self.nsims = 2

        self.known_replicates = set([self.puma_1, self.puma_2])  # noqa: C405
        self.known_sims = self.nsims
        self.known_sims_eq_count = True
        self.known_in_state = True
        self.known_year = True

    def test_default(self):
        observed = multi.synthesize_multi(self.mpu, self.pmds, nsim=self.nsims)

        assert isinstance(observed, pandas.DataFrame)
        observed_replicates = set(observed["replicate"].unique())
        assert observed_replicates == self.known_replicates
        observed_sims = observed["sim"].nunique()
        assert observed_sims == self.known_sims
        observed_sims_eq_count = observed.groupby("sim")["count"].sum().nunique() == 1
        assert observed_sims_eq_count == self.known_sims_eq_count
        observed_in_state = observed["geoid"].str.startswith(self.st_fips).all()
        assert observed_in_state == self.known_in_state

    def test_year(self):
        observed = multi.synthesize_multi(
            self.mpu,
            self.pmds,
            nsim=self.nsims,
            append_year=True,
        )

        assert isinstance(observed, pandas.DataFrame)
        observed_replicates = set(observed["replicate"].unique())
        assert observed_replicates == self.known_replicates
        observed_sims = observed["sim"].nunique()
        assert observed_sims == self.known_sims
        observed_sims_eq_count = observed.groupby("sim")["count"].sum().nunique() == 1
        assert observed_sims_eq_count == self.known_sims_eq_count
        observed_in_state = observed["geoid"].str.startswith(self.st_fips).all()
        assert observed_in_state == self.known_in_state
        observed_year = (observed["year"] == self.year).all()
        assert observed_year == self.known_year
