import pandas
import pytest

from livelike import multi
from livelike.config import constraints

PUMA_FIPS = pytest.data_0100100["puma_fips"]
STATE_FIPS = pytest.data_0100100["st_fips"]


def assert_leading_zero(observed: dict, nreps: int):
    """See gl:livelike#262"""
    dataframes_names = [
        "est_g1",
        "est_g2",
        "moe_g1",
        "moe_g2",
        "se_g1",
        "se_g2",
    ]

    for rep in range(1, nreps + 1):
        for df_name in dataframes_names:
            df = getattr(observed[f"{PUMA_FIPS}_{rep}"], df_name)
            # pandas < 3  --> "O" == object == str
            # pandas >= 3 --> "string[pyarrow]" == string == str
            assert df.index.dtype.name == "str"
            assert df.index.str.startswith(STATE_FIPS).all()


def test_make_replicate_pumas_errors():
    # bad ``target_zone``
    with pytest.raises(
        ValueError, match="Only geographies of type ``bg`` are supported at this time."
    ):
        multi.make_replicate_pumas(None, target_zone="sassy")

    # bad ``nreps`` -- 0
    with pytest.raises(
        ValueError, match="Number of replicates must be between 1 and 80."
    ):
        multi.make_replicate_pumas(None, nreps=0)

    # bad ``nreps`` -- 81
    with pytest.raises(
        ValueError, match="Number of replicates must be between 1 and 80."
    ):
        multi.make_replicate_pumas(None, nreps=81)


@pytest.skip_if_no_censusapikey
def test_make_replicate_pumas():
    nreps = 2

    mpu = multi.make_replicate_pumas(
        fips=PUMA_FIPS,
        constraints=constraints.copy(),
        nreps=nreps,
        censusapikey=pytest.CENSUSAPIKEY,
        cache_folder=pytest.testing_cache / "make_replicate_pumas",
    )

    assert len(mpu) == nreps + 1

    puma_fips_0 = f"{PUMA_FIPS}_0"
    puma_fips_1 = f"{PUMA_FIPS}_1"

    # testing 'constraints' attr separately – see gl:livelike#171
    _contrs_0 = mpu[puma_fips_0].constraints.copy()
    _contrs_1 = mpu[puma_fips_1].constraints.copy()
    for _attr in ["PWGTP", "WGTP"]:
        _contrs_0_val = _contrs_0.loc[_contrs_0["pums1"].str.startswith(_attr)].index
        _contrs_1.loc[_contrs_0_val, "pums1"] = _contrs_0.loc[_contrs_0_val, "pums1"]
    pandas.testing.assert_frame_equal(_contrs_0, _contrs_1)

    attrs = [
        "est_g1",
        "est_g2",
        # "est_ind",  # no loner testing – see gl:livelike#171
        "moe_g1",
        "moe_g2",
        "se_g1",
        "se_g2",
        "sporder",
    ]
    for attr in attrs:
        pandas.testing.assert_frame_equal(
            getattr(mpu[puma_fips_0], attr), getattr(mpu[puma_fips_1], attr)
        )

    attrs = ["est_household", "est_person", "g1", "g2", "geo", "topo"]
    for attr in attrs:
        assert getattr(mpu[puma_fips_0], attr) is None
        assert getattr(mpu[puma_fips_1], attr) is None

    assert mpu[puma_fips_0].year == mpu[puma_fips_1].year

    assert mpu[puma_fips_0].wt.sum() == 92126.0
    assert mpu[puma_fips_1].wt.sum() == 92252.0

    assert_leading_zero(mpu, nreps)
