import pathlib

import geopandas
import numpy
import pandas
import pytest
from geopandas.testing import assert_geodataframe_equal

from livelike import acs
from livelike.config import constraints, up_base_constraints_selection

# # P-MEDM constraints – base
year = 2019
constraints = constraints.loc[
    (constraints["begin_year"] <= year) & (constraints["end_year"] >= year)
]
constraints = acs.select_constraints(constraints, up_base_constraints_selection)

# stored super-tract instances
path_base = pathlib.Path("livelike", "tests")
file_base = "super_tract_expected"
gpkg_name = path_base / file_base / f"{file_base}.zip"


@pytest.skip_if_no_censusapikey
@pytest.mark.parametrize("method,layer", [("louvain", "louvain")])
def test_puma_super_tract(method, layer):
    known = geopandas.read_file(gpkg_name, layer=layer).set_index("GEOID")
    pup = acs.puma(
        "4701603",
        constraints=constraints.copy(),
        target_zone="trt",
        keep_geo=True,
        censusapikey=pytest.CENSUSAPIKEY,
        make_super_trt_method=method,
    )
    observed = pup.geo
    assert_geodataframe_equal(known, observed, check_dtype=False)


def test_puma_super_tracts_bad_method():
    with pytest.raises(
        ValueError,
        match=(
            "The value passed into `make_super_trt_method` "
            "is not valid `2`. See doctrings."
        ),
    ):
        acs.puma(
            "4701603",
            constraints=constraints.copy(),
            target_zone="trt",
            keep_geo=True,
            censusapikey=pytest.CENSUSAPIKEY,
            make_super_trt_method="2",
        )


def test_make_supertracts_few_constrs_universe_constrs():
    layer = "louvain_few_constrs_universe_constrs"

    known = geopandas.read_file(gpkg_name, layer=layer).set_index("GEOID")
    sf_data = known.copy()
    p = "population"
    sf_data[p] = range(1, sf_data.shape[0] + 1)
    sf_data = sf_data.drop(columns=[c for c in sf_data.columns if c != p])

    observed = known.copy().drop(columns="super_trt")
    observed["super_trt"] = acs.make_supertracts(
        observed, sf_data, exclude_universe_constraints=False, method="louvain"
    )
    pandas.testing.assert_series_equal(known["super_trt"], observed["super_trt"])


def test_make_supertracts_island_singleton():
    layer = "louvain_few_constrs_universe_constrs"

    known = geopandas.read_file(gpkg_name, layer=layer).set_index("GEOID")
    sf_data = known.copy()
    p = "population"
    sf_data[p] = range(1, sf_data.shape[0] + 1)
    sf_data = sf_data.drop(columns=[c for c in sf_data.columns if c != p])

    observed = known.copy().drop(columns="super_trt")

    keep_only = [
        "47093004000",
        "47093004100",
        "47093004200",
        "47093007100",
    ]

    observed["super_trt"] = observed["super_trt"] = acs.make_supertracts(
        observed[observed.index.isin(keep_only)],
        sf_data[sf_data.index.isin(keep_only)],
        exclude_universe_constraints=False,
        method="louvain",
    )

    numpy.testing.assert_array_equal(
        observed[~observed["super_trt"].isna()].index, keep_only
    )


def test_make_supertracts_bad_method():
    with pytest.raises(
        AssertionError,
        match="Argument ``method`` must be 'louvain'",
    ):
        acs.make_supertracts(None, None, method="2")


@pytest.skip_if_no_censusapikey
def test_puma_no_supertract():
    pup = acs.puma(
        "4701603",
        constraints=constraints.copy(),
        target_zone="trt",
        keep_intermediates=True,
        censusapikey=pytest.CENSUSAPIKEY,
        make_trt_geo=False,
        make_super_trt=False,
    )
    assert pup.est_g1.empty
    assert pup.moe_g1.empty
    assert pup.se_g1.empty
    assert getattr(pup, "geo", None) is None
    assert pup.g1 is None
    assert pup.g2 is None
    assert pup.topo is None


def test_agg_supertracts():
    def _series_eq(df1: pandas.DataFrame, df2: pandas.DataFrame):
        """Approx dataframe equality testing."""
        pandas.testing.assert_frame_equal(df1, df2, check_exact=False)

    geoids = ["a", "b", "c", "d"]
    ests = pandas.DataFrame({"v1": [2, 3, 4, 5]}, index=geoids)
    stes = pandas.DataFrame({"v1": [0.2, 0.3, 0.4, 0.5]}, index=geoids)
    super_tracts = ["X", "Y"]
    super_labels = pandas.Series(super_tracts * 2, index=geoids)

    known_ests = pandas.DataFrame({"v1": [6, 8]}, index=super_tracts)
    known_stes = pandas.DataFrame({"v1": [0.447214, 0.583095]}, index=super_tracts)
    known_moes = pandas.DataFrame({"v1": [0.735666, 0.959192]}, index=super_tracts)

    observed = acs.aggregate_acs_sf_supertracts(ests, stes, super_labels)

    _series_eq(observed["est"], known_ests)
    _series_eq(observed["se"], known_stes)
    _series_eq(observed["moe"], known_moes)
