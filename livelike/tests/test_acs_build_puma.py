import pathlib

import numpy
import pandas
import pytest

from livelike import acs
from livelike.config import constraints, up_base_constraints_selection

# skip for now if no key available -- reimplement with caches later
if not pytest.USING_KEY:
    pytest.skip(allow_module_level=True)


################################################################################
########################## Build PUMA object ###################################
################################################################################

year = 2019
constraints = constraints.loc[
    (constraints["begin_year"] <= year) & (constraints["end_year"] >= year)
]
constraints = constraints[constraints["geo_base_level"] == "bg"]

sel = up_base_constraints_selection.copy()
sel.update({"economic": ["poverty"]})

# Create baseline PUMA representation within a target PUMA ('4701604')
pup = acs.puma(
    "4701604", constraints, constraints_selection=sel, censusapikey=pytest.CENSUSAPIKEY
)


################################################################################
########################## Test P-MEDM Inputs ##################################
################################################################################


path_test_base = pathlib.Path("livelike", "tests", "puma")


def read_known(ftype):
    """Read in known results data"""
    return pandas.read_parquet(path_test_base / f"{ftype}.parquet")


# testing parameters
# - geo_constraints_L1
# - geo_constraints_L2
# - geo_standard_errors_L1
# - geo_standard_errors_L2
# - ind_constraints
# - sporder
# - wt
parametrize_puma_results = pytest.mark.parametrize(
    "puma_attr, result_type",
    [
        ("est_g1", "est_trt"),
        ("est_g2", "est_bg"),
        ("se_g1", "se_trt"),
        ("se_g2", "se_bg"),
        ("est_ind", "est_ind"),
        ("sporder", "sporder"),
        ("wt", "wt"),
    ],
)


@parametrize_puma_results
def test_build_puma(puma_attr, result_type):
    # observed -----------------------------------------------------
    observed = getattr(pup, puma_attr)

    # known --------------------------------------------------------
    known = read_known(result_type)
    if puma_attr == "wt":
        known = known[0].to_numpy().flatten()
        numpy.testing.assert_array_equal(observed, known)
    else:
        pandas.testing.assert_frame_equal(observed, known)
