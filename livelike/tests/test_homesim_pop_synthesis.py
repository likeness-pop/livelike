import pathlib

import numpy
import pandas
import pytest

from livelike import homesim, pums

path_base = pathlib.Path("livelike", "tests")

################################################################################
########################## Process input params ################################
################################################################################


def read_inputs(ftype):
    """Read in input data"""
    return pandas.read_csv(path_base / "pop_synthesis_input" / f"{ftype}.csv")


# toy PUMS extract - household
gph = read_inputs("gph")

# toy PUMS extract - person
gpp = read_inputs("gpp")

# toy P-MEDM allocation matrix
almat = numpy.array(
    [
        [0.041575492, 0.04868709, 0, 0.030634573],
        [0.028993435, 0.037746171, 0.008752735, 0.024617068],
        [0.012035011, 0.052516411, 0.051969365, 0.013129103],
        [0.023522976, 0.050328228, 0.022428884, 0.047045952],
        [0.022428884, 0.035010941, 0.044310722, 0.021334792],
        [0.003282276, 0.050875274, 0.002188184, 0.054704595],
        [0.031181619, 0.039387309, 0.04595186, 0.031181619],
        [0.001094092, 0.024617068, 0.015864333, 0.001094092],
        [0.029540481, 0.00273523, 0.006564551, 0.042669584],
    ]
)


###############################################################################
################### Test Pre- and Post-Processing Functions ###################
###############################################################################


def read_results(ftype):
    """Read in input data"""
    f = path_base / "prepared_constraints" / f"{ftype}.csv"
    return pandas.read_csv(f, index_col=0)


# testing parameters
# - housing_units
# - occhu
# - population # --> not testing for now. See GL#187. Need test in future.
# - group_quarters_pop
# - sex
# - emp_stat
# - sexcw
# - sexocc
# - sexnaics
# - grade
# - school
# - worked
# -tenure_vehicles
# -veh_occ
# -commute
# -travel
parametrize_pop_synthesis = pytest.mark.parametrize(
    "contr_type, pums_extract, year",
    [
        ("housing_units", gph, None),
        ("occhu", gph, 2019),
        # ("population", gpp, None),
        ("group_quarters_pop", gpp, 2019),
        ("sex", gpp, None),
        ("emp_stat", gpp, None),
        ("sexcw", gpp, None),
        ("sexocc", gpp, None),
        ("sexnaics", gpp, None),
        ("grade", gpp, None),
        ("school", gpp, None),
        ("worked", gpp, None),
        ("tenure_vehicles", gpp, None),
        ("veh_occ", gpp, None),
        ("commute", gpp, None),
        ("travel", gpp, None),
    ],
)


@parametrize_pop_synthesis
def test_prepare_constraints(contr_type, pums_extract, year):
    args = [pums_extract, year] if year else [pums_extract]
    observed = getattr(pums, contr_type)(*args)
    known = read_results(contr_type)
    known = (
        known.values.flatten() if isinstance(observed, pandas.Series) else known.values
    )
    numpy.testing.assert_array_equal(known, observed)


def test_trs():
    observed = homesim.trs(almat * 100)
    known = numpy.array(
        [
            [4, 4, 0, 4],
            [3, 3, 1, 2],
            [2, 6, 5, 1],
            [3, 5, 2, 4],
            [2, 4, 5, 2],
            [1, 5, 0, 6],
            [3, 4, 5, 3],
            [0, 2, 2, 0],
            [2, 0, 1, 4],
        ]
    )
    numpy.testing.assert_array_equal(known, observed)
