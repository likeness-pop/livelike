import pathlib

import dill as pickle
import numpy

# ------------------------------------------------------------------------------
# Known values #

# basic_usage.ipynb
basic_usage_pmedm_conv = -1.5199132897806373

# basic_usage_2020s.ipynb
basic_usage_2020s_pmedm_conv = -1.9071858768243724

# basic_usage__pums_replicates.ipynb
basic_usage__pums_replicates_pmedm_conv = [
    -1.2862829400042641,
    -1.2706565276505228,
    -1.2834215383161953,
    -1.2817294671258241,
    -1.285486544072835,
    -1.2539121494025147,
]


# multi_puma.ipynb
multi_puma_pmedm_conv = [-1.2539121494025147, -1.5199132897806373, -1.2546857169586016]

# tract_supertract.ipynb
tract_supertract_pmedm_conv = -0.8898624720083049

# tract_supertract_2023.ipynb
tract_supertract_2023_pmedm_conv = -1.1279548985719166


# ------------------------------------------------------------------------------
# Tests #

# load P-MEDM convergence values from notebooks
pickle_path = pathlib.Path("livelike", "tests", "notebook_pmedm_solutions")
with open(pickle_path / "notebook_pmedm_solutions.pkl", "rb") as f:
    nb_pmedm_conv = pickle.load(f)


def test_basic_usage():
    observed = nb_pmedm_conv["basic_usage"]
    known = basic_usage_pmedm_conv

    numpy.testing.assert_array_almost_equal(observed, known, decimal=3)


def test_basic_usage_2020s():
    observed = nb_pmedm_conv["basic_usage_2020s"]
    known = basic_usage_2020s_pmedm_conv

    numpy.testing.assert_array_almost_equal(observed, known, decimal=3)


def test_basic_usage__pums_replicates():
    observed = nb_pmedm_conv["basic_usage__pums_replicates"]
    known = basic_usage__pums_replicates_pmedm_conv

    numpy.testing.assert_array_almost_equal(observed, known, decimal=3)


def test_multi_puma():
    observed = nb_pmedm_conv["multi_puma"]
    known = multi_puma_pmedm_conv

    numpy.testing.assert_array_almost_equal(observed, known, decimal=3)


def test_tract_supertract():
    observed = nb_pmedm_conv["tract_supertract"]
    known = tract_supertract_pmedm_conv

    numpy.testing.assert_array_almost_equal(observed, known, decimal=3)


def test_tract_supertract_2023():
    observed = nb_pmedm_conv["tract_supertract_2023"]
    known = tract_supertract_2023_pmedm_conv

    numpy.testing.assert_array_almost_equal(observed, known, decimal=3)
