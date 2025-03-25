import ast
import logging
import os
import pathlib
import shutil
from sys import platform

import pandas
import pytest
from likeness_vitals.vitals import get_censusapikey

TEST_DIR = pathlib.Path("livelike", "tests")
PARQ = "{}.parquet"


# --------------------------------------------------------------
# adding command line options
# ---------------------------


def pytest_addoption(parser):
    """Add custom command line arguments to the testing suite"""

    # flag for local or remote/VM testing
    parser.addoption(
        "--local",
        action="store",
        default="True",
        help="Boolean flag for local or remote/VM testing.",
        choices=("True", "False"),
        type=str,
    )

    # flag for `dev` environment testing (bleeding edge dependencies)
    parser.addoption(
        "--env",
        action="store",
        default="latest",
        help=(
            "Environment type label of dependencies for determining whether certain "
            "tests should be run. Generally we are working with minimum/oldest, "
            "latest/stable, and bleeding edge."
        ),
        type=str,
    )


# --------------------------------------------------------------
# adding accessible attributes & methods to the configuration
# -----------------------------------------------------------


def pytest_configure(config):
    """Set session attributes."""

    # ------------------------------------------------------
    # declaring command line options as attributes
    # --------------------------------------------

    # ``local`` from ``pytest_addoption()``
    pytest.LOCAL = ast.literal_eval(config.getoption("local"))

    # ``env`` from ``pytest_addoption()``
    pytest.ENV = config.getoption("env")
    valid_env_suffix = ["min", "latest", "dev"]
    assert pytest.ENV.split("_")[-1] in valid_env_suffix
    pytest.CI_ENV = "local_testing" if pytest.LOCAL else pytest.ENV

    # ------------------------------------------------------
    # declaring custom attributes and methods
    # ---------------------------------------

    # grouped tests with ``pytest.xdist``
    pytest.xdist_group_1 = pytest.mark.xdist_group(name="xdist_group_1")
    pytest.xdist_group_2 = pytest.mark.xdist_group(name="xdist_group_2")

    pytest.xdist_acs_cache = pytest.mark.xdist_group(name="xdist_acs_cache")
    pytest.xdist_acs_cache_trt = pytest.mark.xdist_group(name="xdist_acs_cache_trt")

    # on windows?
    pytest.SYS_WIN = platform.startswith("win")

    # determine if using Census API key
    pytest.DEV_CI = pytest.ENV.endswith("dev")
    is_local = pytest.LOCAL
    if not is_local:
        key = os.environ.get("CENSUS_API_KEY") if pytest.DEV_CI else None
    else:
        key = get_censusapikey()
    pytest.CENSUSAPIKEY = key
    pytest.USING_KEY = bool(pytest.CENSUSAPIKEY)

    # declare the caching directory
    pytest.testing_cache = pathlib.Path("testing_livelike_acs_cache")

    # flag to run census API queries
    pytest.skip_if_no_censusapikey = pytest.mark.skipif(
        not pytest.DEV_CI and not pytest.USING_KEY, reason=pytest.CI_ENV
    )

    # path to testing
    pytest.TEST_DIR = TEST_DIR

    # parquet file and extension
    pytest.PARQ = PARQ

    # -- pre-built test data --------------
    # Lauderdale, Colbert & Franklin Counties – Alabama
    pytest.data_0100100 = data_0100100()
    # Limestone County – Alabama
    pytest.data_0100200 = data_0100200()


# --------------------------------------------------------------
# run stuff before testing suite starts
# -------------------------------------


def pytest_sessionstart(session):  # noqa: ARG001
    """Do this before starting tests."""

    logging.info(f"Using Census API key: {pytest.USING_KEY}")
    if pytest.testing_cache.is_dir():
        shutil.rmtree(pytest.testing_cache)


# --------------------------------------------------------------
# run stuff after testing suite finishes
# --------------------------------------


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Do this after all tests are finished."""

    # specifically for use with ``pytest-xdist``
    if not hasattr(session.config, "workerinput"):  # noqa: SIM102
        if pytest.testing_cache.is_dir():
            shutil.rmtree(pytest.testing_cache)


# --------------------------------------------------------------
# resusable helpers & data preppers
# ---------------------------------


def _load_puma(puma_fips: str) -> dict[str, str | pandas.DataFrame]:
    """Load pre-computed PUMA/PMEDM data."""

    st_fips = puma_fips[:2]
    builder_dir = TEST_DIR / f"buildup_{puma_fips}"
    almat = pandas.read_parquet(builder_dir / PARQ.format("almat")).to_numpy()
    est_ind = pandas.read_parquet(builder_dir / PARQ.format("est_ind"))
    est_g2 = pandas.read_parquet(builder_dir / PARQ.format("est_g2"))
    sporder = pandas.read_parquet(builder_dir / PARQ.format("sporder"))
    pums_seg_ids = pandas.read_parquet(builder_dir / PARQ.format("pums_segment_ids"))

    return {
        "puma_fips": puma_fips,
        "st_fips": st_fips,
        "almat": almat,
        "est_ind": est_ind,
        "est_g2": est_g2,
        "sporder": sporder,
        "pums_seg_ids": pums_seg_ids,
    }


def data_0100100() -> dict[str, str | pandas.DataFrame]:
    """Lauderdale, Colbert & Franklin Counties – Alabama"""
    return _load_puma("0100100")


def data_0100200() -> dict[str, str | pandas.DataFrame]:
    """Limestone County – Alabama"""
    return _load_puma("0100200")
