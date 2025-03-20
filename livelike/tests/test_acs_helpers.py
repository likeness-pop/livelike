import numpy
import pandas
import pytest

import livelike

# ACS SF variable code
var_code_under = "B01001_001"
var_code_smoosh = var_code_under.replace("_", "")
st = "47"
cty = "093"

# testing parameters
# - acs_default
# - acs_estimate
# - acs_moe
parametrize_format_acs_code = pytest.mark.parametrize(
    "observed, known",
    [
        (livelike.acs.format_acs_code(var_code_smoosh), f"{var_code_under}E"),
        (livelike.acs.format_acs_code(var_code_smoosh, "E"), f"{var_code_under}E"),
        (livelike.acs.format_acs_code(var_code_smoosh, "M"), f"{var_code_under}M"),
    ],
)


@parametrize_format_acs_code
def test_format_acs_code(observed, known):
    assert observed == known


def _parse_pums_variables_subset(level: str) -> pandas.DataFrame:
    """1-row slice of appropriate level of ``livelike.contraints``."""
    constr = livelike.constraints.copy()
    return constr[constr["level"] == level].iloc[:1]


parametrize_parse_pums_variables = pytest.mark.parametrize(
    "level, replicate, known",
    [
        ("person", None, numpy.array(["RELP"])),
        ("household", None, numpy.array(["WGTP"])),
        ("household", 1, numpy.array(["WGTP", "WGTP1"])),
    ],
)


@parametrize_parse_pums_variables
def test_parse_pums_variables(level, replicate, known):
    observed = livelike.acs.parse_pums_variables(
        _parse_pums_variables_subset(level),
        level,
        replicate=replicate,
    )
    numpy.testing.assert_array_equal(observed, known)


def test_update_data_types():
    known = pandas.DataFrame(
        {"SERIALNO": ["1", "2"], "SPORDER": [1, 2], "ADJINC": [1.0, 2.0]}
    )
    observed = livelike.acs.update_data_types(
        pandas.DataFrame(
            {"SERIALNO": [1, 2], "SPORDER": ["1", "2"], "ADJINC": ["1", "2"]}
        )
    )
    pandas.testing.assert_frame_equal(observed, known)


@pytest.mark.parametrize("level, known", [("person", ",SPORDER"), ("household", "")])
def test_build_census_microdata_api_base_request(level, known):
    known = "https://api.census.gov/data/2019/acs/acs5/pums?get=SERIALNO" + known
    observed = livelike.acs.build_census_microdata_api_base_request(level)
    assert observed == known


parametrize_test_build_census_microdata_api_geo_request = pytest.mark.parametrize(
    "year, known",
    [
        (2023, "&for=state:01&PUMA=100"),
        (2019, "&for=public%20use%20microdata%20area:100&in=state:01"),
    ],
)


@parametrize_test_build_census_microdata_api_geo_request
def test_build_census_microdata_api_geo_request(year, known):
    observed = livelike.acs.build_census_microdata_api_geo_request("0100100", year=year)
    assert observed == known


def test_build_census_microdata_api_geo_request_error():
    with pytest.raises(ValueError, match="Supported years are 2016 - 2019 and 2023+."):
        livelike.acs.build_census_microdata_api_geo_request("0100100", year=2015)
