import pathlib

import pandas
import pytest

from livelike import acs

# skip for now if no key available -- reimplement with caches later
if not pytest.USING_KEY:
    pytest.skip(allow_module_level=True)


# target PUMA
p = "4701604"

# test data base directory
path_base = pathlib.Path("livelike", "tests", "supplementary_api_wrapper_extracts")


# testing parameters
# - 1: Extract by Query - Person
# - 2: Extract by Query - Household
# - 3: Extract Descriptors - Person
# - 4: Extract Descriptors - Household
parametrize_extracts = pytest.mark.parametrize(
    "extract_type, level, query_or_features, sort_by",
    [
        ("1", "person", "ESR=1&NAICSP=6111&OCCP=2300:2320&WKHP=40:999", "p_id"),
        ("2", "household", "YBL=8&BLD=2", "SERIALNO"),
        ("3", "person", ["ESR", "NAICSP", "OCCP", "WKHP"], "p_id"),
        ("4", "household", ["YBL", "BLD"], "SERIALNO"),
    ],
)


@parametrize_extracts
def test_extracts(extract_type, level, query_or_features, sort_by):
    # observed -----------------------------------------------------
    extract_func = (
        acs.extract_pums_descriptors
        if isinstance(query_or_features, list)
        else acs.extract_pums_segment_ids
    )
    ext = extract_func(p, level, query_or_features, censusapikey=pytest.CENSUSAPIKEY)
    observed = acs.update_data_types(ext).sort_values(sort_by).reset_index(drop=True)

    # known --------------------------------------------------------
    known = pandas.read_parquet(path_base / f"ext{extract_type}.parquet")

    pandas.testing.assert_frame_equal(observed, known)
