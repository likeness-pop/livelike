import pandas

from livelike import get_vre_tables


def test_get_vre_tables():
    # a simple smoke test that ensures results are returned
    observed = get_vre_tables("56", 2023, "B01001")

    assert isinstance(observed, pandas.DataFrame)
    assert observed.empty is False
