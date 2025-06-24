import re
import urllib

import pandas
import pytest

from livelike import get_vre_tables


def test_get_vre_tables():
    # a simple smoke test that ensures results are returned

    if pytest.DEV_CI:
        # NOTE: When this specific block starts to fail
        # NOTE: that's the cue that certifi/ca-certificates
        # NOTE: is fixed and we can remove the pin.
        # NOTE: See gh#105
        with pytest.raises(
            urllib.error.URLError,
            match=re.escape(
                "<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate "
                "verify failed: unable to get local issuer certificate"
            ),
        ):
            observed = get_vre_tables("56", 2023, "B01001")
    else:
        observed = get_vre_tables("56", 2023, "B01001")

        assert isinstance(observed, pandas.DataFrame)
        assert observed.empty is False
