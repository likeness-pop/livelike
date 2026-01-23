import re

import numpy
import pytest

from livelike.acs import extract_geographies, parse_bg_fips_by_tract
from livelike.config import rel

year = 2019
d = "2010" if year < 2020 else "2020"
rel = rel.get_group(d)

puma = "4701603"
trt_geoids = rel[rel["puma"] == puma]["geoid"].values

# known values #
bg_geoids = numpy.array(
    [
        "470930001001",
        "470930008001",
        "470930008002",
        "470930008003",
        "470930009011",
        "470930009021",
        "470930009022",
        "470930014001",
        "470930014002",
        "470930014003",
        "470930015001",
        "470930015002",
        "470930015003",
        "470930016001",
        "470930016002",
        "470930017001",
        "470930017002",
        "470930018001",
        "470930018002",
        "470930019001",
        "470930020001",
        "470930020002",
        "470930020003",
        "470930021001",
        "470930021002",
        "470930022001",
        "470930022002",
        "470930022003",
        "470930023001",
        "470930023002",
        "470930024001",
        "470930024002",
        "470930026001",
        "470930026002",
        "470930027001",
        "470930027002",
        "470930028001",
        "470930028002",
        "470930029001",
        "470930029002",
        "470930030001",
        "470930030002",
        "470930030003",
        "470930031001",
        "470930031002",
        "470930032001",
        "470930032002",
        "470930033001",
        "470930034001",
        "470930034002",
        "470930035001",
        "470930035002",
        "470930035003",
        "470930037001",
        "470930037002",
        "470930037003",
        "470930038011",
        "470930038012",
        "470930038013",
        "470930038021",
        "470930038022",
        "470930039011",
        "470930039012",
        "470930039021",
        "470930039022",
        "470930040001",
        "470930040002",
        "470930041001",
        "470930041002",
        "470930042001",
        "470930042002",
        "470930043001",
        "470930043002",
        "470930044011",
        "470930044012",
        "470930044031",
        "470930044032",
        "470930044033",
        "470930044041",
        "470930044042",
        "470930045001",
        "470930045002",
        "470930045003",
        "470930045004",
        "470930046081",
        "470930046082",
        "470930046151",
        "470930046152",
        "470930047001",
        "470930047002",
        "470930048001",
        "470930048002",
        "470930048003",
        "470930050001",
        "470930050002",
        "470930050003",
        "470930066001",
        "470930066002",
        "470930067001",
        "470930067002",
        "470930067003",
        "470930068001",
        "470930068002",
        "470930068003",
        "470930068004",
        "470930069001",
        "470930069002",
        "470930069003",
        "470930070001",
        "470930070002",
        "470930071001",
        "470930071002",
        "470930071003",
    ]
)

bg_vals = {
    "dims": (113, 18),
    "bbox": numpy.array([-84.062909, 35.888132, -83.824544, 36.058036]),
}

trt_vals = {
    "dims": (49, 17),
    "bbox": numpy.array([-84.062909, 35.888132, -83.824544, 36.058036]),
}


# tests #
def test_parse_bg_fips_by_tract():
    observed = parse_bg_fips_by_tract(
        year=year,
        targets=trt_geoids,
    )
    known = numpy.sort(bg_geoids)
    numpy.testing.assert_array_equal(observed, known)


def test_invalid_geo():
    with pytest.raises(
        ValueError,
        match=re.escape("Target zone type (``geo``) must be one of ``bg``, ``trt``."),
    ):
        extract_geographies(None, None, geo="galaxy")


@pytest.skip_if_no_censusapikey
def test_extract_bg():
    _bg_geoids = parse_bg_fips_by_tract(
        year=year,
        targets=trt_geoids,
    )

    bg = extract_geographies(year, _bg_geoids, geo="bg")
    observed = {"dims": bg.shape, "bbox": bg.total_bounds}

    observed_dims = observed["dims"]
    known_dims = bg_vals["dims"]
    numpy.testing.assert_equal(observed_dims, known_dims)

    observed_bbox = observed["bbox"]
    known_bbox = bg_vals["bbox"]
    numpy.testing.assert_array_almost_equal(observed_bbox, known_bbox)


@pytest.skip_if_no_censusapikey
def test_extract_trt():
    trt = extract_geographies(year, trt_geoids, geo="trt")
    observed = {"dims": trt.shape, "bbox": trt.total_bounds}

    observed_dims = observed["dims"]
    known_dims = trt_vals["dims"]
    numpy.testing.assert_equal(observed_dims, known_dims)

    observed_bbox = observed["bbox"]
    known_bbox = trt_vals["bbox"]
    numpy.testing.assert_array_almost_equal(observed_bbox, known_bbox)
