import numpy
import pytest

from livelike import acs
from livelike.config import constraints, rel, up_base_constraints_selection

year = 2019
d = "2010" if year < 2020 else "2020"
rel = rel.get_group(d)

fips = "4701601"
trt_geoids = rel[rel["puma"] == fips]["geoid"].values

constraints = constraints.loc[
    (constraints["begin_year"] <= year) & (constraints["end_year"] >= year)
]
constraints = acs.select_constraints(constraints, up_base_constraints_selection)

v = constraints["code"].head().tolist()
v_fmt = [acs.format_acs_code(i, "E") for i in v]

# Observed values #

fips_tree_vals = {
    "47": {
        "001": [
            "020100",
            "020201",
            "020202",
            "020300",
            "020400",
            "020500",
            "020600",
            "020700",
            "020800",
            "020901",
            "020902",
            "021000",
            "021100",
            "021201",
            "021202",
            "021301",
            "021302",
            "980100",
        ],
        "093": [
            "006102",
            "006301",
            "006302",
        ],
        "173": [
            "040100",
            "040201",
            "040202",
            "040300",
        ],
    }
}

ext_trt_vals = numpy.array(
    [
        [3196, 7, 3196, 1806, 1510],
        [4080, 108, 3972, 1899, 1712],
        [3973, 90, 3883, 2074, 1731],
        [3851, 0, 3851, 1760, 1486],
        [4275, 71, 4262, 2287, 1877],
        [3655, 49, 3655, 1748, 1434],
        [2540, 0, 2531, 1021, 917],
        [1575, 0, 1575, 740, 571],
        [4909, 119, 4790, 2122, 1888],
        [5473, 133, 5340, 2807, 2220],
        [6568, 0, 6568, 2641, 2459],
        [6239, 0, 6239, 2626, 2504],
        [4118, 0, 4118, 1811, 1649],
        [5231, 379, 4863, 2073, 1891],
        [5297, 129, 5168, 2941, 2556],
        [3662, 0, 3662, 1596, 1361],
        [7419, 227, 7419, 3019, 2775],
        [5402, 0, 5402, 2266, 2073],
        [3591, 0, 3591, 1468, 1309],
        [2708, 0, 2708, 1178, 1075],
        [7062, 9, 7062, 3243, 2723],
        [4060, 184, 3876, 1677, 1469],
        [5966, 0, 5966, 2798, 2175],
        [2400, 0, 2400, 1663, 1038],
    ]
)

ext_bg_vals = numpy.array(
    [
        [1678, 0, 1678, 1129, 911],
        [1518, 7, 1518, 677, 599],
        [2760, 0, 2760, 1279, 1172],
        [1320, 108, 1212, 620, 540],
        [1169, 0, 1169, 551, 467],
        [1362, 90, 1272, 790, 655],
        [1442, 0, 1442, 733, 609],
        [1258, 0, 1258, 593, 506],
        [1169, 0, 1169, 429, 402],
        [1424, 0, 1424, 738, 578],
        [848, 0, 848, 541, 432],
        [1847, 34, 1847, 847, 731],
        [1580, 37, 1567, 899, 714],
        [1342, 49, 1342, 674, 525],
        [668, 0, 668, 416, 349],
        [1645, 0, 1645, 658, 560],
        [1453, 0, 1444, 615, 559],
        [1087, 0, 1087, 406, 358],
        [627, 0, 627, 353, 275],
        [948, 0, 948, 387, 296],
        [1577, 0, 1577, 736, 639],
        [1594, 119, 1475, 636, 547],
        [1738, 0, 1738, 750, 702],
        [1029, 23, 1006, 595, 507],
        [1652, 0, 1652, 1040, 642],
        [2792, 110, 2682, 1172, 1071],
        [1325, 0, 1325, 537, 473],
        [2948, 0, 2948, 1137, 1137],
        [676, 0, 676, 319, 256],
        [1619, 0, 1619, 648, 593],
        [1188, 0, 1188, 546, 454],
        [1994, 0, 1994, 778, 748],
        [1727, 0, 1727, 687, 687],
        [1330, 0, 1330, 615, 615],
        [1501, 0, 1501, 676, 588],
        [1254, 0, 1254, 517, 472],
        [1363, 0, 1363, 618, 589],
        [1424, 368, 1056, 429, 429],
        [1064, 0, 1064, 527, 433],
        [1736, 11, 1736, 723, 670],
        [1007, 0, 1007, 394, 359],
        [955, 0, 955, 523, 426],
        [1269, 129, 1140, 709, 601],
        [1674, 0, 1674, 939, 855],
        [1399, 0, 1399, 770, 674],
        [1100, 0, 1100, 483, 395],
        [2562, 0, 2562, 1113, 966],
        [1163, 0, 1163, 600, 544],
        [1317, 0, 1317, 494, 450],
        [1808, 0, 1808, 641, 621],
        [2171, 0, 2171, 999, 875],
        [960, 227, 960, 285, 285],
        [1988, 0, 1988, 761, 717],
        [1585, 0, 1585, 462, 462],
        [1829, 0, 1829, 1043, 894],
        [1802, 0, 1802, 824, 704],
        [1789, 0, 1789, 644, 605],
        [1716, 0, 1716, 718, 665],
        [992, 0, 992, 460, 410],
        [1287, 0, 1287, 671, 461],
        [887, 0, 887, 502, 405],
        [1136, 0, 1136, 548, 497],
        [1488, 0, 1488, 739, 603],
        [2264, 9, 2264, 783, 757],
        [1379, 0, 1379, 574, 530],
        [1461, 89, 1372, 523, 448],
        [1220, 95, 1125, 580, 491],
        [1001, 0, 1001, 675, 385],
        [1280, 0, 1280, 707, 499],
        [1492, 0, 1492, 622, 558],
        [2193, 0, 2193, 794, 733],
        [736, 0, 736, 573, 366],
        [1664, 0, 1664, 1090, 672],
    ]
)

# Tests #


def test_fips_tree():
    observed = acs.build_fips_tree(trt_geoids)
    known = fips_tree_vals
    numpy.testing.assert_equal(observed, known)


def test_fips_tree_invalid():
    with pytest.raises(
        ValueError,
        match="All FIPS codes must be 11 characters or more.",
    ):
        acs.build_fips_tree(["1" * 11, "2" * 10])


def test_invalid_level():
    with pytest.raises(
        ValueError,
        match="Argument ``level`` must be one of: ``tract``, ``bg``.",
    ):
        acs.build_acs_sf_inputs(None, None, level="galaxy")


@pytest.skip_if_no_censusapikey
def test_build_acs_sf_trt():
    observed = acs.build_acs_sf_inputs(
        v_fmt,
        trt_geoids,
        level="trt",
        year=year,
        censusapikey=pytest.CENSUSAPIKEY,
    )
    known = ext_trt_vals
    numpy.testing.assert_array_equal(observed, known)


@pytest.skip_if_no_censusapikey
def test_build_acs_sf_bg():
    observed = acs.build_acs_sf_inputs(
        v_fmt,
        trt_geoids,
        level="bg",
        year=year,
        censusapikey=pytest.CENSUSAPIKEY,
    )
    known = ext_bg_vals
    numpy.testing.assert_array_equal(observed, known)
