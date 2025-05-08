import os

import numpy
import pandas

from livelike import _attribution

eval_data_person = pandas.read_parquet(
    os.path.join("livelike", "tests", "attribution_eval_data", "eval_data_person.parquet")
)

eval_data_household = pandas.read_parquet(
    os.path.join(
        "livelike", "tests", "attribution_eval_data", "eval_data_household.parquet"
    )
)


## Person-level attributes ##
def test__class_of_worker():
    observed = _attribution.class_of_worker(eval_data_person).tolist()
    known = [
        "private",
        "",
        "",
        "st.gov",
        "",
        "",
        "",
        "private",
        "",
        "",
        "",
        "",
        "",
        "",
        "private",
        "",
        "non.prof",
        "private",
        "private",
        "",
        "",
        "",
        "",
        "private",
        "",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__commute_drive_type():
    observed = _attribution.commute_drive_type(eval_data_person).tolist()
    known = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "drove_alone",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__commute_mode():
    observed = _attribution.commute_mode(eval_data_person).tolist()
    known = [
        "public_transportation",
        "",
        "",
        "walked",
        "",
        "",
        "",
        "wfh",
        "",
        "",
        "",
        "",
        "",
        "",
        "car_truck_van",
        "",
        "walked",
        "walked",
        "walked",
        "",
        "",
        "",
        "",
        "public_transportation",
        "",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__commute_time_m():
    observed = _attribution.commute_time_m(eval_data_person).tolist()
    known = [
        "10.14",
        "",
        "",
        "L10",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "45.59",
        "",
        "10.14",
        "15.19",
        "10.14",
        "",
        "",
        "",
        "",
        "30.34",
        "",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__employment_status():
    observed = _attribution.employment_status(eval_data_person).tolist()
    known = [
        "employed",
        "not.in.force",
        "not.in.force",
        "employed",
        "not.in.force",
        "not.in.force",
        "not.in.force",
        "employed",
        "",
        "",
        "",
        "not.in.force",
        "not.in.force",
        "not.in.force",
        "employed",
        "not.in.force",
        "employed",
        "employed",
        "employed",
        "not.in.force",
        "not.in.force",
        "not.in.force",
        "not.in.force",
        "employed",
        "not.in.force",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__grade():
    observed = _attribution.grade(eval_data_person).tolist()
    known = [
        "",
        "undergrad",
        "",
        "undergrad",
        "",
        "",
        "",
        "",
        "3rd",
        "1st",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__hours_worked():
    observed = _attribution.hours_worked(eval_data_person).tolist()
    known = [
        "GE35",
        "",
        "",
        "15.34",
        "",
        "",
        "",
        "GE35",
        "",
        "",
        "",
        "",
        "",
        "",
        "GE35",
        "",
        "1.14",
        "15.34",
        "GE35",
        "",
        "",
        "",
        "",
        "GE35",
        "",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__industry():
    observed = _attribution.industry(eval_data_person).tolist()
    known = [
        "mfg",
        "",
        "",
        "edu_med_sca",
        "",
        "",
        "",
        "edu_med_sca",
        "",
        "",
        "",
        "",
        "",
        "",
        "mfg",
        "",
        "srv",
        "ent",
        "ent",
        "",
        "",
        "",
        "",
        "ret",
        "",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__occupation():
    observed = _attribution.occupation(eval_data_person).tolist()
    known = [
        "prd",
        "",
        "",
        "cmp",
        "",
        "",
        "",
        "med.tech",
        "",
        "",
        "",
        "",
        "",
        "",
        "prd",
        "",
        "cln",
        "eat",
        "eat",
        "",
        "",
        "",
        "",
        "off",
        "",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__school_type():
    observed = _attribution.school_type(eval_data_person).tolist()
    known = [
        "",
        "public",
        "",
        "public",
        "",
        "",
        "",
        "",
        "private",
        "private",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    numpy.testing.assert_array_equal(observed, known)


## Household-level attributes ##
def test__household_size():
    observed = _attribution.household_size(eval_data_household).tolist()
    known = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "1p",
        "1p",
        "1p",
        "1p",
        "",
        "2p",
        "5p",
        "2p",
        "1p",
        "2p",
        "3p",
        "1p",
        "2p",
        "1p",
        "1p",
        "1p",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__household_type():
    observed = _attribution.household_type(eval_data_household).tolist()
    known = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "nonfam",
        "nonfam",
        "nonfam",
        "nonfam",
        "",
        "fam",
        "fam",
        "fam",
        "nonfam",
        "fam",
        "fam",
        "nonfam",
        "nonfam",
        "nonfam",
        "nonfam",
        "nonfam",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__tenure():
    observed = _attribution.tenure(eval_data_household).tolist()
    known = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "rent",
        "rent",
        "own",
        "rent",
        "",
        "own",
        "own",
        "own",
        "own",
        "own",
        "rent",
        "own",
        "rent",
        "own",
        "rent",
        "own",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__vehicles_available():
    observed = _attribution.vehicles_available(eval_data_household).tolist()
    known = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "none",
        "01",
        "01",
        "01",
        "",
        "02",
        "02",
        "03",
        "01",
        "none",
        "02",
        "none",
        "none",
        "01",
        "none",
        "none",
    ]
    numpy.testing.assert_array_equal(observed, known)
