import os

import numpy
import pandas

from livelike import _attribution

eval_data_person = pandas.read_parquet(
    os.path.join(
        "livelike", "tests", "attribution_eval_data", "eval_data_person.parquet"
    )
)

eval_data_household = pandas.read_parquet(
    os.path.join(
        "livelike", "tests", "attribution_eval_data", "eval_data_household.parquet"
    )
)


## Person-level attributes ##
def test_age():
    observed = _attribution.age(eval_data_person).tolist()
    known = [
        "50_54",
        "15_19",
        "25_29",
        "20_24",
        "80_84",
        "75_79",
        "35_39",
        "30_34",
        "05_09",
        "05_09",
        "05u",
        "75_79",
        "75_79",
        "65_69",
        "65_69",
        "60_64",
        "50_54",
        "30_34",
        "50_54",
        "85o",
        "65_69",
        "55_59",
        "55_59",
        "65_69",
        "65_69",
    ]
    numpy.testing.assert_array_equal(observed, known)


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


def test_edu_attainment():
    observed = _attribution.edu_attainment(eval_data_person).tolist()
    known = [
        "highschoolgrad",
        "",
        "highschoolgrad",
        "",
        "bach",
        "collegenodeg",
        "assoc",
        "bach",
        "",
        "",
        "",
        "bach",
        "doc",
        "collegenodeg",
        "highschoolgrad",
        "collegenodeg",
        "highschoolgrad",
        "highschoolgrad",
        "highschoolgrad",
        "collegenodeg",
        "10th",
        "9th",
        "9th",
        "assoc",
        "highschoolgrad",
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


def test__hispanic_latino():
    observed = _attribution.hispanic_latino(eval_data_person).tolist()
    known = [
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "yes",
        "no",
        "no",
        "no",
        "no",
        "yes",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
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


def test__income_to_poverty_ratio():
    observed = _attribution.income_to_poverty_ratio(eval_data_person).tolist()
    known = [
        "100_124",
        "",
        "",
        "",
        "GE200",
        "GE200",
        "GE200",
        "GE200",
        "GE200",
        "GE200",
        "GE200",
        "GE200",
        "GE200",
        "100_124",
        "150_184",
        "150_184",
        "050_099",
        "050_099",
        "GE200",
        "GE200",
        "050_099",
        "100_124",
        "050_099",
        "GE200",
        "050_099",
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


def test__race():
    observed = _attribution.race(eval_data_person).tolist()
    known = [
        "white",
        "white",
        "blk_af_amer",
        "white",
        "white",
        "white",
        "white",
        "white",
        "white",
        "white",
        "white",
        "white",
        "white",
        "mult",
        "blk_af_amer",
        "blk_af_amer",
        "white",
        "white",
        "white",
        "white",
        "white",
        "white",
        "white",
        "white",
        "blk_af_amer",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__residence_type__person():
    observed = _attribution.residence_type__person(eval_data_person).tolist()
    known = [
        "gq",
        "gq",
        "gq",
        "gq",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
        "hu",
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


def test__sex():
    observed = _attribution.sex(eval_data_person).tolist()
    known = [
        "female",
        "female",
        "male",
        "male",
        "male",
        "female",
        "female",
        "male",
        "male",
        "male",
        "female",
        "male",
        "female",
        "female",
        "male",
        "female",
        "male",
        "male",
        "male",
        "female",
        "male",
        "female",
        "male",
        "male",
        "female",
    ]
    numpy.testing.assert_array_equal(observed, known)


## Household-level attributes ##
def test__dwelling_type():
    observed = _attribution.dwelling_type(eval_data_household).tolist()
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
        "GE50_unit",
        "3_4_unit",
        "single_fam_detach",
        "GE50_unit",
        "10_19_unit",
        "single_fam_detach",
        "single_fam_detach",
        "single_fam_detach",
        "single_fam_detach",
        "single_fam_detach",
        "3_4_unit",
        "single_fam_detach",
        "GE50_unit",
        "single_fam_detach",
        "10_19_unit",
        "single_fam_detach",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__household_income():
    observed = _attribution.household_income(eval_data_household).tolist()
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
        "15_20k",
        "40_45k",
        "45_50k",
        "15_20k",
        "",
        "75_100k",
        "GE200k",
        "75_100k",
        "10_15k",
        "30_35k",
        "60_75k",
        "50_60k",
        "25_30k",
        "10_15k",
        "75_100k",
        "10_15k",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__house_heating_fuel():
    observed = _attribution.house_heating_fuel(eval_data_household).tolist()
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
        "electricity",
        "electricity",
        "electricity",
        "electricity",
        "",
        "electricity",
        "electricity",
        "utility_gas",
        "utility_gas",
        "electricity",
        "electricity",
        "electricity",
        "electricity",
        "utility_gas",
        "electricity",
        "electricity",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__housing_costs_pct_income():
    observed = _attribution.housing_costs_pct_income(eval_data_household).tolist()
    known = [
        "",
        "",
        "",
        "",
        "less10",
        "less10",
        "less10",
        "less10",
        "less10",
        "less10",
        "less10",
        "10to14.9",
        "10to14.9",
        "more50",
        "10to14.9",
        "10to14.9",
        "30to34.9",
        "30to34.9",
        "30to34.9",
        "10to14.9",
        "25to29.9",
        "25to29.9",
        "more50",
        "10to14.9",
        "more50",
    ]
    numpy.testing.assert_array_equal(observed, known)


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


def test__living_arrangement():
    observed = _attribution.living_arrangement(eval_data_household).tolist()
    known = [
        "",
        "",
        "",
        "",
        "married",
        "married",
        "married",
        "married",
        "married",
        "married",
        "married",
        "married",
        "married",
        "alone",
        "married",
        "married",
        "male_no_spouse",
        "male_no_spouse",
        "male_no_spouse",
        "alone",
        "not_alone",
        "not_alone",
        "alone",
        "alone",
        "alone",
    ]
    numpy.testing.assert_array_equal(observed, known)


def test__mortgage():
    observed = _attribution.mortgage(eval_data_household).tolist()
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
        "mortgage",
        "",
        "",
        "nomortgage",
        "mortgage",
        "nomortgage",
        "nomortgage",
        "nomortgage",
        "",
        "nomortgage",
        "",
        "nomortgage",
        "",
        "mortgage",
    ]
    numpy.testing.assert_array_equal(observed, known)


def residence_type__household():
    observed = _attribution.residence_type__household(eval_data_household).tolist()
    known = [
        "gq",
        "gq",
        "gq",
        "gq",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
        "hu_occ",
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


def test__year_dwelling_built():
    observed = _attribution.year_dwelling_built(eval_data_household).tolist()
    known = [
        "",
        "",
        "",
        "",
        "40_49",
        "40_49",
        "00_09",
        "00_09",
        "00_09",
        "00_09",
        "00_09",
        "50_59",
        "50_59",
        "50_59",
        "40_49",
        "40_49",
        "GE2020",
        "GE2020",
        "GE2020",
        "50_59",
        "60_69",
        "60_69",
        "L1939",
        "70_79",
        "40_49",
    ]
    numpy.testing.assert_array_equal(observed, known)
