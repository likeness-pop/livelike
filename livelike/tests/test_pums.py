import io

import numpy
import pandas
import pytest

import livelike

################################################################################
# helpers ----------------------------------------------------------------------
################################################################################


def test_intersect_dummies():
    df1 = pandas.DataFrame({"df1": [1, 0, 1]})
    df2 = pandas.DataFrame({"df2": [0, 1, 1]})

    known = pandas.DataFrame({"df2_df1": [0, 0, 1]})
    observed = livelike.pums.intersect_dummies(df1, df2)
    pandas.testing.assert_frame_equal(observed, known)


def test_reclass_dummies():
    df = pandas.DataFrame({"animal": ["cat_code", "dog_code", "eel_code"]})
    reclasser = {"cat_class": "^cat", "dog_class": "^dog", "eel_class": "^eel"}

    known = pandas.DataFrame(
        {"cat_class": [1, 0, 0], "dog_class": [0, 1, 0], "eel_class": [0, 0, 1]}
    )
    observed = livelike.pums.reclass_dummies(df, "animal", reclasser)
    pandas.testing.assert_frame_equal(observed, known)


################################################################################
# constraint recode functions --------------------------------------------------
################################################################################


def test_age_cohort():
    gpp = pandas.DataFrame({"AGEP": [15, 25, 45, 75]})

    known = pandas.read_csv(
        io.StringIO(
            "age_L5,age_5_17,age_18_24,age_25_34,age_35_44,age_45_54,age_55_59,age_60_61,age_62_64,age_65_74,age_GE75\nFalse,True,False,False,False,False,False,False,False,False,False\nFalse,False,False,True,False,False,False,False,False,False,False\nFalse,False,False,False,False,True,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.age_cohort(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_age_simple():
    gpp = pandas.DataFrame({"AGEP": [15, 25, 45, 75]})

    known = pandas.read_csv(
        io.StringIO(
            "age_16_19,age_20_24,age_25_44,age_45_54,age_55_64,age_65_69,age_GE70\nFalse,False,False,False,False,False,False\nFalse,False,True,False,False,False,False\nFalse,False,False,True,False,False,False\nFalse,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.age_simple(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_bedrooms():
    gph = pandas.DataFrame({"BDSP": [0, 1, 4, 6]})

    known = pandas.read_csv(
        io.StringIO(
            "bedr_00,bedr_01,bedr_02,bedr_03,bedr_04,bedr_GE05\nTrue,False,False,False,False,False\nFalse,True,False,False,False,False\nFalse,False,False,False,True,False\nFalse,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.bedrooms(gph)
    pandas.testing.assert_frame_equal(observed, known)


@pytest.mark.parametrize("column, year", [("TYPE", 2019), ("TYPEHUGQ", 2020)])
def test_civ_noninst_pop(column, year):
    gpp = pandas.DataFrame({column: [1, 2, 1]})

    known = pandas.Series([1, 0, 1], name="civ_noninst_pop")
    observed = livelike.pums.civ_noninst_pop(gpp, year)
    pandas.testing.assert_series_equal(observed, known)


def test_commute():
    gpp = pandas.DataFrame({"JWMNP": [5, 14, 29, 61]})

    known = pandas.read_csv(
        io.StringIO(
            "cmt_mins_L10,cmt_mins_10.14,cmt_mins_15.19,cmt_mins_20.24,cmt_mins_25.29,cmt_mins_30.34,cmt_mins_35.44,cmt_mins_45.59,cmt_mins_GE60\nTrue,False,False,False,False,False,False,False,False\nFalse,True,False,False,False,False,False,False,False\nFalse,False,False,False,True,False,False,False,False\nFalse,False,False,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.commute(gpp)
    observed.columns = observed.columns.astype(str)
    pandas.testing.assert_frame_equal(observed, known)


def test_disability():
    gpp = pandas.DataFrame({"AGEP": [20, 30, 40, 50], "DIS": [0, 1, 2, 1]})

    known = pandas.read_csv(
        io.StringIO(
            "aL19_disability,aL19_no_disability,a19_64_disability,a19_64_no_disability,aGE65_disability,aGE65_no_disability\nFalse,False,False,True,False,False\nFalse,False,True,False,False,False\nFalse,False,False,True,False,False\nFalse,False,True,False,False,False\n"  # noqa: E501
        )
    )
    observed = livelike.pums.disability(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_edu_attainment():
    gpp = pandas.DataFrame(
        {"SEX": [1, 2, 1, 2], "AGEP": [20, 30, 40, 50], "SCHL": [12, 15, 20, 24]}
    )

    known = pandas.read_csv(
        io.StringIO(
            "schl_female_noschool,schl_female_nurseryto4thgrade,schl_female_5thand6th,schl_female_7thand8th,"
            "schl_female_9th,schl_female_10th,schl_female_11th,schl_female_12th,schl_female_highschoolgrad,"
            "schl_female_less1yearcollege,schl_female_collegenodeg,schl_female_assoc,schl_female_bach,"
            "schl_female_masters,schl_female_prof,schl_female_doc,schl_male_noschool,schl_male_nurseryto4thgrade,"
            "schl_male_5thand6th,schl_male_7thand8th,schl_male_9th,schl_male_10th,schl_male_11th,schl_male_12th,"
            "schl_male_highschoolgrad,schl_male_less1yearcollege,schl_male_collegenodeg,schl_male_assoc,"
            "schl_male_bach,schl_male_masters,schl_male_prof,schl_male_doc\n"
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
            "0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0\n"
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
        )
    )
    observed = livelike.pums.edu_attainment(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_emp_stat():
    gpp = pandas.DataFrame({"ESR": [1, 3, 4, 6]})

    known = pandas.read_csv(
        io.StringIO(
            "emp_stat_employed,emp_stat_unemp,emp_stat_mil,emp_stat_not.in.force\nTrue,False,False,False\nFalse,True,False,False\nFalse,False,True,False\nFalse,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.emp_stat(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_foreign_born():
    gpp = pandas.DataFrame({"POBP": [99, 100]})

    known = pandas.Series([0, 1], name="foreign_born")
    observed = livelike.pums.foreign_born(gpp)
    pandas.testing.assert_series_equal(observed, known)


def test_grade():
    gpp = pandas.DataFrame({"SCHG": [5, 8, 12, 16]})

    known = pandas.read_csv(
        io.StringIO(
            "grade_preschl,grade_kind,grade_1st,grade_2nd,grade_3rd,grade_4th,grade_5th,grade_6th,grade_7th,grade_8th,grade_9th,grade_10th,grade_11th,grade_12th,grade_undergrad,grade_grad\nFalse,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.grade(gpp)
    pandas.testing.assert_frame_equal(observed, known)


@pytest.mark.parametrize("column, year", [("TYPE", 2019), ("TYPEHUGQ", 2020)])
def test_group_quarters(column, year):
    gpp = pandas.DataFrame({column: [1, 2]})

    known = pandas.Series([0, 1], name="group_quarters")
    observed = livelike.pums.group_quarters(gpp, year)
    pandas.testing.assert_series_equal(observed, known)


@pytest.mark.parametrize("column, year", [("RELP", 2018), ("RELSHIPP", 2019)])
def test_group_quarters_pop(column, year):
    gpp = pandas.DataFrame({column: [5, 15, 37]})

    known = pandas.Series([0, 0, 1], name="group_quarters_pop")
    observed = livelike.pums.group_quarters_pop(gpp, year)
    pandas.testing.assert_series_equal(observed, known)


def test_health_ins():
    gpp = pandas.DataFrame(
        {
            "TYPE" : [2, 1, 1, 3],
            "AGEP": [20, 30, 40, 50],
            "HICOV": [1, 1, 1, 1],
            "HINS1": [2, 2, 2, 1],
            "HINS2": [2, 2, 2, 2],
            "HINS3": [2, 1, 2, 2],
            "HINS4": [1, 2, 2, 2],
            "HINS5": [2, 2, 2, 2],
            "HINS6": [2, 2, 2, 2],
            "HINS7": [2, 2, 1, 2],
        }
    )

    known = pandas.read_csv(
        io.StringIO(
            "hicov_aL19_employer_only,hicov_aL19_dpch_only,hicov_aL19_medicare_only,hicov_aL19_mcdmeans_only,hicov_aL19_trimil_only,"
            "hicov_aL19_va_only,hicov_aL19_employer_dpch,hicov_aL19_employer_medicare,hicov_aL19_medicare_mcdmeans,hicov_aL19_none,"
            "hicov_a19to34_employer_only,hicov_a19to34_dpch_only,hicov_a19to34_medicare_only,hicov_a19to34_mcdmeans_only,"
            "hicov_a19to34_trimil_only,hicov_a19to34_va_only,hicov_a19to34_employer_dpch,hicov_a19to34_employer_medicare,"
            "hicov_a19to34_medicare_mcdmeans,hicov_a19to34_none,hicov_a35to64_employer_only,hicov_a35to64_dpch_only,"
            "hicov_a35to64_medicare_only,hicov_a35to64_mcdmeans_only,hicov_a35to64_trimil_only,hicov_a35to64_va_only,"
            "hicov_a35to64_employer_dpch,hicov_a35to64_employer_medicare,hicov_a35to64_dpch_medicare,hicov_a35to64_medicare_mcdmeans,"
            "hicov_a35to64_none,hicov_aGE65_employer_only,hicov_aGE65_dpch_only,hicov_aGE65_medicare_only,hicov_aGE65_trimil_only,"
            "hicov_aGE65_va_only,hicov_aGE65_employer_dpch,hicov_aGE65_employer_medicare,hicov_aGE65_dpch_medicare,"
            "hicov_aGE65_medicare_mcdmeans,hicov_aGE65_none\n"
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
            "0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
        )
    )
    observed = livelike.pums.health_ins(gpp, 2019)
    pandas.testing.assert_frame_equal(observed, known)


def test_hhf():
    gph = pandas.DataFrame({"HFL": [1, 3, 5, 9]})

    known = pandas.read_csv(
        io.StringIO(
            "hhf_utility_gas,hhf_bottled_tank_LP,hhf_electricity,hhf_oil_kerosene_etc,hhf_coal_coke,hhf_wood,hhf_solar,hhf_other,hhf_no_fuel\nTrue,False,False,False,False,False,False,False,False\nFalse,False,True,False,False,False,False,False,False\nFalse,False,False,False,True,False,False,False,False\nFalse,False,False,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.hhf(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_hhinc():
    gph = pandas.DataFrame(
        {"HINCP": [-5, 9_999, 100_000, 200_000], "ADJINC": [10, 3, 1.5, 1.1]}
    )

    known = pandas.read_csv(
        io.StringIO(
            "hhinc_L10k,hhinc_10_15k,hhinc_15_20k,hhinc_20_25k,hhinc_25_30k,hhinc_30_35k,hhinc_35_40k,hhinc_40_45k,hhinc_45_50k,hhinc_50_60k,hhinc_60_75k,hhinc_75_100k,hhinc_100_125k,hhinc_125_150k,hhinc_150_200k,hhinc_GE200k\nTrue,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.hhinc(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_hhsize_vehicles():
    gph = pandas.DataFrame({"NP": [1, 3, 4, 5], "VEH": [1, 3, 4, 5]})

    known = pandas.read_csv(
        io.StringIO(
            "p01_no_vehicle,p01_01_vehicle,p01_02_vehicle,p01_03_vehicle,p01_GE04_vehicle,p02_no_vehicle,p02_01_vehicle,p02_02_vehicle,p02_03_vehicle,p02_GE04_vehicle,p03_no_vehicle,p03_01_vehicle,p03_02_vehicle,p03_03_vehicle,p03_GE04_vehicle,pGE04_no_vehicle,pGE04_01_vehicle,pGE04_02_vehicle,pGE04_03_vehicle,pGE04_GE04_vehicle\nFalse,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.hhsize_vehicles(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_hhtype():
    gph = pandas.DataFrame({"HHT": [1, 4, 7, 9]})

    known = pandas.read_csv(
        io.StringIO(
            "hht_alone,hht_female_no_spouse,hht_male_no_spouse,hht_married,hht_not_alone\nFalse,False,False,True,False\nTrue,False,False,False,False\nFalse,False,False,False,True\nFalse,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.hhtype(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_hhtype_hhsize():
    gph = pandas.DataFrame({"HHT": [0, 1, 2, 3, 4, 5], "NP": [5, 4, 3, 2, 1, 0]})

    known = pandas.read_csv(
        io.StringIO(
            "hht_fam_hhsize_2p,hht_fam_hhsize_3p,hht_fam_hhsize_4p,hht_fam_hhsize_5p,hht_fam_hhsize_6p,hht_fam_hhsize_7pm,hht_nonfam_hhsize_1p,hht_nonfam_hhsize_2p,hht_nonfam_hhsize_3p,hht_nonfam_hhsize_4p,hht_nonfam_hhsize_5p,hht_nonfam_hhsize_6p,hht_nonfam_hhsize_7pm\n0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,1,0,0,0,0,0,0,0,0,0,0\n0,1,0,0,0,0,0,0,0,0,0,0,0\n1,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,1,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0\n"  # noqa: E501
        )
    )
    observed = livelike.pums.hhtype_hhsize(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_housing_units():
    gph = pandas.DataFrame({"WGTP": [4, 2, 1, 0]})

    known = pandas.DataFrame({"housing_units": [1, 1, 1, 0]})
    observed = livelike.pums.housing_units(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_hsplat():
    gpp = pandas.DataFrame({"HISP": [0, 1, 2]})

    known = pandas.read_csv(
        io.StringIO("hsplat_no,hsplat_yes\nTrue,False\nTrue,False\nFalse,True\n")
    )
    observed = livelike.pums.hsplat(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_internet_2010s():
    gph = pandas.DataFrame({"ACCESS": [1, 2, 0, 1, 1, 3, 1]})
    year = 2019

    known_ = (
        "internet_subscription,internet_no_subscription,internet_none\n"
        "True,False,False\n"
        "False,True,False\n"
        "False,False,False\n"
        "True,False,False\n"
        "True,False,False\n"
        "False,False,True\n"
        "True,False,False\n"
    )
    known = pandas.read_csv(io.StringIO(known_))
    observed = livelike.pums.internet(gph, year)
    pandas.testing.assert_frame_equal(observed, known)


def test_internet_2020s():
    gph = pandas.DataFrame({"ACCESSINET": [1, 2, 0, 1, 1, 3, 1]})
    year = 2023

    known_ = (
        "internet_subscription,internet_no_subscription,internet_none\n"
        "True,False,False\n"
        "False,True,False\n"
        "False,False,False\n"
        "True,False,False\n"
        "True,False,False\n"
        "False,False,True\n"
        "True,False,False\n"
    )
    known = pandas.read_csv(io.StringIO(known_))
    observed = livelike.pums.internet(gph, year)
    pandas.testing.assert_frame_equal(observed, known)


def test_ipr():
    gpp = pandas.DataFrame({"POVPIP": [49, 99, 186, 201]})

    known = pandas.read_csv(
        io.StringIO(
            "ipr_L050,ipr_050_099,ipr_100_124,ipr_125_149,ipr_150_184,ipr_185_199,ipr_GE200\n1,0,0,0,0,0,0\n0,1,0,0,0,0,0\n0,0,0,0,0,1,0\n0,0,0,0,0,0,1\n"  # noqa: E501
        )
    )
    observed = livelike.pums.ipr(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_language():
    gph = pandas.DataFrame({"HHL": [1, 2, 4, 5]})

    known = pandas.read_csv(
        io.StringIO(
            "language_english,language_spanish,language_oth_indo_euro,language_asian_pac_isl,language_other\nTrue,False,False,False,False\nFalse,True,False,False,False\nFalse,False,False,True,False\nFalse,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.language(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_lep():
    gpp = pandas.DataFrame(
        {
            "RAC1P": [2, 4, 6, 10],
            "HISP": [0, 1, 0, 1],
            "POBP": [99, 99, 101, 101],
            "ENG": [1, 2, 1, 2],
        }
    )

    known = pandas.read_csv(
        io.StringIO(
            "race_white_foreign_born_eng_less_vwell,race_white_native_eng_less_vwell,race_blk_af_amer_foreign_born_eng_less_vwell,race_blk_af_amer_native_eng_less_vwell,race_native_amer_foreign_born_eng_less_vwell,race_native_amer_native_eng_less_vwell,race_asian_foreign_born_eng_less_vwell,race_asian_native_eng_less_vwell,race_pac_island_foreign_born_eng_less_vwell,race_pac_island_native_eng_less_vwell,race_other_foreign_born_eng_less_vwell,race_other_native_eng_less_vwell,race_mult_foreign_born_eng_less_vwell,race_mult_native_eng_less_vwell\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,True,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,True,False\n"  # noqa: E501
        )
    )
    observed = livelike.pums.lep(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_lingisol():
    gpp = pandas.DataFrame({"LNGI": [1, 2]})

    known = pandas.Series([False, True], name="lngi_yes")
    observed = livelike.pums.lingisol(gpp)
    pandas.testing.assert_series_equal(observed, known)


def test_minors():
    gph = pandas.DataFrame({"R18": [0, 1, 2]})

    known = pandas.read_csv(
        io.StringIO("minors_no,minors_yes\nTrue,False\nFalse,True\nFalse,False\n")
    )
    observed = livelike.pums.minors(gph)
    pandas.testing.assert_frame_equal(observed, known)


@pytest.mark.parametrize("column, year", [("TYPE", 2019), ("TYPEHUGQ", 2020)])
def test_occhu(column, year):
    gph = pandas.DataFrame({column: [1, 3, 1], "VACS": [0, 0, 1]})

    known = pandas.Series([1, 0, 0], name="occhu")
    observed = livelike.pums.occhu(gph, year)
    pandas.testing.assert_series_equal(observed, known)


def test_population():
    gpp = pandas.DataFrame(
        {
            "SERIALNO": ["A101", "A101", "B202"],
            "SPORDER": [1, 2, 1],
            "PWGTP": [10, 20, 5],
        }
    )

    known = pandas.Series([1.0, 2.0, 1.0], name="population")
    observed = livelike.pums.population(gpp)
    pandas.testing.assert_series_equal(observed, known)


def test_owncost():
    gph = pandas.DataFrame({"TEN": [1, 2, 1, 2], "OCPIP": [10, 25, 35, 51]})

    known = pandas.read_csv(
        io.StringIO(
            "owncost_mortgage_less10,owncost_mortgage_10to14.9,owncost_mortgage_15to19.9,owncost_mortgage_20to24.9,owncost_mortgage_25to29.9,owncost_mortgage_30to34.9,owncost_mortgage_35to39.9,owncost_mortgage_40to49.9,owncost_mortgage_more50,owncost_nomortgage_less10,owncost_nomortgage_10to14.9,owncost_nomortgage_15to19.9,owncost_nomortgage_20to24.9,owncost_nomortgage_25to29.9,owncost_nomortgage_30to34.9,owncost_nomortgage_35to39.9,owncost_nomortgage_40to49.9,owncost_nomortgage_more50\n0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0\n0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1\n"  # noqa: E501
        )
    )
    observed = livelike.pums.owncost(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_poverty():
    gpp = pandas.DataFrame({"POVPIP": [numpy.nan, -1, 25, 50]})

    known = pandas.read_csv(
        io.StringIO(
            "poverty_no,poverty_yes\nTrue,False\nFalse,False\nFalse,True\nFalse,True\n"
        )
    )
    observed = livelike.pums.poverty(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_race():
    gpp = pandas.DataFrame({"RAC1P": [2, 4, 6, 10]})

    known = pandas.read_csv(
        io.StringIO(
            "race_white,race_blk_af_amer,race_native_amer,race_asian,race_pac_island,race_other,race_mult\nFalse,True,False,False,False,False,False\nFalse,False,True,False,False,False,False\nFalse,False,False,True,False,False,False\nFalse,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.race(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_rentcost():
    gph = pandas.DataFrame({"GRPIP": [5, 15, 25, 51]})

    known = pandas.read_csv(
        io.StringIO(
            "rentcost_less10,rentcost_10to14.9,rentcost_15to19.9,rentcost_20to24.9,rentcost_25to29.9,rentcost_30to34.9,rentcost_35to39.9,rentcost_40to49.9,rentcost_more50\n1,0,0,0,0,0,0,0,0\n0,0,1,0,0,0,0,0,0\n0,0,0,0,1,0,0,0,0\n0,0,0,0,0,0,0,0,1\n"  # noqa: E501
        )
    )
    observed = livelike.pums.rentcost(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_rooms():
    gph = pandas.DataFrame({"RMSP": [2, 5, 8, 10]})

    known = pandas.read_csv(
        io.StringIO(
            "rooms_01,rooms_02,rooms_03,rooms_04,rooms_05,rooms_06,rooms_07,rooms_08,rooms_GE09\nFalse,True,False,False,False,False,False,False,False\nFalse,False,False,False,True,False,False,False,False\nFalse,False,False,False,False,False,False,True,False\nFalse,False,False,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.rooms(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_school():
    gpp = pandas.DataFrame(
        {"SEX": [2, 1, 2, 1], "SCH": [2, 2, 3, 3], "SCHG": [5, 8, 12, 16]}
    )

    known = pandas.read_csv(
        io.StringIO(
            "sch_female_pre_public,sch_female_pre_private,sch_female_kind_public,sch_female_kind_private,sch_female_01.04_public,sch_female_01.04_private,sch_female_05.08_public,sch_female_05.08_private,sch_female_09.12_public,sch_female_09.12_private,sch_female_undergrad_public,sch_female_undergrad_private,sch_female_grad.prof_public,sch_female_grad.prof_private,sch_male_pre_public,sch_male_pre_private,sch_male_kind_public,sch_male_kind_private,sch_male_01.04_public,sch_male_01.04_private,sch_male_05.08_public,sch_male_05.08_private,sch_male_09.12_public,sch_male_09.12_private,sch_male_undergrad_public,sch_male_undergrad_private,sch_male_grad.prof_public,sch_male_grad.prof_private\nFalse,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.school(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_seniors():
    gph = pandas.DataFrame({"R60": [0, 1, 2, 3]})

    known = pandas.read_csv(
        io.StringIO(
            "seniors_no,seniors_yes\nTrue,False\nFalse,True\nFalse,True\nFalse,False\n"
        )
    )
    observed = livelike.pums.seniors(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_sex():
    gpp = pandas.DataFrame({"SEX": [2, 1, 2]})

    known = pandas.DataFrame(
        {"female": [True, False, True], "male": [False, True, False]}
    )
    observed = livelike.pums.sex(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_sex_age():
    gpp = pandas.DataFrame({"SEX": [2, 1, 2, 1], "AGEP": [4, 20, 39, 55]})

    known = pandas.read_csv(
        io.StringIO(
            "female_a05u,female_a05_09,female_a10_14,female_a15_17,female_a18_19,female_a20,female_a21,female_a22_24,female_a25_29,female_a30_34,female_a35_39,female_a40_44,female_a45_49,female_a50_54,female_a55_59,female_a60_61,female_a62_64,female_a65_66,female_a67_69,female_a70_74,female_a75_79,female_a80_84,female_a85o,male_a05u,male_a05_09,male_a10_14,male_a15_17,male_a18_19,male_a20,male_a21,male_a22_24,male_a25_29,male_a30_34,male_a35_39,male_a40_44,male_a45_49,male_a50_54,male_a55_59,male_a60_61,male_a62_64,male_a65_66,male_a67_69,male_a70_74,male_a75_79,male_a80_84,male_a85o\nTrue,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False\n"  # noqa: E501
        )
    )
    observed = livelike.pums.sex_age(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_sexcw():
    gpp = pandas.DataFrame(
        {"SEX": [2, 1, 2, 1], "ESR": [1, 3, 4, 6], "COW": [2, 4, 6, 8]}
    )

    known = pandas.read_csv(
        io.StringIO(
            "sexcw_female_private,sexcw_female_non.prof,sexcw_female_local.gov,sexcw_female_st.gov,sexcw_female_fed.gov,sexcw_female_self.emp_non.inc,sexcw_female_self.emp_inc,sexcw_female_wo.pay,sexcw_male_private,sexcw_male_non.prof,sexcw_male_local.gov,sexcw_male_st.gov,sexcw_male_fed.gov,sexcw_male_self.emp_non.inc,sexcw_male_self.emp_inc,sexcw_male_wo.pay\nFalse,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False\n"  # noqa: E501
        )
    )
    observed = livelike.pums.sexcw(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_sexnaics():
    gpp = pandas.DataFrame(
        {
            "SEX": [2, 1, 2, 1],
            "ESR": [1, 3, 4, 6],
            "NAICSP": ["61", "44", "71", "9281"],
        }
    )

    known = pandas.read_csv(
        io.StringIO(
            "sexnaics_female_adm,sexnaics_female_agr_ext,sexnaics_female_con,sexnaics_female_edu_med_sca,sexnaics_female_ent,sexnaics_female_fin,sexnaics_female_inf,sexnaics_female_mfg,sexnaics_female_mil,sexnaics_female_prf,sexnaics_female_ret,sexnaics_female_srv,sexnaics_female_utl_trn,sexnaics_female_whl,sexnaics_male_adm,sexnaics_male_agr_ext,sexnaics_male_con,sexnaics_male_edu_med_sca,sexnaics_male_ent,sexnaics_male_fin,sexnaics_male_inf,sexnaics_male_mfg,sexnaics_male_mil,sexnaics_male_prf,sexnaics_male_ret,sexnaics_male_srv,sexnaics_male_utl_trn,sexnaics_male_whl\n0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"  # noqa: E501
        )
    )
    observed = livelike.pums.sexnaics(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_sexnaics_det():
    gpp = pandas.DataFrame(
        {
            "SEX": [2, 1, 2, 1],
            "ESR": [1, 3, 4, 6],
            "NAICSP": ["61", "44", "71", "9281"],
        }
    )

    known = pandas.read_csv(
        io.StringIO(
            "sexnaics_det_female_agr_ffh,sexnaics_det_female_ext,sexnaics_det_female_con,sexnaics_det_female_mfg,sexnaics_det_female_whl,sexnaics_det_female_ret,sexnaics_det_female_trn_whs,sexnaics_det_female_utl,sexnaics_det_female_inf,sexnaics_det_female_fin_ins,sexnaics_det_female_rrl,sexnaics_det_female_prf,sexnaics_det_female_mgt,sexnaics_det_female_adm_wmr,sexnaics_det_female_edu,sexnaics_det_female_med_sca,sexnaics_det_female_ent,sexnaics_det_female_afs,sexnaics_det_female_srv,sexnaics_det_female_pad,sexnaics_det_male_agr_ffh,sexnaics_det_male_ext,sexnaics_det_male_con,sexnaics_det_male_mfg,sexnaics_det_male_whl,sexnaics_det_male_ret,sexnaics_det_male_trn_whs,sexnaics_det_male_utl,sexnaics_det_male_inf,sexnaics_det_male_fin_ins,sexnaics_det_male_rrl,sexnaics_det_male_prf,sexnaics_det_male_mgt,sexnaics_det_male_adm_wmr,sexnaics_det_male_edu,sexnaics_det_male_med_sca,sexnaics_det_male_ent,sexnaics_det_male_afs,sexnaics_det_male_srv,sexnaics_det_male_pad\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"  # noqa: E501
        )
    )
    observed = livelike.pums.sexnaics_det(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_sexocc():
    gpp = pandas.DataFrame(
        {
            "SEX": [2, 1, 2, 1],
            "ESR": [1, 3, 4, 6],
            "OCCP": ["000", "222", "133", "199"],
        }
    )

    known = pandas.read_csv(
        io.StringIO(
            "sexocc_female_mgmt,sexocc_female_bus.fin,sexocc_female_cmp,sexocc_female_eng,sexocc_female_sci,sexocc_female_cms,sexocc_female_lgl,sexocc_female_edu,sexocc_female_ent,sexocc_female_med,sexocc_female_med.tech,sexocc_female_hls,sexocc_female_prt,sexocc_female_law.enf,sexocc_female_eat,sexocc_female_cln,sexocc_female_prs,sexocc_female_sal,sexocc_female_off,sexocc_female_fff,sexocc_female_con.ext,sexocc_female_rpr,sexocc_female_prd,sexocc_female_trn,sexocc_female_trn.mat,sexocc_male_mgmt,sexocc_male_bus.fin,sexocc_male_cmp,sexocc_male_eng,sexocc_male_sci,sexocc_male_cms,sexocc_male_lgl,sexocc_male_edu,sexocc_male_ent,sexocc_male_med,sexocc_male_med.tech,sexocc_male_hls,sexocc_male_prt,sexocc_male_law.enf,sexocc_male_eat,sexocc_male_cln,sexocc_male_prs,sexocc_male_sal,sexocc_male_off,sexocc_male_fff,sexocc_male_con.ext,sexocc_male_rpr,sexocc_male_prd,sexocc_male_trn,sexocc_male_trn.mat\n1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"  # noqa: E501
        )
    )
    observed = livelike.pums.sexocc(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_tenure():
    gph = pandas.DataFrame({"TEN": [0, 1, 2, 3, 4, 5]})

    known = pandas.read_csv(
        io.StringIO(
            "tenure_own,tenure_rent\nFalse,False\nTrue,False\nTrue,False\nFalse,True\nFalse,False\nFalse,False\n"  # noqa: E501
        )
    )
    observed = livelike.pums.tenure(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_tenure_vehicles():
    gph = pandas.DataFrame({"TEN": [1, 2, 3], "VEH": [1, 2, 3]})

    known = pandas.read_csv(
        io.StringIO(
            "txv_own_no_vehicle,txv_own_01_vehicle,txv_own_02_vehicle,txv_own_03_vehicle,txv_own_04_vehicle,txv_own_GE05_vehicle,txv_rent_no_vehicle,txv_rent_01_vehicle,txv_rent_02_vehicle,txv_rent_03_vehicle,txv_rent_04_vehicle,txv_rent_GE05_vehicle\nFalse,True,False,False,False,False,False,False,False,False,False,False\nFalse,False,True,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,True,False,False\n"  # noqa: E501
        )
    )
    observed = livelike.pums.tenure_vehicles(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_travel():
    gpp = pandas.DataFrame({"JWTRNS": [1, 5, 11, 13]})

    known = pandas.read_csv(
        io.StringIO(
            "travel_car_truck_van,travel_public_transportation,travel_taxicab,travel_motorcycle,travel_bicycle,travel_walked,travel_wfh,travel_other\nTrue,False,False,False,False,False,False,False\nFalse,True,False,False,False,False,False,False\nFalse,False,False,False,False,False,True,False\nFalse,False,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.travel(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_units():
    gph = pandas.DataFrame({"BLD": [1, 6, 11]})

    known = pandas.read_csv(
        io.StringIO(
            "dwg_mob_home,dwg_single_fam_detach,dwg_single_fam_attach,dwg_2_unit,dwg_3_4_unit,dwg_5_9_unit,dwg_10_19_unit,dwg_20_49_unit,dwg_GE50_unit,dwg_other\nTrue,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,True,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,False\n"  # noqa: E501
        )
    )
    observed = livelike.pums.units(gph)
    pandas.testing.assert_frame_equal(observed, known)


def test_veh_occ():
    gpp = pandas.DataFrame({"JWRIP": [1, 2, 3]})

    known = pandas.read_csv(
        io.StringIO(
            "veh_occ_drove_alone,veh_occ_carpooled\nTrue,False\nFalse,True\nFalse,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.veh_occ(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_veteran():
    gpp = pandas.DataFrame({"VPS": [0, 1, 1], "MIL": [0, 1, 2], "AGEP": [17, 18, 30]})

    known = pandas.read_csv(io.StringIO("nonvet,vet\n0,0\n1,0\n0,1\n\n"))
    observed = livelike.pums.veteran(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_vet_edu():
    gpp = pandas.DataFrame(
        {
            "VPS": [0, 1, 1],
            "MIL": [0, 1, 2],
            "AGEP": [17, 25, 30],
            "SCHL": [15, 17, 22],
        }
    )

    known = pandas.read_csv(
        io.StringIO(
            "nonvet_less_hs,nonvet_hs,nonvet_some_clg,nonvet_bach_higher,vet_less_hs,vet_hs,vet_some_clg,vet_bach_higher\n0,0,0,0,0,0,0,0\n0,1,0,0,0,0,0,0\n0,0,0,0,0,0,0,1\n"  # noqa: E501
        )
    )
    observed = livelike.pums.vet_edu(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_worked():
    gpp = pandas.DataFrame({"SEX": [2, 1, 2, 1], "WKHP": [9, 20, 50, 34]})

    known = pandas.read_csv(
        io.StringIO(
            "female_hours_1.14,female_hours_15.34,female_hours_GE35,male_hours_1.14,male_hours_15.34,male_hours_GE35\nTrue,False,False,False,False,False\nFalse,False,False,False,True,False\nFalse,False,True,False,False,False\nFalse,False,False,False,True,False\n"  # noqa: E501
        )
    )
    observed = livelike.pums.worked(gpp)
    pandas.testing.assert_frame_equal(observed, known)


def test_year_built():
    gph = pandas.DataFrame({"YBL": [1, 6, 19]})

    known = pandas.read_csv(
        io.StringIO(
            "year_built_L1939,year_built_40_49,year_built_50_59,year_built_60_69,year_built_70_79,year_built_80_89,year_built_90_99,year_built_00_09,year_built_10_13,year_built_GE2014\nTrue,False,False,False,False,False,False,False,False,False\nFalse,False,False,False,False,True,False,False,False,False\nFalse,False,False,False,False,False,False,False,False,True\n"  # noqa: E501
        )
    )
    observed = livelike.pums.year_built(gph, year=2019)
    pandas.testing.assert_frame_equal(observed, known)
