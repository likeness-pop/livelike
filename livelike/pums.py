import re

import numpy as np
import pandas as pd

# Helper functions #


def intersect_dummies(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Intersects two data frames consisting of dummy variables.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Target data frame.
    df2 : pandas.DataFrame
        Intersect data frame. Column name prefixes in the result will
        consist of the input column names.

    Returns
    -------
    intersect_df : pandas.DataFrame
        Intersected data from ``df1`` and ``df2``.
    """

    # create dummies
    intersect_df = pd.DataFrame()

    for col in df2.columns:
        ixt = pd.DataFrame(df2[col].values[:, None] * df1.values)
        ixt.columns = col + "_" + df1.columns
        intersect_df = pd.concat([intersect_df, ixt], axis=1)

    return intersect_df


def reclass_dummies(df: pd.DataFrame, v: str, rc: dict) -> pd.DataFrame:
    """Reclassifies ACS PUMS variable item levels based on a user-specified
    dictionary. Returns a data frame of dummy variables representing the new
    item levels.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame.
    v : str
        Target variable name.
    rc : dict
        A dictionary whose keys consist of the new item levels for
        reclassificaiton and whose values contain regular expressions for
        string matching on the target variable.

    Returns
    -------
    df_rc : pandas.DataFrame
        Reclassified ``df``.
    """

    df_rc = pd.DataFrame()

    for key, value in rc.items():
        df_rc[key] = df[v].str.match(value).astype("int")

    return df_rc


# Constraint recode functions #


def age_cohort(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of age cohort that harmonizes
    ACS PUMS questionnaire item AGEP with ACS SF table B06001.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing an AGEP column.

    Returns
    -------
    aco : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS age cohort
        categories based on ACS SF table B06001.
    """

    cohort_desc = [
        "L5",
        "5_17",
        "18_24",
        "25_34",
        "35_44",
        "45_54",
        "55_59",
        "60_61",
        "62_64",
        "65_74",
        "GE75",
    ]

    aco = pd.cut(
        gpp["AGEP"],
        bins=(0, 5, 18, 25, 35, 45, 55, 60, 62, 65, 75, np.inf),
        labels=cohort_desc,
        right=False,
    )

    aco = pd.get_dummies(aco, prefix="age")

    return aco


def age_simple(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of simplified age that harmonizes
    ACS PUMS questionnaire item AGEP with ACS SF table B23027.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing an AGEP column.

    Returns
    -------
    agep : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS simplified age
        categories based on ACS SF table B23027.
    """

    age_lv = ["16_19", "20_24", "25_44", "45_54", "55_64", "65_69", "GE70"]

    agep = pd.cut(
        gpp["AGEP"],
        bins=(16, 20, 25, 45, 55, 65, 70, np.inf),
        labels=age_lv,
        right=False,
    )

    agep = pd.get_dummies(agep, prefix="age")

    return agep


def bedrooms(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of bedrooms in dwelling that
    harmonizes ACS PUMS questionnaire item BDSP with ACS SF table B08201.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing a BDSP column.

    Returns
    -------
    bdsp : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS bedrooms in dwelling
        categories based on ACS SF table B08201.
    """

    bdsp_desc = ["00", "01", "02", "03", "04", "GE05"]
    bdsp = pd.cut(
        gph["BDSP"], bins=(0, 1, 2, 3, 4, 5, np.inf), labels=bdsp_desc, right=False
    )
    bdsp = pd.get_dummies(bdsp, prefix="bedr")

    return bdsp


def civ_noninst_pop(gpp: pd.DataFrame, year: int | str) -> pd.Series:
    """Generates a person-level flag for civilian non-institutionalized population
    that harmonizes ACS PUMS questionnaire items TYPE (2015 - 2019) and TYPEHUGQ
    (2020+) with ACS SF variable B27010001.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing TYPE or TYPEHUGQ column.
    year : int | str
        ACS 5-Year Estimates vintage. Not optional, here, but defaults
        as ``2019`` passed in from ``acs.build_acs_pums_inputs()``).

    Returns
    -------
    cni : pandas.Series
        One-hot encoded flag for civilian non-institutionalized
        population based on ACS SF variable B27010001.
    """

    hh_type_var = "TYPE" if int(year) < 2020 else "TYPEHUGQ"

    cni = (gpp.loc[:, hh_type_var] != 2).astype("int")

    cni = cni.rename("civ_noninst_pop")

    return cni


def commute(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of commute time to work that
    harmonizes ACS PUMS questionnaire item JWMNP with ACS SF table B08134.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a JWMNP column.

    Returns
    -------
    jwmnp : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS JWMNP
        categories based on ACS SF table B08134.
    """

    lv = ["L10", "10.14", "15.19", "20.24", "25.29", "30.34", "35.44", "45.59", "GE60"]

    jwmnp = pd.cut(
        gpp["JWMNP"],
        bins=(1, 10, 15, 20, 25, 30, 35, 45, 60, np.inf),
        labels=["cmt_mins_" + i for i in lv],
        right=False,
    )

    jwmnp = pd.get_dummies(jwmnp)

    return jwmnp


def disability(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of age by
    disability status that harmonizes ACS PUMS questionnaire
    items AGEP and DIS with ACS SF table B18135.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing AGEP and DIS columns.

    Returns
    -------
    axd : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS age by disability
        status categories based on ACS SF table B18135.
    """

    age_desc = ["L19", "19_64", "GE65"]
    age = pd.cut(gpp["AGEP"], bins=(0, 19, 65, np.inf), labels=age_desc, right=False)
    age = pd.get_dummies(age, prefix="a", prefix_sep="")
    age.columns = age.columns.astype("str")

    dis = pd.get_dummies(np.where(gpp["DIS"] == 1, "disability", "no_disability"))
    dis.columns = dis.columns.astype("str")

    axd = intersect_dummies(dis, age)

    return axd


def edu_attainment(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of educational attainment
    for population 25+ sex by age that harmonizes ACS PUMS questionnaire
    item SCHL and AGEP with ACS SF table B15001.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing SEX, SCHL, and AGEP columns.

    Returns
    -------
    schl : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS educational attainment
        by age categories based on ACS SF table B01001.
    """

    school_lv = [
        "noschool",
        "nurseryto4thgrade",
        "5thand6th",
        "7thand8th",
        "9th",
        "10th",
        "11th",
        "12th",
        "highschoolgrad",
        "less1yearcollege",
        "collegenodeg",
        "assoc",
        "bach",
        "masters",
        "prof",
        "doc",
    ]

    aGE25 = np.where(gpp["AGEP"] >= 25, 1, 0)  # noqa N806

    school = pd.cut(
        gpp["SCHL"],
        bins=(1, 2, 8, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, np.inf),
        labels=school_lv,
        right=False,
    )

    schl = pd.get_dummies(school)
    gnd = sex(gpp)

    schl.columns = schl.columns.astype("str")

    edu_att = intersect_dummies(schl, gnd)
    edu_att.columns = [f"schl_{col}" for col in edu_att.columns]
    edu_att = edu_att.multiply(aGE25, axis=0)

    return edu_att


def emp_stat(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of employment status that
    harmonizes ACS PUMS questionnaire item ESR with ACS SF table B23025.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing an ESR column.

    Returns
    -------
    esr : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS employment
        status categories based on ACS SF table B23025.
    """

    esr = pd.cut(
        gpp["ESR"],
        bins=(1, 3, 4, 6, np.inf),
        labels=["employed", "unemp", "mil", "not.in.force"],
        right=False,
    )

    esr = pd.get_dummies(esr, prefix="emp_stat")

    return esr


def foreign_born(gpp: pd.DataFrame) -> pd.Series:
    """Generates a person-level flag for foreign born status that
    harmonizes ACS PUMS questionnaire item POBP with ACS SF table B06007.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a POBP column.

    Returns
    -------
    fbn : pandas.Series
        One-hot encoded flag for ACS PUMS foreign born status
        categories based on ACS SF table B06007.
    """

    fbn = (gpp["POBP"] >= 100).astype("int")

    return fbn.rename("foreign_born")


def grade(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of school grade level attending that
    harmonizes ACS PUMS questionnaire item SCHG with ACS SF table B14007.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a SCHG column.

    Returns
    -------
    grade : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS school grade level
        attending categories based on ACS SF table B14007.
    """

    grade = pd.cut(
        gpp["SCHG"],
        bins=np.append(np.arange(1, 17), np.inf),
        labels=[
            "preschl",
            "kind",
            "1st",
            "2nd",
            "3rd",
            "4th",
            "5th",
            "6th",
            "7th",
            "8th",
            "9th",
            "10th",
            "11th",
            "12th",
            "undergrad",
            "grad",
        ],
        right=False,
    )

    grade = pd.get_dummies(grade, prefix="grade")

    return grade


def group_quarters(gpp: pd.DataFrame, year: int | str) -> pd.Series:
    """Generates a person-level flag for group
    quarters population that harmonizes ACS PUMS
    questionnaire items TYPE (2016 - 2019) and TYPEHUGQ
    (2020+) with ACS SF variable B26001001.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a TYPE or TYPEHUGQ column.
    year : int | str
        ACS 5-Year Estimates vintage. Not optional, here, but defaults
        as ``2019`` passed in from ``acs.build_acs_pums_inputs()``.

    Returns
    -------
    gq : pandas.Series
        One-hot encoded group quarters flag based on ACS SF variable B26001001.
    """

    hh_type_var = "TYPE" if int(year) < 2020 else "TYPEHUGQ"

    ## Group quarters population
    gq = (gpp.loc[:, hh_type_var] > 1).astype("int")

    return gq.rename("group_quarters")


def group_quarters_pop(gpp: pd.DataFrame, year: str | int) -> pd.Series:
    """Generates a person-level flag for group
    quarters population that harmonizes ACS PUMS
    questionnaire items RELP (2016 - 2018) and RELSHIPP
    (2019+) with ACS SF variable B26001001.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a RELP or RELSHIPP column.
    year : int | str
        ACS 5-Year Estimates vintage. Not optional, here, but defaults
        as ``2019`` passed in from ``acs.build_acs_pums_inputs()``.

    Returns
    -------
    gq : pandas.Series
        One-hot encoded group quarters flag based on ACS SF variable B26001001.
    """

    if int(year) <= 2018:
        gq = (gpp["RELP"] >= 16).astype("int")

    else:
        gq = (gpp["RELSHIPP"] >= 37).astype("int")

    gq = gq.rename("group_quarters_pop")

    return gq


def health_ins(gpp: pd.DataFrame, year: int | str) -> pd.DataFrame:
    """Generates a person-level flag for health insurance
    coverage by age that harmonizes ACS PUMS questionnaire
    items AGEP and HINS1-7 with ACS SF table B27010.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing AGEP, HICOV, and HINS1-7 columns.
    year : int | str
        ACS 5-Year Estimates vintage. Not optional, here, but defaults
        as ``2019`` passed in from ``acs.build_acs_pums_inputs()``.

    Returns
    -------
    hxa : pandas.Series
        One-hot encoded DataFrame of ACS PUMS health insurance
        by age categories based on ACS SF table B27010.

    Notes
    -----
    * ``'HINS_NR'`` is created internally to record "other health insurance" for
        responses not covered in the survey. It also accounts for one type of known
        insurance "only" vs. in combination with some other types.
    * ``'HINS7'`` represents Indian Health Service coverage. Its most direct usage
        here is determining whether a person has a known type of coverage (employer,
        medicare, medicate, TRICARE, VA) only or in combination with other insurance
        types (see also ``'HINS_NR'``).
    """

    # Recode PUMS HINS items (`1 : Yes`, `2 : No`) as one-hot
    hins = 2 - gpp.loc[:, gpp.columns.str.contains("^HINS")]

    # make an "other health insurance" col
    # for other types of coverage not in survey
    hins["HINS_NR"] = 1 * ((hins.sum(axis=1) == 0) & (gpp["HICOV"] == 1))

    ## Flags ##
    # Employer-based coverage #
    employer_hins_cols = ["HINS1"]

    # any employer based coverage
    hins_employer_based = hins.loc[:, employer_hins_cols]

    # any coverage other than employer-based
    hins_otherthan_employer = 1 * (
        hins.loc[:, ~hins.columns.isin(employer_hins_cols)].sum(axis=1) > 0
    )

    # employer-based only
    hins_employer_only = (
        hins_employer_based.values.flatten() * abs(hins_otherthan_employer - 1).values
    )
    hins_employer_only = pd.Series(hins_employer_only, name="employer_only")

    # Direct-purchase coverage #
    dpch_hins_cols = ["HINS2"]

    # any dpch based coverage
    hins_dpch_based = hins.loc[:, dpch_hins_cols]

    # any coverage other than dpch-based
    hins_otherthan_dpch = 1 * (
        hins.loc[:, ~hins.columns.isin(dpch_hins_cols)].sum(axis=1) > 0
    )

    # dpch-based only
    hins_dpch_only = (
        hins_dpch_based.values.flatten() * abs(hins_otherthan_dpch - 1).values
    )
    hins_dpch_only = pd.Series(hins_dpch_only, name="dpch_only")

    # Medicare #
    medicare_hins_cols = ["HINS3"]

    # any medicare based coverage
    hins_medicare_based = hins.loc[:, medicare_hins_cols]

    # any coverage other than medicare-based
    hins_otherthan_medicare = 1 * (
        hins.loc[:, ~hins.columns.isin(medicare_hins_cols)].sum(axis=1) > 0
    )

    # medicare-based only
    hins_medicare_only = (
        hins_medicare_based.values.flatten() * abs(hins_otherthan_medicare - 1).values
    )
    hins_medicare_only = pd.Series(hins_medicare_only, name="medicare_only")

    # Medicaid/Means-Tested #
    mcdmeans_hins_cols = ["HINS4"]

    # any mcdmeans based coverage
    hins_mcdmeans_based = hins.loc[:, mcdmeans_hins_cols]

    # any coverage other than mcdmeans-based
    hins_otherthan_mcdmeans = 1 * (
        hins.loc[:, ~hins.columns.isin(mcdmeans_hins_cols)].sum(axis=1) > 0
    )

    # mcdmeans-based only
    hins_mcdmeans_only = (
        hins_mcdmeans_based.values.flatten() * abs(hins_otherthan_mcdmeans - 1).values
    )
    hins_mcdmeans_only = pd.Series(hins_mcdmeans_only, name="mcdmeans_only")

    # TRICARE/Military #
    trimil_hins_cols = ["HINS5"]

    # any trimil based coverage
    hins_trimil_based = hins.loc[:, trimil_hins_cols]

    # any coverage other than trimil-based
    hins_otherthan_trimil = 1 * (
        hins.loc[:, ~hins.columns.isin(trimil_hins_cols)].sum(axis=1) > 0
    )

    # trimil-based only
    hins_trimil_only = (
        hins_trimil_based.values.flatten() * abs(hins_otherthan_trimil - 1).values
    )
    hins_trimil_only = pd.Series(hins_trimil_only, name="trimil_only")

    # VA Healthcare #
    va_hins_cols = ["HINS6"]

    # any va based coverage
    hins_va_based = hins.loc[:, va_hins_cols]

    # any coverage other than va-based
    hins_otherthan_va = 1 * (
        hins.loc[:, ~hins.columns.isin(va_hins_cols)].sum(axis=1) > 0
    )

    # va-based only
    hins_va_only = hins_va_based.values.flatten() * abs(hins_otherthan_va - 1).values
    hins_va_only = pd.Series(hins_va_only, name="va_only")

    # Two or more: Employer and Direct-purchase #
    employer_dpch_cols = employer_hins_cols + dpch_hins_cols
    hins_employer_dpch = 1 * (
        (hins.loc[:, employer_dpch_cols].sum(axis=1) == 2)
        & (hins.loc[:, ~hins.columns.isin(employer_dpch_cols)].sum(axis=1) == 0)
    )
    hins_employer_dpch.name = "employer_dpch"

    # Two or more: Employer and Medicare #
    employer_medicare_cols = employer_hins_cols + medicare_hins_cols
    hins_employer_medicare = 1 * (
        (hins.loc[:, employer_medicare_cols].sum(axis=1) == 2)
        & (hins.loc[:, ~hins.columns.isin(employer_medicare_cols)].sum(axis=1) == 0)
    )
    hins_employer_medicare.name = "employer_medicare"

    # Two ore more: Direct-purchase and Medicare #
    dpch_medicare_cols = dpch_hins_cols + medicare_hins_cols
    hins_dpch_medicare = 1 * (
        (hins.loc[:, dpch_medicare_cols].sum(axis=1) == 2)
        & (hins.loc[:, ~hins.columns.isin(dpch_medicare_cols)].sum(axis=1) == 0)
    )
    hins_dpch_medicare.name = "dpch_medicare"

    # Two or more: Medicare and Medicaid/Means-Tested #
    medicare_mcdmeans_cols = medicare_hins_cols + mcdmeans_hins_cols
    hins_medicare_mcdmeans = 1 * (
        (hins.loc[:, medicare_mcdmeans_cols].sum(axis=1) == 2)
        & (hins.loc[:, ~hins.columns.isin(medicare_mcdmeans_cols)].sum(axis=1) == 0)
    )
    hins_medicare_mcdmeans.name = "medicare_mcdmeans"

    # No coverage #
    # hins_none = 1 * (gpp["HICOV"] == 2)
    hins_none = 1 * (hins.sum(axis=1) == 0)
    hins_none.name = "none"

    ## Combine flags ##
    hicov = pd.concat(
        [
            hins_employer_only,
            hins_dpch_only,
            hins_medicare_only,
            hins_mcdmeans_only,
            hins_trimil_only,
            hins_va_only,
            hins_employer_dpch,
            hins_employer_medicare,
            hins_dpch_medicare,
            hins_medicare_mcdmeans,
            hins_none,
        ],
        axis=1,
    )

    ## Age cohorts ##
    age_levels = ["aL19", "a19to34", "a35to64", "aGE65"]

    agep = pd.cut(
        gpp["AGEP"],
        bins=(0, 19, 35, 65, np.inf),
        labels=age_levels,
        right=False,
    )
    agep = pd.get_dummies(agep)

    ## Combine dfs ##
    hxa = intersect_dummies(hicov, agep)
    hxa.columns = [f"hicov_{col}" for col in hxa.columns]

    # drop cols not in b27010
    hxa = hxa.drop(
        [
            "hicov_aL19_dpch_medicare",
            "hicov_a19to34_dpch_medicare",
            "hicov_aGE65_mcdmeans_only",
        ],
        axis=1,
    )

    # limit to civilian noninst pop
    # to match b27010 universe
    cnp = civ_noninst_pop(gpp, int(year))
    hxa = hxa * cnp.values[:, None]

    return hxa


def hhf(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of house heating fuel that
    harmonizes ACS PUMS questionnaire item HFL with ACS SF table B25040.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses including a HFL column.

    Returns
    -------
    hfl : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS HFL
        categories based on ACS SF table B25040.
    """

    hfl_lv = [
        "utility_gas",
        "bottled_tank_LP",
        "electricity",
        "oil_kerosene_etc",
        "coal_coke",
        "wood",
        "solar",
        "other",
        "no_fuel",
    ]

    hfl = pd.cut(
        gph["HFL"],
        bins=(1, 2, 3, 4, 5, 6, 7, 8, 9, np.inf),
        labels=hfl_lv,
        right=False,
        ordered=False,
    )

    hfl = pd.get_dummies(hfl, prefix="hhf")

    return hfl


def hhinc(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of household income, adjusted
    for inflation in the ACS 5Y release year, that harmonizes ACS PUMS
    questionnaire item HINCP with ACS SF table B19001.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing HINCP and ADJINC columns.

    Returns
    -------
    hincp : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS household
        type categories based on ACS SF table B19001.
    """

    # apply income adjustment
    HINCP_unadj = gph["HINCP"].values  # noqa N806
    gph["HINCP"] = gph["HINCP"] * gph["ADJINC"]

    # if an income level (hh w negative cash income)
    # got knocked below -60k (placeholder for OOU cases)
    # during income adjustment, convert it to the lowest
    # valid value for in-universe cases
    gph.loc[(gph["HINCP"] <= -60000) & (HINCP_unadj > -60000), "HINCP"] = -59999

    hincp_lv = [
        "L10k",
        "10_15k",
        "15_20k",
        "20_25k",
        "25_30k",
        "30_35k",
        "35_40k",
        "40_45k",
        "45_50k",
        "50_60k",
        "60_75k",
        "75_100k",
        "100_125k",
        "125_150k",
        "150_200k",
        "GE200k",
    ]

    hincp = pd.cut(
        gph["HINCP"] / 1000,
        bins=(
            -59.999,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            60,
            75,
            100,
            125,
            150,
            200,
            np.inf,
        ),
        labels=hincp_lv,
        right=False,
    )

    hincp = pd.get_dummies(hincp, prefix="hhinc")

    return hincp


def hhsize_vehicles(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of household
    size by vehicles available that harmonizes ACS PUMS
    questionnaire items NP and VEH with ACS SF table B08201.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing NP and VEH columns.

    Returns
    -------
    hxv: pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS household size by
        vehicles available categories based on ACS SF table B08201.s
    """

    hhsize = pd.cut(
        gph["NP"],
        bins=(1, 2, 3, 4, np.inf),
        labels=["01", "02", "03", "GE04"],
        right=False,
    )
    hhsize = pd.get_dummies(hhsize, prefix="p", prefix_sep="")
    hhsize.columns = hhsize.columns.astype("str")

    veh_lv = [i + "_vehicle" for i in ["no", "01", "02", "03", "GE04"]]
    veh = pd.cut(gph["VEH"], bins=(0, 1, 2, 3, 4, np.inf), labels=veh_lv, right=False)
    veh = pd.get_dummies(veh)
    veh.columns = veh.columns.astype("str")

    hxv = intersect_dummies(veh, hhsize)

    return hxv


def hhtype(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of household type that
    harmonizes ACS PUMS questionnaire item HHT with ACS SF table B11001.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing a HHT column.

    Returns
    -------
    hht : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS household
        type categories based on ACS SF table B1101.
    """

    hht_lv = [
        "married",
        "male_no_spouse",
        "female_no_spouse",
        "alone",
        "not_alone",
        "alone",
        "not_alone",
    ]

    hht = pd.cut(
        gph["HHT"],
        bins=(1, 2, 3, 4, 5, 6, 7, np.inf),
        labels=hht_lv,
        right=False,
        ordered=False,
    )

    hht = pd.get_dummies(hht, prefix="hht")

    return hht


def hhtype_hhsize(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of household type by household size
    that harmonizes ACS PUMS questionnaire item NP with ACS SF table B11016.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing HHT and NP columns.

    Returns
    -------
    hht_hnp : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS household type by household size
        categories based on ACS SF table B11016.
    """

    # household type
    hht_lv = ["fam", "nonfam"]
    hht = pd.cut(
        gph["HHT"],
        bins=(1, 4, np.inf),
        labels=hht_lv,
        right=False,
        ordered=False,
    )
    hht = pd.get_dummies(hht, prefix="hht").astype("int")

    # household size
    hnp_lv = ["vac", "1p", "2p", "3p", "4p", "5p", "6p", "7pm"]
    hnp = pd.cut(
        gph["NP"],
        bins=tuple(np.append(np.arange(0, 8), np.inf)),
        labels=hnp_lv,
        right=False,
    )
    hnp = pd.get_dummies(hnp, prefix="hhsize").astype("int")

    # exclude vacant hhsize category
    hnp = hnp.iloc[:, 1:]

    # combine variables
    hht_hnp = intersect_dummies(hnp, hht)

    # drop this col, as it should always be empty
    hht_hnp = hht_hnp.drop("hht_fam_hhsize_1p", axis=1)

    return hht_hnp


def housing_units(gph: pd.DataFrame) -> pd.Series:
    """Generates a household-level flag for housing units that harmonizes
    ACS PUMS questionnaire item WGTP with ACS SF table B25001.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing a WGTP column.

    Returns
    -------
    hsu : pandas.Series
        One-hot encoded flag for housing units as defined by ACS SF table B25001.
    """

    wgtp_col = gph.columns[gph.columns.str.contains("^WGTP")].values[0]
    hsu = (gph[[wgtp_col]] >= 1).astype("int")
    hsu = hsu.rename({wgtp_col: "housing_units"}, axis=1)

    return hsu


def hsplat(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of Hispanic/Latino origin that
    harmonizes ACS PUMS questionnaire item HISP with ACS SF table B03003.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses inlcuding a HISP column.

    Returns
    -------
    hisp : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS race
        categories based on ACS SF table B03003.
    """

    hisp = pd.get_dummies(np.where(gpp["HISP"] > 1, "yes", "no"), prefix="hsplat")

    return hisp


def internet(gph: pd.DataFrame, year: int | str) -> pd.DataFrame:
    """Generates a household-level representation of household internet access
    that harmonizes ACS PUMS questionnaire item ACCESS/ACCESSINET
    with ACS SF table B28002.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing an ACCESS/ACCESINET column.
    year : int | str
        ACS 5-Year Estimates vintage. Not optional, here, but defaults
        as ``2019`` passed in from ``acs.build_acs_pums_inputs()``.

    Returns
    -------
    inet : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS household
        internet access categories based on ACS SF table B28002.
    """

    access = "ACCESS" if int(year) <= 2019 else "ACCESSINET"

    inet_desc = ["subscription", "no_subscription", "none"]
    inet = pd.cut(gph[access], bins=(1, 2, 3, np.inf), labels=inet_desc, right=False)

    inet = pd.get_dummies(inet, prefix="internet")

    return inet


def ipr(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of income to poverty ratio that
    harmonizes ACS PUMS questionnaire item POVPIP with ACS SF table C17002.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a POVPIP column.

    Returns
    -------
    ipr : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS income to poverty
        ratio categories based on ACS SF table C17002.
    """

    ipr_desc = ["L050", "050_099", "100_124", "125_149", "150_184", "185_199", "GE200"]

    povpip = pd.cut(
        gpp["POVPIP"],
        bins=(0, 50, 100, 125, 150, 185, 200, np.inf),
        labels=ipr_desc,
        right=False,
    )

    ipr = pd.get_dummies(povpip, prefix="ipr", dtype="int")

    return ipr


def language(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of language spoken at home
    that harmonizes ACS PUMS questionnaire item HHL with ACS SF table C16002.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing a HHL column.

    Returns
    -------
    hhl : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS language spoken
        at home categories based on ACS SF table C16002.
    """

    hhl_desc = ["english", "spanish", "oth_indo_euro", "asian_pac_isl", "other"]
    hhl = pd.cut(gph["HHL"], bins=(1, 2, 3, 4, 5, 6), labels=hhl_desc, right=False)

    hhl = pd.get_dummies(hhl, prefix="language")

    return hhl


def lep(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of limited English proficiency by
    race/ethnicity and nativity that harmonizes ACS PUMS questionnaire items RAC1P,
    HISP, POBP, and ENG with ACS SF tables B16005:A-I.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing RAC1P, HISP, POBP, and ENG columns.

    Returns
    -------
    lep : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS race/ethnicity by
        nativity categories based on ACS SF tables B16005:A-I.
    """

    ## race
    rac1p = race(gpp)

    ## Hispanic/Latino origin
    hisp = hsplat(gpp)

    ## nativity
    nat = pd.get_dummies(np.where(gpp["POBP"] < 100, "native", "foreign_born"))

    ## english ability
    eng = pd.cut(
        gpp["ENG"], bins=(1, 2, np.inf), labels=["vwell", "less_vwell"], right=False
    )
    eng = eng.cat.add_categories("only").fillna("only")
    eng = pd.get_dummies(eng, prefix="eng")

    ## combine nativity and english ability
    exn = intersect_dummies(eng, nat)

    ## limited english proficiency (base)
    lep0 = exn.loc[:, exn.columns.str.contains("_less_vwell$")]

    ## LEP by race
    rxl = intersect_dummies(lep0, rac1p)

    ## LEP by Hispanic/Latino origin
    hxl = intersect_dummies(lep0, hisp)
    hxl = hxl.loc[:, hxl.columns.str.contains("^hsplat_yes")]
    hxl.columns = [i.replace("hsplat_yes", "hsplat") for i in hxl.columns]

    ## final LEP table
    lep = pd.concat([rxl, hxl], axis=1)

    return lep


def lingisol(gpp: pd.DataFrame) -> pd.Series:
    """Generates a person-level flag for household linguistic isolation that
    harmonizes ACS PUMS questionnaire item LNGI with ACS SF variable B16003001.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a LNGI column.

    Returns
    -------
    lngi : pandas.Series
        One-hot encoded flag for ACS PUMS linguistic isolation
        categories based on ACS SF variable B16003001.
    """

    lngi0 = pd.Categorical(
        gpp["LNGI"].values.astype("str"), categories=["1", "2"], ordered=True
    )
    lngi = pd.get_dummies(lngi0)
    lngi.columns = ["lngi_no", "lngi_yes"]
    lngi = lngi["lngi_yes"]

    return lngi


def minors(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of presence of minors (under age
    18) that harmonizes ACS PUMS questionnaire item R18 with ACS SF table B11005.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing an R18 column.

    Returns
    -------
    r18 : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS presence of minors
        categories based on ACS SF table B11005.
    """

    r18 = pd.cut(gph["R18"], bins=(0, 1, 2), labels=["no", "yes"], right=False)
    r18 = pd.get_dummies(r18, prefix="minors")

    return r18


def occhu(gph: pd.DataFrame, year: int | str) -> pd.Series:
    """Generates a household-level flag for occupied housing units
    that harmonizes ACS PUMS questionnaire items TYPE (2016 - 2019)
    and TYPEHUGQ (2020+) with ACS SF table B25003.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing a TYPE or TYPEHUGQ column.
    year : int
        ACS 5-Year Estimates vintage. Not optional, here, but defaults
        as ``2019`` passed in from ``acs.build_acs_pums_inputs()``.

    Returns
    -------
    ohu : pandas.Series
        One-hot encoded flag for occupied housing units as
        defined by ACS SF table B25003.
    """

    hh_type_var = "TYPE" if int(year) < 2020 else "TYPEHUGQ"

    ohu = ((gph["VACS"] == 0) & (gph.loc[:, hh_type_var] == 1)).astype("int")

    return ohu.rename("occhu")


def owncost(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of monthly homeowner
    housing costs as a percentage of household income in the last 12 months
    by mortgage status that harmonizes ACS PUMS questionnaire item
    OCPIP and TEN with ACS SF table B25087.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing a OCPIP and TEN columns.

    Returns
    -------
    oc : pandas.DataFrame
        One-hot encoded flag for monthly homeowner housing
        costs as defined by ACS SF table B25087.
    """

    mtg = 1 * pd.DataFrame({"mortgage": gph["TEN"] == 1, "nomortgage": gph["TEN"] == 2})

    income_percent = [
        "less10",
        "10to14.9",
        "15to19.9",
        "20to24.9",
        "25to29.9",
        "30to34.9",
        "35to39.9",
        "40to49.9",
        "more50",
    ]

    ocpip = pd.cut(
        gph["OCPIP"],
        bins=(0, 10, 15, 20, 25, 30, 35, 40, 50, np.inf),
        labels=income_percent,
        right=False,
    )
    ocpip = pd.get_dummies(ocpip, dtype="int")
    ocpip.columns = ocpip.columns.astype("str")

    oc = intersect_dummies(ocpip, mtg)
    oc.columns = [f"owncost_{col}" for col in oc.columns]

    return oc


def population(gpp: pd.DataFrame) -> pd.Series:
    """Generates a population-level representation of total population
    that harmonizes the PUMS with ACS SF variable B01001001.

    Within each PUMS household, each person-level record is
    weighted by the corresponding person sample weight (``PWGTP``)
    divided by the reference person's sample weight (``WGTP``).
    This is used to approximate the representativeness of
    each individual in the PUMA population. Individuals in group
    quarters also receive a value of 1, as they are considered
    the representative person for a group quarters residence.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing PWGTP, SERIALNO, and SPORDER columns.

    Returns
    -------
    popul : pandas.Series
        A person-level representation of ACS variable B01001001.
    """

    pwgtp_col = gpp.columns[gpp.columns.str.contains("^PWGTP")].values[0]

    # # Rarely, we will encounter negative replicate weights
    # # This causes problems when creating the P-MEDM allocation matrix
    # # Our current workaround is to convert negative weights to zero
    # gpp.loc[gpp[pwgtp_col] < 0, pwgtp_col] = 0
    popul_ = gpp.groupby("SERIALNO")[[pwgtp_col, "SPORDER"]].apply(
        lambda x: pd.DataFrame(
            {
                "SPORDER": x["SPORDER"].values,
                "population": x.loc[:, pwgtp_col]
                / np.where(
                    x.loc[:, pwgtp_col].iloc[0] <= 0, 1, x.loc[:, pwgtp_col].iloc[0]
                ),
            }
        )
    )
    popul_ = popul_.reset_index()
    popul_["p_id"] = popul_["SERIALNO"] + popul_["SPORDER"].astype("str").str.zfill(2)

    gpp_ = gpp.copy()
    gpp_["p_id"] = gpp_["SERIALNO"] + gpp_["SPORDER"].astype("str").str.zfill(2)
    gpp_ = pd.concat(
        [gpp_.set_index("p_id"), popul_.set_index("p_id").population], axis=1
    )

    popul = gpp_.population.reset_index(drop=True)

    return popul


def poverty(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of poverty status that
    harmonizes ACS PUMS questionnaire item POVPIP with ACS SF table B17021.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a POVPIP column.

    Returns
    -------
    pov : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS poverty status
        categories based on ACS SF table B17021.
    """

    pov = pd.get_dummies(np.where(gpp["POVPIP"] < 100, "yes", "no"), prefix="poverty")
    pov.loc[(gpp["POVPIP"] == -1), "poverty_yes"] = False  # do not include NAs

    return pov


def race(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of race that harmonizes
    ACS PUMS questionnaire item RAC1P with and ACS SF table B02001.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses including a RAC1P column.

    Returns
    -------
    rac1p : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS race
        categories based on ACS SF table B02001.
    """

    race_lv = [
        "white",
        "blk_af_amer",
        "native_amer",
        "asian",
        "pac_island",
        "other",
        "mult",
    ]

    rac1p = pd.cut(
        gpp["RAC1P"], bins=(1, 2, 3, 6, 7, 8, 9, np.inf), labels=race_lv, right=False
    )

    rac1p = pd.get_dummies(rac1p, prefix="race")

    return rac1p


def rentcost(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of monthly gross rent as a
    percentage of household income in the last 12 months by tenure that harmonizes
    ACS PUMS questionnaire item GRPIP with ACS SF table B25070.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing a GRPIP column.

    Returns
    -------
    rent_pct : pandas.DataFrame
        One-hot encoded flag for monthly gross rent
        cost as defined by ACS SF table B25070.
    """

    income_percent = [
        "less10",
        "10to14.9",
        "15to19.9",
        "20to24.9",
        "25to29.9",
        "30to34.9",
        "35to39.9",
        "40to49.9",
        "more50",
    ]

    rent_pct = pd.cut(
        gph["GRPIP"],
        bins=(1, 10, 15, 20, 25, 30, 35, 40, 50, np.inf),
        labels=income_percent,
        right=False,
    )
    rent_pct = pd.get_dummies(rent_pct, dtype="int")
    rent_pct.columns = rent_pct.columns.astype("str")
    rent_pct.columns = [f"rentcost_{col}" for col in rent_pct.columns]

    return rent_pct


def rooms(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of total rooms in dwelling
    that harmonizes ACS PUMS questionnaire item RMSP with ACS SF table B25017.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing a RMSP column.

    Returns
    -------
    rmsp : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS bedrooms in dwelling
        categories based on ACS SF table B25017.
    """

    rmsp_desc = ["01", "02", "03", "04", "05", "06", "07", "08", "GE09"]
    rmsp = pd.cut(
        gph["RMSP"],
        bins=(1, 2, 3, 4, 5, 6, 7, 8, 9, np.inf),
        labels=rmsp_desc,
        right=False,
    )
    rmsp = pd.get_dummies(rmsp, prefix="rooms")

    return rmsp


def school(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of sex by grade level
    by school type that harmonizes ACS PUMS questionnaire items SEX,
    SCHG, and SCH with ACS SF table B14002.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing SEX, SCHG, and SCH columns.

    Returns
    -------
    sch_type : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS sex by grade level by
        school type categories based on ACS SF table B14002.
    """

    # school type
    sch = pd.cut(
        gpp["SCH"], bins=(2, 3, np.inf), labels=["public", "private"], right=False
    )
    sch = pd.get_dummies(sch)
    sch.columns = sch.columns.astype("str")

    # grade level
    glev = pd.cut(
        gpp["SCHG"],
        bins=(1, 2, 3, 7, 11, 15, 16, np.inf),
        labels=["pre", "kind", "01.04", "05.08", "09.12", "undergrad", "grad.prof"],
        right=False,
    )

    glev = pd.get_dummies(glev)
    glev.columns = glev.columns.astype("str")

    # sex
    sexp = sex(gpp)

    # combine variables
    sch_type0 = intersect_dummies(sch, glev)
    sch_type = intersect_dummies(sch_type0, sexp)

    # append prefix
    sch_type.columns = ["sch_" + i for i in sch_type.columns]

    return sch_type


def seniors(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of presence
    of seniors (age 60 and over) that harmonizes ACS PUMS
    questionnaire item R60 with ACS SF table B11006.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing an R60 column.

    Returns
    -------
    r60 : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS presence of seniors
        categories based on ACS SF table B11006.
    """

    r60 = pd.cut(gph["R60"], bins=(0, 1, 3), labels=["no", "yes"], right=False)
    r60 = pd.get_dummies(r60, prefix="seniors")

    return r60


def sex(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of sex that harmonizes
    ACS PUMS questionnaire item SEX with ACS SF table B01001.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a SEX column.

    Returns
    -------
    sex : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS sex
        categories based on ACS SF table B01001.
    """

    sex = pd.get_dummies(np.where(gpp["SEX"] == 1, "male", "female"))

    return sex


def sex_age(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of sex by age that harmonizes
    ACS PUMS questionnaire itemd SEX and AGEP with ACS SF table B01001.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing SEX and AGEP columns.

    Returns
    -------
    sxa : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS sex by age
        categories based on ACS SF table B01001.
    """

    sexp = sex(gpp)

    age_lv = [
        "a05u",
        "a05_09",
        "a10_14",
        "a15_17",
        "a18_19",
        "a20",
        "a21",
        "a22_24",
        "a25_29",
        "a30_34",
        "a35_39",
        "a40_44",
        "a45_49",
        "a50_54",
        "a55_59",
        "a60_61",
        "a62_64",
        "a65_66",
        "a67_69",
        "a70_74",
        "a75_79",
        "a80_84",
        "a85o",
    ]

    age = pd.cut(
        gpp["AGEP"],
        bins=(
            0,
            5,
            10,
            15,
            18,
            20,
            21,
            22,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            62,
            65,
            67,
            70,
            75,
            80,
            85,
            np.inf,
        ),
        labels=age_lv,
        right=False,
    )

    age = pd.get_dummies(age)
    age.columns = age.columns.astype("str")

    sxa = intersect_dummies(age, sexp)

    return sxa


def sexcw(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of sex by class of worker that
    harmonizes ACS PUMS questionnaire  items SEX and COW with ACS SF table B24080.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing SEX, COW, and ESR columns.

    Returns
    -------
    sxc : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS sex by class
        of worker categories based on ACS SF table B24080.
    """

    sexp = sex(gpp)

    cow0 = pd.cut(
        gpp["COW"],
        bins=range(1, 10),
        labels=[
            "private",
            "non.prof",
            "local.gov",
            "st.gov",
            "fed.gov",
            "self.emp_non.inc",
            "self.emp_inc",
            "wo.pay",
        ],
        right=False,
    )

    cow0 = pd.get_dummies(cow0)
    cow0.columns = cow0.columns.astype("str")

    # subset to civilian employed pop age 16+
    emp = gpp["ESR"].isin([1, 2])
    cow = cow0.values * emp.values[:, None]
    cow = pd.DataFrame(cow, columns=cow0.columns)

    # combine with sex
    sxc = intersect_dummies(cow, sexp)
    sxc.columns = ["sexcw_" + i for i in sxc.columns]

    return sxc


def sexnaics(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of sex by
    NAICS industry that harmonizes ACS PUMS questionnaire
    items SEX and NAICSP with ACS SF table C24030.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing SEX, NAICSP, and ESR columns.

    Returns
    -------
    sex_naicsp : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS sex by NAICS
        industry categories based on ACS SF table C24030.
    """

    sexp = sex(gpp)

    # NAICSP to ACS categories
    naicsp2acs = dict(  # noqa C418
        {
            "adm": "^921|^923|^928P|^92M",
            "agr_ext": "^11|^21",
            "con": "^23",
            "edu_med_sca": "^61|^62",
            "ent": "^71|^72",
            "fin": "^52|^53",
            "inf": "^51",
            "mfg": "^31|^32|^33|^3MS",
            "mil": "^9281",
            "prf": "^54|^55|^56",
            "ret": "^44|^45|^4MS",
            "srv": "^81",
            "utl_trn": "^22|^48|^49",
            "whl": "^42",
        }
    )

    # format NAICSP and subset to civilian employed age 16+
    gpp_naicsp = gpp["NAICSP"].copy().astype("str")
    gpp["naicsp_"] = np.where(gpp["ESR"].isin([1, 2]), gpp_naicsp, "-999")

    naicsp = reclass_dummies(gpp, "naicsp_", naicsp2acs)
    sex_naicsp = intersect_dummies(naicsp, sexp)
    sex_naicsp.columns = ["sexnaics_" + i for i in sex_naicsp.columns]

    return sex_naicsp


def sexnaics_det(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a detailed person-level representation of sex by
    NAICS industry that harmonizes ACS PUMS questionnaire
    items SEX and NAICSP with ACS SF table C24030.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing SEX, NAICSP, ESR columns.

    Returns
    -------
    sex_naicsp_det : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS sex by  NAICS industry
        categories based on ACS SF table C24030.
    """

    sexp = sex(gpp)

    # NAICSP to ACS categories
    naicsp2acs_det = dict(  # noqa C418
        {
            "agr_ffh": "^11",
            "ext": "^21",
            "con": "^23",
            "mfg": "^31|^32|^33|^3MS",
            "whl": "^42",
            "ret": "^44|^45|^4MS",
            "trn_whs": "^48|^49",
            "utl": "^22",
            "inf": "^51",
            "fin_ins": "^52",
            "rrl": "^53",
            "prf": "^54",
            "mgt": "^55",
            "adm_wmr": "^56",
            "edu": "^61",
            "med_sca": "^62",
            "ent": "^71",
            "afs": "^72",
            "srv": "^81",
            "pad": "^921|^923|^928P|^92M",
        }
    )

    # format NAICSP and subset to civilian employed age 16+
    gpp_naicsp = gpp["NAICSP"].copy().astype("str")
    gpp["naicsp_"] = np.where(gpp["ESR"].isin([1, 2]), gpp_naicsp, "-999")

    naicsp_det = reclass_dummies(gpp, "naicsp_", naicsp2acs_det)
    sex_naicsp_det = intersect_dummies(naicsp_det, sexp)
    sex_naicsp_det.columns = ["sexnaics_det_" + i for i in sex_naicsp_det.columns]

    return sex_naicsp_det


def sexocc(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of sex by
    occupation that harmonizes ACS PUMS questionnaire items
    SEX and OCCP with ACS SF table C24010.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing SEX, OCCP, and ESR columns.

    Returns
    -------
    sex_occp : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS sex by occupation
        categories based on ACS SF table C24010.
    """

    sexp = sex(gpp)

    # OCCP to ACS categories
    occp2acs = dict(  # noqa C418
        {
            "mgmt": "^00|^01|^02|^03|^04",
            "bus.fin": "^05|^06|^07|^08|^09",
            "cmp": "^10|^11|^12",
            "eng": "^13|^14|^15",
            "sci": "^16|^17|^18|^19",
            "cms": "^20",
            "lgl": "^21",
            "edu": "^22|^23|^24|^25",
            "ent": "^26|^27|^28|^29",
            "med": "^30|^31|^32",
            "med.tech": "^33|^34|^35",
            "hls": "^36",
            "prt": "^3720|^3725|^3730|^3740|^3750|^39",
            "law.enf": "^3700|^3710|^38",
            "eat": "^40|^41",
            "cln": "^42",
            "prs": "^43|^44|^45|^46",
            "sal": "^47|^48|^49",
            "off": "^50|^51|^52|^53|^54|^55|^56|^57|^58|^59",
            "fff": "^60|^61",
            "con.ext": "^62|^63|^64|^65|^66|^67|^68|^69",
            "rpr": "^70|^71|^72|^73|^74|^75|^76",
            "prd": "^77|^78|^79|^80|^81|^82|^83|^84|^85|^86|^87|^88|^89",
            "trn": "^90|^91|^92|^93|^94",
            "trn.mat": "^95|^96|^97",
        }
    )

    # format OCCP and subset to civilian employed age 16+
    gpp_occp = gpp["OCCP"].copy().astype("str").str.zfill(4).replace("0009", "")
    gpp["occp_"] = np.where(gpp["ESR"].isin([1, 2]), gpp_occp, "-999")

    occp = reclass_dummies(gpp, "occp_", occp2acs)
    sex_occp = intersect_dummies(occp, sexp)
    sex_occp.columns = ["sexocc_" + i for i in sex_occp.columns]

    return sex_occp


def tenure(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of tenure that
    harmonizes ACS PUMS questionnaire item TEN with ACS SF table B25003.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing a TEN column.

    Returns
    -------
    ten : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS tenure
        categories based on ACS SF table B25003.
    """

    ten_lv = ["own", "rent"]

    ten = pd.cut(gph["TEN"], bins=(1, 3, 4), labels=ten_lv, right=False)

    ten = pd.get_dummies(ten, prefix="tenure")

    return ten


def tenure_vehicles(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of tenure
    by vehicles available that harmonizes ACS PUMS questionnaire
    items TEN and VEH with ACS SF table B08201.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing TEN and VEH columns.

    Returns
    -------
    txv : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS tenure by vehicles
        available categories based on ACS SF table B08201.
    """

    ten = tenure(gph)

    veh_lv = [i + "_vehicle" for i in ["no", "01", "02", "03", "04", "GE05"]]
    veh = pd.cut(
        gph["VEH"], bins=(0, 1, 2, 3, 4, 5, np.inf), labels=veh_lv, right=False
    )
    veh = pd.get_dummies(veh)
    veh.columns = veh.columns.astype("str")

    txv = intersect_dummies(veh, ten)

    txv.columns = [re.sub("tenure", "txv", v) for v in txv.columns]

    return txv


def travel(gpp: pd.DataFrame, year: int | str) -> pd.DataFrame:
    """Generates a person-level representation of means of travel to work that
    harmonizes ACS PUMS questionnaire item JWTR/JWTRNS with ACS SF table B08301.
        * JWTR < 2019
        * JWTRNS >= 2019

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a JWTR or JWTRNS column.
    year : int | str
        ACS 5-Year Estimates vintage. Not optional, here, but defaults
        as ``2019`` passed in from ``acs.build_acs_pums_inputs()``.

    Returns
    -------
    job_work_travel : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS means of travel
        to work categories based on ACS SF table B08301.
    """

    travel_var = "JWTR" if int(year) < 2019 else "JWTRNS"

    lv = [
        "car_truck_van",
        "public_transportation",
        "taxicab",
        "motorcycle",
        "bicycle",
        "walked",
        "wfh",
        "other",
    ]

    job_work_travel = pd.cut(
        gpp[travel_var],
        bins=(1, 2, 7, 8, 9, 10, 11, 12, np.inf),
        labels=lv,
        right=False,
    )

    job_work_travel = pd.get_dummies(job_work_travel, prefix="travel")

    return job_work_travel


def units(gph: pd.DataFrame) -> pd.DataFrame:
    """Generates a household-level representation of units in structure that
    harmonizes ACS PUMS questionnaire item BLD with ACS SF table B25024.

    Parameters
    ----------
    gph : pandas.DataFrame
        Household-level PUMS responses containing a BLD column.

    Returns
    -------
    bld : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS units in
        structure categories based on ACS SF table B25024.
    """

    bld_lv = [
        "mob_home",
        "single_fam_detach",
        "single_fam_attach",
        "2_unit",
        "3_4_unit",
        "5_9_unit",
        "10_19_unit",
        "20_49_unit",
        "GE50_unit",
        "other",
    ]

    bld = pd.cut(
        gph["BLD"],
        bins=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        labels=bld_lv,
        right=False,
    )

    bld = pd.get_dummies(bld, prefix="dwg")

    return bld


def veh_occ(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of vehicle occupancy
    for work commutes by car/truck/van that harmonizes ACS PUMS
    questionnaire item JWRIP with ACS SF table B08301.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing a JWRIP column.

    Returns
    -------
    jwrip : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS vehicle occupancy
        categories based on ACS SF table B08301.
    """

    jwrip = pd.cut(
        gpp["JWRIP"],
        bins=(1, 2, np.inf),
        labels=["drove_alone", "carpooled"],
        right=False,
    )

    jwrip = pd.get_dummies(jwrip, prefix="veh_occ")

    return jwrip


def veteran(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of veteran status for the
    civilian population age 18 years and over that harmonizes ACS PUMS
    questionnaire items VPS, MIL, and AGEP with ACS SF table B21001.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing MIL, VPS, and AGEP columns.

    Returns
    -------
    vet : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS veteran status based on
        ACS SF table B21001.
    """

    vet = pd.get_dummies(
        np.where((gpp["VPS"] > 0) & (gpp["MIL"] != 1), "vet", "nonvet")
    )
    vet = vet * ((gpp["AGEP"] >= 18).values[:, None] * 1)
    vet.columns = vet.columns.astype("str")

    return vet


def vet_edu(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of veteran status
    by educational attainment that harmonizes ACS PUMS questionnaire
    items SCHL and VPS with ACS SF table B21003.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing MIL, VPS, AGEP, and SCHL columns.

    Returns
    -------
    vxe : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS veteran status by
        educational attainment categories based on ACS SF table B21003.
    """

    aGE25 = np.where(gpp["AGEP"] >= 25, 1, 0)  # noqa N806

    vet_ = veteran(gpp)

    edu_desc = ["less_hs", "hs", "some_clg", "bach_higher"]
    edu = pd.cut(
        gpp["SCHL"], bins=(1, 16, 18, 21, np.inf), labels=edu_desc, right=False
    )
    edu = pd.get_dummies(edu)
    edu.columns = edu.columns.astype("str")

    vxe = intersect_dummies(edu, vet_)
    vxe = vxe.multiply(aGE25, axis=0)

    return vxe


def worked(gpp: pd.DataFrame) -> pd.DataFrame:
    """Generates a person-level representation of sex by hours
    worked per week that harmonizes ACS PUMS questionnaire items
    SEX and WKHP with ACS SF table B23022.

    Parameters
    ----------
    gpp : pandas.DataFrame
        Person-level PUMS responses containing SEX and WKHP columns.

    Returns
    -------
    sex_wkhp : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS sex by hours worked
        per week categories based on ACS SF table B23022.
    """

    sexp = sex(gpp)

    wkhp = pd.cut(
        gpp["WKHP"],
        bins=(1, 15, 35, np.inf),
        labels=["1.14", "15.34", "GE35"],
        right=False,
    )

    wkhp = pd.get_dummies(wkhp, prefix="hours")

    sex_wkhp = intersect_dummies(wkhp, sexp)

    return sex_wkhp


def year_built(gph: pd.DataFrame, year: int | str) -> pd.DataFrame:
    """Generates a household-level representation of year of dwelling construction
    that harmonizes ACS PUMS  questionnaire item YBL with ACS SF table B25034.

    Parameters
    ----------
    gph : pandas.DataFrame
       Household-level PUMS responses containing a YBL column.
    year : int | str
        ACS 5-Year Estimates vintage. Not optional, here, but defaults
        as ``2019`` passed in from ``acs.build_acs_pums_inputs()``.

    Returns
    -------
    ybl : pandas.DataFrame
        One-hot encoded DataFrame of ACS PUMS year of dwelling
        construction categories based on ACS SF table B25034.
    """

    if int(year) < 2022:
        ybl_lv = [
            "L1939",
            "40_49",
            "50_59",
            "60_69",
            "70_79",
            "80_89",
            "90_99",
            "00_09",
            "10_13",
            "GE2014",
        ]

        ybl_ = pd.cut(
            gph["YBL"],
            bins=(1, 2, 3, 4, 5, 6, 7, 8, 14, 18, np.inf),
            labels=ybl_lv,
            right=False,
        )

    else:
        ybl_lv = [
            "L1939",
            "40_49",
            "50_59",
            "60_69",
            "70_79",
            "80_89",
            "90_99",
            "00_09",
            "10_19",
            "GE2020",
        ]

        ybl_ = pd.cut(
            gph["YRBLT"],
            bins=(1939, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, np.inf),
            labels=ybl_lv,
            right=False,
        )

    ybl = pd.get_dummies(ybl_, prefix="year_built")

    return ybl
