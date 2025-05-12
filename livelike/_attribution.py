import re

import numpy as np
import pandas as pd


def columns_to_labels(
    df: pd.DataFrame,
    filter_regex: str,
    scrub: None | str = None,
    keep_index: bool = True,
) -> pd.DataFrame:
    """
    Converts dummy columns to labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing dummy columns.
    filter_regex : str
        A regular expression for filter_regexing the columns to
        use for labeling.
    scrub : str
        A regular expression for label text to exclude.
    keep_index : bool
        Whether to keep the input data frame index for
        the output labels

    Returns
    -------
    labels : pandas.Series
        Converted labels.
    """
    df = df.loc[:, df.columns.str.contains(filter_regex)]

    df_ = pd.DataFrame()
    for col in df.columns:
        df_[col] = np.where(df[col] > 0, col, "")

    labels = df_.sum(axis=1)
    if scrub is not None:
        labels = labels.apply(lambda x: re.sub(scrub, "", x))

    if keep_index:
        labels.index = df.index

    return labels


## Labeling Functions ##

def age(est_person: pd.DataFrame):
    """
    Produces person-level labels for age cohort.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Sex by age labels.
    """
    filter_regex = "^female_a|^male_a"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "age"

    # consolidate labels as needed 
    # for consistent 5-year intervals
    labels = labels.replace(
        {
            "15_17" : "15_19",
            "18_19" : "15_19",
            "20" : "20_24",
            "21" : "20_24",
            "22_24" : "20_24",
            "60_61" : "60_64",
            "62_64" : "60_64",
            "65_66" : "65_69",
            "67_69" : "65_69"
        }
    )

    return labels        


def class_of_worker(est_person: pd.DataFrame):
    """
    Produces person-level labels for class of worker.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Class of worker labels.
    """
    filter_regex = "^sexcw_male_|^sexcw_female_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "class_of_worker"

    return labels


def commute_drive_type(est_person: pd.DataFrame) -> pd.Series:
    """
    Produces person-level labels for commute drive type.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Commute drive type labels.
    """
    filter_regex = "^veh_occ_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "commute_drive_type"

    return labels


def commute_mode(est_person: pd.DataFrame):
    """
    Produces person-level labels for commute mode.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Commute mode labels.
    """
    filter_regex = "^travel_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "commute_mode"

    return labels


def commute_time_m(est_person: pd.DataFrame):
    """
    Produces person-level labels for commute time in minutes.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Commute time labels.
    """
    filter_regex = "^cmt_mins_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "commute_time_m"

    return labels


def dwelling_type(est_household: pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for dwelling type.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Dwelling type labels.
    """
    filter_regex = "^dwg_"
    labels = columns_to_labels(est_household, filter_regex=filter_regex, scrub=filter_regex)
    labels.name = "dwelling_type"

    return labels


def edu_attainment(est_person: pd.DataFrame):
    """
    Produces person-level labels for educational
    attainment for adults 25+.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Educational attainment labels.
    """
    filter_regex = "^schl_"
    scrub = "schl_|female_|male_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=scrub
    )
    labels.name = "edu_attainment"

    return labels        


def employment_status(est_person: pd.DataFrame) -> pd.Series:
    """
    Produces person-level labels for employment status.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Employment status labels.
    """
    filter_regex = "emp_stat_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "employment_status"

    return labels


def grade(est_person: pd.DataFrame) -> pd.Series:
    """
    Produces person-level labels for school grade level.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Grade level labels.
    """
    filter_regex = "^grade_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "grade"

    return labels


def hh_income(est_household: pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for household income.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Household income labels.
    """
    filter_regex = "^hhinc_"
    labels = columns_to_labels(est_household, filter_regex=filter_regex, scrub=filter_regex)
    labels.name = "hh_income"

    return labels


def hours_worked(est_person: pd.DataFrame) -> pd.Series:
    """
    Produces person-level labels for hours worked.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Hours worked labels.
    """
    filter_regex = "^male_hours_|^female_hours_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "hours_worked"

    return labels


def house_heating_fuel(est_household: pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for house heating fuel.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
       House heating fuel labels.
    """
    filter_regex = "^hhf_"
    labels = columns_to_labels(est_household, filter_regex=filter_regex, scrub=filter_regex)
    labels.name = "house_heating_fuel"

    return labels


def household_size(est_household: pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for household size.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Household size labels.
    """
    filter_regex = "^hht_fam_|^hht_nonfam_"
    scrub = filter_regex + "|hhsize_"
    labels = columns_to_labels(est_household, filter_regex=filter_regex, scrub=scrub)
    labels.name = "household_size"

    return labels


def household_type(est_household: pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for household type.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Household type labels.
    """
    filter_regex = "^hht_"
    scrub = "hht_|_hhsize_|1p|2p|3p|4p|5p|6p|7pm|"
    labels = columns_to_labels(est_household, filter_regex=filter_regex, scrub=scrub)
    labels.name = "household_type"

    return labels


def housing_costs_pct_income(est_household: pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for monthly housing
    costs as a percentage of income.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Housing costs labels.
    """
    filter_regex = "^owncost_|^rentcost_"
    scrub = filter_regex + "|mortgage_|nomortgage_"
    labels = columns_to_labels(est_household, filter_regex=filter_regex, scrub=scrub)
    labels.name = "housing_costs_pct_income"

    return labels


def hispanic_latino(est_person: pd.DataFrame):
    """
    Produces person-level labels for Hispanic/Latino ethnicity.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Hispanic/Latino ethnicity labels.
    """
    filter_regex = "^hsplat_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "hispanic_latino"

    return labels        


def income_to_poverty_ratio(est_person: pd.DataFrame):
    """
    Produces person-level labels for income to poverty ratio.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Income to poverty ratio labels.
    """
    filter_regex = "^ipr_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "income_to_poverty_ratio"

    return labels        


def industry(est_person: pd.DataFrame) -> pd.Series:
    """
    Produces person-level labels for NAICS industry.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        NAICS industry time labels.
    """
    filter_regex = "^sexnaics_female_|^sexnaics_male_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "industry"

    return labels


def living_arrangement(est_household: pd.DataFrame):
    """
    Produces household-level labels for living arrangement.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Sex labels.
    """
    filter_regex = "^hht_"
    # manually exclude ``hhsize`` cols
    est_household = est_household.loc[:,~est_household.columns.str.contains("hhsize")]
    labels = columns_to_labels(
        est_household, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "living_arrangement"

    return labels        


def mortgage(est_household: pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for home mortgage
    for owner-occupied households.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Home mortgage labels.
    """
    filter_regex = "^owncost_"
    scrub = (
        "^owncost_|_less10|_10to14.9|_15to19.9|"
        "_20to24.9|_25to29.9|_30to34.9|_35to39.9|"
        "_40to49.9|_more50"
    )

    labels = columns_to_labels(est_household, filter_regex=filter_regex, scrub=scrub)
    labels.name = "mortgage"

    return labels


def occupation(est_person: pd.DataFrame) -> pd.Series:
    """
    Produces person-level labels for occupation
    based on Standard Occupation Classification.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Occupation labels.
    """
    filter_regex = "^sexocc_female_|^sexocc_male_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "occupation"

    return labels


def race(est_person: pd.DataFrame):
    """
    Produces person-level labels for race.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Race labels.
    """
    filter_regex = "^race_"
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=filter_regex
    )
    labels.name = "race"

    return labels        


def residence_type__person(est_person):
    """
    Produces person-level labels for residence type.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Residence type labels.
    """
    labels = pd.Series(
        np.where(est_person.group_quarters_pop == 1, "gq", "hu"),
        index=est_person.index,
        name="residence_type"
    )

    return labels    


def residence_type__household(est_household):
    """
    Produces household-level labels for residence type.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Residence type labels.
    """
    labels = pd.Series(
        np.where(
            est_household.housing_units == 0,
            "gq",
            np.where(
                est_household.occhu > 0,
                "hu_occ",
                "hu_vac"
            )
        ),
        index=est_household.index,
        name="residence_type"
    )

    return labels    



def school_type(est_person: pd.DataFrame) -> pd.Series:
    """
    Produces person-level labels for school type in minutes.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
       School type labels.
    """
    filter_regex = "^sch_female_|^sch_male_"
    scrub = "sch_|female_|male_|pre_|kind_|01.04_|05.08_|09.12_|undergrad_|grad.prof_"
    labels = columns_to_labels(est_person, filter_regex=filter_regex, scrub=scrub)
    labels.name = "school_type"

    return labels


def sex(est_person: pd.DataFrame):
    """
    Produces person-level labels for sex.

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Sex labels.
    """
    filter_regex = "^female_a|^male_a"
    scrub = (
        "_a|05u|05_09|10_14|15_17|18_19|"
        "20|21|22_24|25_29|30_34|35_39|"
        "40_44|45_49|50_54|55_59|60_61|"
        "62_64|65_66|67_69|70_74|75_79|"
        "80_84|85o"
    )
    labels = columns_to_labels(
        est_person, filter_regex=filter_regex, scrub=scrub
    )
    labels.name = "age_cohort"

    return labels        


def tenure(est_household: pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for tenure.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Tenure labels.
    """
    filter_regex = "^txv_"
    scrub = "txv_|_no_|_01_|_02_|_03_|_04_|_GE05_|vehicle"
    labels = columns_to_labels(est_household, filter_regex=filter_regex, scrub=scrub)
    labels.name = "tenure"

    return labels


def vehicles_available(est_household: pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for number of vehicles available.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Vehicles available labels.
    """
    filter_regex = "^txv_own_|^txv_rent_"
    scrub = filter_regex + "|_vehicle"
    labels = columns_to_labels(est_household, filter_regex=filter_regex, scrub=scrub)
    labels = labels.replace({"no": "none"})
    labels.name = "vehicles_available"

    return labels


def year_dwelling_built(est_household: pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for year dwelling built.

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        year dwelling built labels.
    """
    filter_regex = "^year_built_"
    labels = columns_to_labels(est_household, filter_regex=filter_regex, scrub=filter_regex)
    labels.name = "year_dwelling_built"

    return labels