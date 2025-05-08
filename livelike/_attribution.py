import numpy as np
import pandas as pd
import re

def columns_to_labels(
    df: pd.DataFrame, 
    filter: str, 
    scrub: None | str = None, 
    keep_index: bool = True,
) -> pd.DataFrame:
    """
    Converts dummy columns to labels. 

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing dummy columns.
    filter : str
        A regular expression for filtering the columns to 
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
    df = df.loc[:,df.columns.str.contains(filter)]

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
    filter = "^sexcw_male_|^sexcw_female_"
    labels = columns_to_labels(est_person, filter=filter, scrub=filter)
    labels.name = "class_of_worker"

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
    filter = "^cmt_mins_"
    labels = columns_to_labels(est_person, filter=filter, scrub=filter)
    labels.name = "commute_time_m"

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
    filter = "^travel_"
    labels = columns_to_labels(est_person, filter=filter, scrub=filter)
    labels.name = "commute_mode"

    return labels


def commute_drive_type(est_person : pd.DataFrame) -> pd.Series:
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
    filter = "^veh_occ_"
    labels = columns_to_labels(est_person, filter=filter, scrub=filter)
    labels.name = "commute_drive_type"

    return labels


def employment_status(est_person : pd.DataFrame) -> pd.Series:
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
    filter = "emp_stat_"
    labels = columns_to_labels(est_person, filter=filter, scrub=filter)
    labels.name = "employment_status"

    return labels


def grade(est_person : pd.DataFrame) -> pd.Series:
    """
    Produces person-level labels for school grade lavel. 

    Parameters
    ----------
    est_person: pandas.DataFrame
        Person-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Grade level labels.
    """
    filter = "^grade_"
    labels = columns_to_labels(est_person, filter=filter, scrub=filter)
    labels.name = "grade"

    return labels   


def hours_worked(est_person : pd.DataFrame) -> pd.Series:
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
    filter = "^male_hours_|^female_hours_"
    labels = columns_to_labels(est_person, filter=filter, scrub=filter)
    labels.name = "hours_worked"

    return labels


def household_size(est_household : pd.DataFrame) -> pd.Series:
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
    filter = "^hht_fam_|^hht_nonfam_"
    scrub = filter + "|hhsize_"
    labels = columns_to_labels(est_household, filter=filter, scrub=scrub)
    labels.name = "household_size"

    return labels


def household_type(est_household : pd.DataFrame) -> pd.Series:
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
    filter = "^hht_"
    scrub = "hht_|_hhsize_|1p|2p|3p|4p|5p|6p|7pm|"
    labels = columns_to_labels(est_household, filter=filter, scrub=scrub)
    labels.name = "household_type"

    return labels


def industry(est_person : pd.DataFrame) -> pd.Series:
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
    filter = "^sexnaics_female_|^sexnaics_male_"
    labels = columns_to_labels(est_person, filter=filter, scrub=filter)
    labels.name = "industry"

    return labels


def occupation(est_person : pd.DataFrame) -> pd.Series:
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
    filter = "^sexocc_female_|^sexocc_male_"
    labels = columns_to_labels(est_person, filter=filter, scrub=filter)
    labels.name = "occupation"

    return labels


def school_type(est_person : pd.DataFrame) -> pd.Series:
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
    filter = "^sch_female_|^sch_male_"
    scrub = (
        "sch_|female_|male_|pre_|kind_|01.04_|05.08_|09.12_|"
        "undergrad_|grad.prof_"
    )
    labels = columns_to_labels(est_person, filter=filter, scrub=scrub)
    labels.name = "school_type"

    return labels


def tenure(est_household : pd.DataFrame) -> pd.Series:
    """
    Produces household-level labels for tenure. 

    Parameters
    ----------
    est_household: pandas.DataFrame
        Household-level PUMS constraints.

    Returns
    -------
    labels : pandas.Series
        Household size labels.
    """
    filter="^txv_"
    scrub = "txv_|_no_|_01_|_02_|_03_|_04_|_GE05_|vehicle"
    labels = columns_to_labels(est_household, filter=filter, scrub=scrub)
    labels.name = "tenure"

    return labels


def vehicles_available(est_household : pd.DataFrame) -> pd.Series:
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
    filter = "^txv_own_|^txv_rent_"
    scrub = filter + "|_vehicle"
    labels = columns_to_labels(est_household, filter=filter, scrub=scrub)
    labels = labels.replace({"no" : "none"})
    labels.name = "vehicles_available"

    return labels