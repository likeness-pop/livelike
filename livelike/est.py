import numpy as np
import pandas as pd
from likeness_vitals.vitals import match


def tabulate_by_serial(
    syp: pd.DataFrame,
    serial: np.ndarray | pd.Series,
    level: str,
) -> pd.Series:
    """Tabulates the number of people or households matching a specified set of
    PUMS household IDs, appends the counts to synthetic populations, then
    tabulates the counts by simulation index (``sim``) and FIPS (``geoid``).
    If a column for ``year`` is present, uses that as a third grouping variable.

    Parameters
    ----------
    syp : pandas.DataFrame
        Synthetic populations.
    serial : numpy.ndarray | pandas.Series
        Target PUMS household IDs.
    level : str
        Tabulation level (``person`` or ``household``).

    Returns
    -------
    tab : pandas.Series
        Counts of people or households matching the target PUMS IDs ``serial``
        by simulation index (``sim``) and FIPS (``geoid``).
    """

    levels = np.array(["person", "household"])
    assert any(levels == level), (
        "Argument ``level`` must be one of: ``person``, ``household``."
    )

    if not isinstance(serial, np.ndarray | pd.Series):
        raise TypeError(
            "``serial`` must be ``numpy.ndarray`` or ``pandas.Series``. Check type."
        )

    # subset synthetic population to those present in ``serial``
    syp = syp[syp.index.isin(serial)]

    # grouping vars
    grouping = ["sim", "geoid"]

    if any(syp.columns == "year"):
        grouping.append("year")

    if level == "person":
        hnp = serial.value_counts()
        syp = syp.copy()
        syp["npers"] = match(x1=syp, x2=hnp)
        tab = syp.groupby(grouping).apply(lambda x: sum(x["count"] * x["npers"]))
    else:
        # household
        tab = syp.groupby(grouping).size()

    return tab


def tabulate_by_count(
    syp: pd.DataFrame,
    count: pd.Series,
    label: None | str = None,
):
    """Attaches a count variable to synthetic populations, then tabulates
    the counts by simulation index and FIPS (``geoid``). If a column for
    ``year`` is present, uses that as a third grouping variable.

    Parameters
    ----------
    syp : pandas.DataFrame
        Synthetic populations.
    count : pandas.Series
        Count variable.
    label : None | str = None
        Label for the count column. If ``None``, defaults to ``'count'``.

    Returns
    -------
    pandas.Series
        The count variable aggregated by simulation index
        (``'sim'``) and FIPS (``'geoid'``).
    """

    if count.name is None:
        if label is None:
            count.name = "count"
        else:
            count.name = label

    # grouping vars
    grouping = ["sim", "geoid"]

    if any(syp.columns == "year"):
        grouping.append("year")

    syp = syp.copy()
    syp[count.name] = match(x1=syp, x2=count)

    return syp.groupby(grouping)[count.name].sum()


def to_prop(tab: pd.Series, total: pd.Series) -> pd.Series:
    """Normalizes tabulations on synthetic populations by population or
    sub-population totals by simulation number (``sim``) and FIPS (``geoid``).

    Parameters
    ----------
    tab : pandas.Series
        A variable tabulated on a residential synthetic population, indexed
        by simulation number (``sim``) and FIPS (``geoid``).
    total : pandas.Series
        Population or sub-population totals, indexed by simulation number
        (``sim``) and FIPS (``geoid``), and possibly ``year``.

    Returns
    -------
    pandas.Series
        The normalized tabulation, indexed by simulation number (``sim``) and
        FIPS (``geoid``).
    """

    tab.name = "est"
    total.name = "total"

    tp = pd.concat([tab, total], axis=1)

    tp.est = tp.est.fillna(0)

    return tp.est / tp.total


def monte_carlo_estimate(x):
    """Performs a Monte Carlo estimate by FIPS (``geoid``) of
    tabulations or proportional estimates made on synthetic
    populations. If a ``year`` index is present, also groups by year.

    Parameters
    ----------
    x : pandas.Series
        Tabulation or proportional estimate indexed by simulation number
        (``sim``) and FIPS (``geoid``).

    Returns
    -------
    mc : pandas.DataFrame
        Monte Carlo estimates (``est``) and standard errors (``se``)
        by FIPS (``geoid``).
    """

    # grouping indices
    grouping = ["geoid"]

    if any([i == "year" for i in x.index.names]):  # noqa C419
        grouping.append("year")

    mc = x.groupby(grouping).agg(["mean", "std"])
    mc.columns = ["est", "se"]

    return mc
