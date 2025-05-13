import numpy as np
import pandas as pd

from .acs import puma


def estimate(
    pumas: puma | dict[str, puma],
    pmedms: dict[str, ...],
    serial: pd.Index | pd.MultiIndex,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Estimates counts, and optionally proportions, of a population
    segment matching some condition of interest using a solved P-MEDM
    problem and input PUMA data, or an ensemble of these.

    Parameters
    ----------
    pumas : livelike.acs.puma | dict[str, livelike.acs.puma]
        Input PUMA data.
    pmedms : pymedm.pmedm.PMEDM | dict[str, pymedm.pmedm.PMEDM]
        P-MEDM problems. Must have a solution including an allocation matrix.
    serial : pandas.Index | pandas.MultiIndex
        Index of person or residence IDs defining the population segment within
        the PUMA. Person or residence (``'household'``) level is inferred internally
        based on whether a single index described by PUMS serial number
        (``SERIALNO``) is given (residence-level) vs. a multi-index (person)
        described by ``SERIALNO`` and household membership order (``SPORDER``).
        In the latter case, the number of household members by ``SERIALNO:SPODER``
        is tabulated to produce the person level estimates.
    normalize : bool = False
        Whether to normalize the estimate by total population
        (``'population'``) or total residences (``'household'``).

    Returns
    -------
    seg_est : pandas.DataFrame
        A Pandas DataFrame with rows representing small areas 
        and columns representing estimates, numbered as 
        ``rep{0...80}`` where ``rep0`` is the base estimate
        taken from a P-MEDM solution on the full PUMS weights.
    """

    # If only the base PUMA/P-MEDM is passed
    # convert to dict to mimic replicates structure
    if isinstance(pumas, puma):
        pumas = {f"{pumas.fips}_0": pumas}
    fips = list(pumas.items())[0][1].fips

    if not isinstance(pmedms, dict):
        pmedms = {f"{fips}_0": pmedms}

    if len(pumas) != len(pmedms):
        raise ValueError("Inputs ``pumas`` and ``pmedms`` must have the same length.")

    # infer level from index type
    if isinstance(serial, pd.MultiIndex):
        level = "person"
        if serial.names != ["SERIALNO", "SPORDER"]:
            raise ValueError(
                "Input ``serial`` must be named as ``['SERIALNO', 'SPORDER']``."
            )
    elif isinstance(serial, pd.Index):
        level = "household"
        if serial.name != "SERIALNO":
            raise ValueError("Input ``serial`` must be named as ``'SERIALNO'``.")
    else:
        raise ValueError(
            "Input ``serial`` must be a pandas.DataFrame Index or MultiIndex."
        )

    if level == "person":
        # count household members associated with target IDs
        seg_ct = serial.get_level_values(0).value_counts(sort=False).values
    else:  # household
        # one count per residence
        seg_ct = np.array([1] * len(serial))

    # flag segments within full PUMS index for PUMA
    pmd = pmedms[f"{fips}_0"]
    if level == "person":
        is_seg = np.where(pmd.serial.isin(serial.get_level_values(0)))
    else:  # household
        is_seg = np.where(pmd.serial.isin(serial))

    seg_est_ = np.array(
        [
            (pmedms[f"{fips}_{r}"].almat[is_seg] * seg_ct[:, None]).sum(axis=0)
            for r in range(len(pmedms))
        ]
    )

    # normalize counts if specified
    if normalize:
        if level == "person":
            totals_ = np.array(
                [
                    (
                        pmedms[f"{fips}_{r}"].almat
                        * pumas[f"{fips}_{r}"].est_ind.population.values[:, None]
                    ).sum(axis=0)
                    for r in range(len(pmedms))
                ]
            )
        else:  # household
            totals_ = np.array(
                [pmedms[f"{fips}_{r}"].almat.sum(axis=0) for r in range(len(pmedms))]
            )
        seg_est_ = seg_est_ / totals_

    # format results
    geoids = list(pumas.items())[0][1].est_g2.index.values
    seg_est = pd.DataFrame(
        seg_est_.T, 
        index=geoids, 
        columns=[f"rep{str(r)}" for r in range(len(seg_est_))],
    )

    return seg_est