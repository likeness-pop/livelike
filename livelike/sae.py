import numpy as np
import pandas as pd

from .acs import puma


def estimate(
    pumas: puma | dict[str, puma],
    pmedms: dict[str, ...],
    serial: pd.Index | pd.MultiIndex,
    normalize: bool = False,
) -> dict:
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
    est : dict
        A dictionary of pandas.DataFrame numpy.ndarray objects containing
        the point estimates (``'est'``), standard errors (``'se'``), and
        coefficients of variation (``'cv'``) for each area.
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
    elif isinstance(serial, pd.core.Index):
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
    # count number of ests (1: base only, 2+: ensemble)
    n_ests = seg_est_.shape[0]

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

    # collate point estimates, standard errors, coeffs of variation
    est = {}
    if n_ests > 1:
        est["est"] = np.apply_along_axis(func1d=np.mean, axis=0, arr=seg_est_)
        est["se"] = np.apply_along_axis(func1d=np.std, axis=0, arr=seg_est_)
        est["cv"] = est["se"] / est["est"]
    elif n_ests == 1:
        est["est"] = seg_est_.flatten()
        est["se"] = np.nan
        est["cv"] = np.nan
    else:
        raise RuntimeError(
            f"Something went wrong. Estimates should be 1d or 2d but got{n_ests}d."
        )

    return est
