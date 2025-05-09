import geopandas as gpd
import numpy as np
import pandas as pd
from likeness_vitals.vitals import get_censusapikey
import livelike
from livelike import attribution
from livelike.config import (
    up_base_attributes_person,
    up_base_attributes_household
)
import pymedm

from livelike import acs, est, homesim, multi
from livelike.utils import clear_acs_cache

def estimate(pumas, pmedms, atts, condition, level, normalize=False) -> dict:
    """
    Estimates counts, and optionally proportions, of a population 
    segment matching some condition of interest using a solved P-MEDM 
    problem and input PUMA data, or an ensemble of these. 

    Parameters
    ----------
    pumas : livelike.acs.puma | dict
        Input PUMA data. 
    pmedms : pymedm.pmedm.PMEDM | dict
        P-MEDM problems. Must have a solution including 
        an allocation matrix.
    atts : pandas.DataFarme
        Population attributes 
        (see ``livelike.attribution.build_attributes()``).
    condition : str
        Condition for querying attributes to define the 
        population segment of interest. 
    level : str
        Level of interest (``'person'`` or ``'household'``).
    normalize : bool=False
        Whether to normalize the estimate by total population 
        (``'population'``) or total residences (``'household'``).
    """
    # If only the based PUMA/P-MEDM is passed
    # convert to dict to mimic replicates structure
    if isinstance(pumas, livelike.acs.puma):
        pumas = {f"{pumas.fips}_0" : pumas}

    if isinstance(pmedms, pymedm.pmedm.PMEDM):
        pmedms = {f"{pmedms.fips}_0" : pumas}

    if len(pumas) != len(pmedms):
        raise ValueError(
            "Inputs ``pumas`` and ``pmedms`` must "
            "have the same length."
        )
    
    # Get segment by condition
    if level == "person" and not isinstance(
        atts.index, "pd.core.indexes.multi.MultiIndex"):
        raise ValueError(
            "Attributes (``atts``) must be indexed by " \
            "both ``SERIALNO`` and ``SPORDER``."
        )
    seg = atts.loc[eval(condition)]
    if len(seg) == 0:
        raise RuntimeError(
            "No cases found matching the condition."
        )

    # 
    if level == "person":
        serials = seg.index.get_level_values(0)
        seg_ct = serials.value_counts(sort=False).values
    else: # household
        serials = seg.index
        seg_ct = None
    
    fips = list(pumas.items())[0][1].fips
    pmd = pmedms[f"{fips}_0"]
    is_seg = np.where(pmd.serial.isin(serials))

    seg_est_ = np.array([
        (pmedms[f"{pumas}_{r}"].almat[is_seg] * seg_ct[:,None]).\
            sum(axis=0) 
        for r in range(len(pmedms))
    ])
    nd_seg_est_ = seg_est_.ndims

    if normalize:
        if level == "person":
            pop_totals_ = np.array([
                (pmedms[f"{fips}_{r}"].almat * pumas[f"{fips}_{r}"].\
                est_ind.population.values[:,None]).\
                    sum(axis=0) 
                for r in range(len(pmedms))
            ])
        else: # household
            pop_totals_ = np.array([
                pmedms[f"{fips}_{r}"].almat.sum(axis=0)
                for r in range(len(pmedms))
            ])
        seg_prop_ = seg_est_ / pop_totals_        

    est = {}
    if nd_seg_est_ == 2:
        est["count"] = np.apply_along_axis(func1d=np.mean, axis=0, arr=seg_est_)
        if normalize:
            est["prop"] = np.apply_along_axis(func1d=np.mean, axis=0, arr=seg_prop_)
        est["se"] = np.apply_along_axis(func1d=np.std, axis=0, arr=seg_est_)
        est["cv"] = est["se"] / est["est"]

    elif nd_seg_est_ == 1:
        est["count"] = seg_est_
        est["prop"] = seg_prop_
        est["se"] = np.nan
        est["cv"] = np.nan

    else:
        raise RuntimeError(
            "Something went wrong. Estimates should be 1d or 2d but got" \
            f"{nd_seg_est_}d."
        )

    return est