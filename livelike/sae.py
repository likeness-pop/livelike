import geopandas as gpd
import numpy as np
import pandas as pd
from likeness_vitals.vitals import get_censusapikey
from livelike import attribution
from livelike.acs import extract_geographies
from livelike.config import (
    up_base_attributes_person,
    up_base_attributes_household
)
from pymedm import PMEDM, batch, diagnostics

from livelike import acs, est, homesim, multi
from livelike.utils import clear_acs_cache

def estimate(pumas, pmedms, atts, serials, level):
    # If only the based PUMA/P-MEDM is passed
    # convert to dict to mimic replicates structure
    if isinstance(pumas, livelike.acs.puma):
        pumas = {f"{pumas.fips}_0" : pumas}

    if isinstance(pmedms, pymedm.pmedm.PMEDM):
        pmedms = {f"{pmedms.fips}_0" : pumas}

    fips = list(pumas.items())[0][1].fips

    if len(pumas) != len(pmedms):
        raise ValueError(
            "Inputs ``pumas`` and ``pmedms`` must "
            "have the same length."
        )
    
    # pmd = pmedms[f"{fips}_0"]
    # is_seg = np.where(pmd.serial.isin(serials))
    # seg_ct = seg.index.get_level_values(0).\
    #     value_counts(sort=False).\
    #     values
