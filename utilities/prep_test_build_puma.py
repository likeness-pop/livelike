import pathlib
import time

import pandas as pd
from likeness_vitals.vitals import get_censusapikey

from livelike import acs
from livelike.config import constraints, up_base_constraints_selection

puma_data_path = pathlib.Path("livelike", "tests", "puma")
parq = "{}.parquet"


def columns_name(df: pd.DataFrame, name: str = "constraint") -> pd.DataFrame:
    """Helper to rename columns axis to ``name``."""
    return df.rename_axis(columns=name)


def write_parq(df: pd.DataFrame, name: str):
    """Helper to write a parquet file."""
    df.to_parquet(puma_data_path / parq.format(name))


# --------------------------------------------------------------------------
t1 = time.time()
print("Starting `prep_test_build_puma.py`...")

# read Census API Key
key = get_censusapikey()

year = 2019
constraints = constraints.loc[
    (constraints.begin_year <= year) & (constraints.end_year >= year)
]
constraints = constraints[constraints.geo_base_level == "bg"]

sel = up_base_constraints_selection.copy()
sel.update({"economic": ["poverty"]})

print(f"\n\nConstraint dims:{constraints.shape}\n\n")


# make PUMA
p = "4701604"
pup = acs.puma(p, constraints, constraints_selection=sel, censusapikey=key)


## Regenerate Test data
write_parq(pup.est_ind, "est_ind")
write_parq(columns_name(pup.est_g1), "est_trt")
write_parq(columns_name(pup.est_g2), "est_bg")
write_parq(columns_name(pup.se_g1), "se_trt")
write_parq(columns_name(pup.se_g2), "se_bg")
write_parq(pd.DataFrame(pup.wt), "wt")
write_parq(pup.sporder, "sporder")

t2 = round((time.time() - t1) / 60.0, 2)
print(f"* `prep_test_build_puma.py`: {t2} minutes")
