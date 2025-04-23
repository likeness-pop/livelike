import pathlib
import time

import dill as pickle
from likeness_vitals.vitals import get_censusapikey
from pymedm import PMEDM, batch

from livelike import acs, multi
from livelike.utils import clear_acs_cache

key = get_censusapikey()

nb_pmedm_conv = {}

# clear cache (default cache name)
clear_acs_cache()

# --------------------------------------------------------------------------
# basic_usage.ipynb
print("Generating solution from `basic_usage.ipynb`...")
t1 = time.time()

puma_basic_usage = "4701603"
pup_basic_usage = acs.puma(puma_basic_usage, censusapikey=key)

pmd_basic_usage = PMEDM(
    serial=pup_basic_usage.est_ind.index,
    year=pup_basic_usage.year,
    wt=pup_basic_usage.wt,
    cind=pup_basic_usage.est_ind,
    cg1=pup_basic_usage.est_g1,
    cg2=pup_basic_usage.est_g2,
    sg1=pup_basic_usage.se_g1,
    sg2=pup_basic_usage.se_g2,
)
pmd_basic_usage.solve()

basic_usage_pmedm_conv = float(pmd_basic_usage.res.state.value)

nb_pmedm_conv["basic_usage"] = basic_usage_pmedm_conv

t2 = round((time.time() - t1) / 60.0, 2)
print(f"* `basic_usage.ipynb`: {t2} minutes")

# --------------------------------------------------------------------------
# basic_usage_2020s.ipynb
print("Generating solution from `basic_usage_2020s.ipynb`...")
t1 = time.time()

puma_basic_usage_2020s = "4701501"
pup_basic_usage_2020s = acs.puma(puma_basic_usage_2020s, year=2023, censusapikey=key)

pmd_basic_usage_2020s = PMEDM(
    serial=pup_basic_usage_2020s.est_ind.index,
    year=pup_basic_usage_2020s.year,
    wt=pup_basic_usage_2020s.wt,
    cind=pup_basic_usage_2020s.est_ind,
    cg1=pup_basic_usage_2020s.est_g1,
    cg2=pup_basic_usage_2020s.est_g2,
    sg1=pup_basic_usage_2020s.se_g1,
    sg2=pup_basic_usage_2020s.se_g2,
)
pmd_basic_usage_2020s.solve()

basic_usage_2020s_pmedm_conv = float(pmd_basic_usage_2020s.res.state.value)

nb_pmedm_conv["basic_usage_2020s"] = basic_usage_2020s_pmedm_conv

t2 = round((time.time() - t1) / 60.0, 2)
print(f"* `basic_usage_2020s.ipynb`: {t2} minutes")


# --------------------------------------------------------------------------
# basic_usage__pums_replicates.ipynb
print("Generating solutions from `basic_usage__pums_replicates.ipynb`...")
t1 = time.time()

puma_basic_usage__pums_replicates = "4701602"

mpu_basic_usage__pums_replicates = multi.make_replicate_pumas(
    fips=puma_basic_usage__pums_replicates,
    nreps=5,
    censusapikey=key,
)

pmds_basic_usage__pums_replicates = batch.batch_solve(mpu_basic_usage__pums_replicates)

basic_usage__pums_replicates_pmedm_conv = [
    float(v.res.state.value) for k, v in pmds_basic_usage__pums_replicates.items()
]

nb_pmedm_conv["basic_usage__pums_replicates"] = basic_usage__pums_replicates_pmedm_conv

t2 = round((time.time() - t1) / 60.0, 2)
print(f"* `basic_usage__pums_replicates.ipynb`: {t2} minutes")

# clear cache (default cache name)
clear_acs_cache()


# --------------------------------------------------------------------------
# multi_puma.ipynb
print("Generating solutions from `multi_puma.ipynb`...")
t1 = time.time()

puma_multi_puma = ["4701602", "4701603", "4701604"]

mpu_multi_puma = multi.make_pumas(puma_multi_puma, censusapikey=key)

pmds_multi_puma = batch.batch_solve(mpu_multi_puma)

multi_puma_pmedm_conv = [float(v.res.state.value) for k, v in pmds_multi_puma.items()]
nb_pmedm_conv["multi_puma"] = multi_puma_pmedm_conv

t2 = round((time.time() - t1) / 60.0, 2)
print(f"* `multi_puma.ipynb`: {t2} minutes")


# --------------------------------------------------------------------------
# tract_supertract.ipynb
print("Generating solution from `tract_supertract_2019.ipynb`...")
t1 = time.time()

tract_supertract_puma = "4701603"

pup_tract_supertract = acs.puma(
    tract_supertract_puma, target_zone="trt", keep_geo=True, censusapikey=key
)

pmd_tract_supertract = PMEDM(
    serial=pup_tract_supertract.est_ind.index,
    year=pup_tract_supertract.year,
    wt=pup_tract_supertract.wt,
    cind=pup_tract_supertract.est_ind,
    cg1=pup_tract_supertract.est_g1,
    cg2=pup_tract_supertract.est_g2,
    sg1=pup_tract_supertract.se_g1,
    sg2=pup_tract_supertract.se_g2,
    topo=pup_tract_supertract.topo,
)
pmd_tract_supertract.solve()

tract_supertract_pmedm_conv = float(pmd_tract_supertract.res.state.value)
nb_pmedm_conv["tract_supertract"] = tract_supertract_pmedm_conv

t2 = round((time.time() - t1) / 60.0, 2)
print(f"* `tract_supertract.ipynb`: {t2} minutes")


# --------------------------------------------------------------------------
# tract_supertract.ipynb
print("Generating solution from `tract_supertract_2023.ipynb`...")
t1 = time.time()

tract_supertract_2023_puma = "4701501"

pup_tract_supertract_2023 = acs.puma(
    tract_supertract_2023_puma,
    year=2023,
    target_zone="trt",
    keep_geo=True,
    censusapikey=key
)

pmd_tract_supertract_2023 = PMEDM(
    serial=pup_tract_supertract_2023.est_ind.index,
    year=pup_tract_supertract_2023.year,
    wt=pup_tract_supertract_2023.wt,
    cind=pup_tract_supertract_2023.est_ind,
    cg1=pup_tract_supertract_2023.est_g1,
    cg2=pup_tract_supertract_2023.est_g2,
    sg1=pup_tract_supertract_2023.se_g1,
    sg2=pup_tract_supertract_2023.se_g2,
    topo=pup_tract_supertract_2023.topo,
)
pmd_tract_supertract_2023.solve()

tract_supertract_2023_pmedm_conv = float(pmd_tract_supertract_2023.res.state.value)
nb_pmedm_conv["tract_supertract_2023"] = tract_supertract_2023_pmedm_conv

t2 = round((time.time() - t1) / 60.0, 2)
print(f"* `tract_supertract_2023.ipynb`: {t2} minutes")


# --------------------------------------------------------------------------
# save out pickled results
pickle_path = pathlib.Path("livelike", "tests", "notebook_pmedm_solutions")
with open(pickle_path / "notebook_pmedm_solutions.pkl", "wb") as f:
    pickle.dump(nb_pmedm_conv, f)


print("Done.\n-------------------------------------")
