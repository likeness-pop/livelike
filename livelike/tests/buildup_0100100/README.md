# PUMA 0100100

## Lauderdale, Colbert & Franklin Counties – Alabama

* The data in this directory is pre-built for testing and can be regenerated with the script below.
* The path to your census API key may need to change.

```python
%load_ext watermark
%watermark

import pathlib

import geopandas
import likeness_vitals
import numpy
import pandas
import pymedm

import livelike

%watermark -w
%watermark -iv

key = likeness_vitals.vitals.get_censusapikey()

# -- Lauderdale, Colbert & Franklin Counties – Alabama
puma_fips = "0100100"

pup = livelike.acs.puma(puma_fips, censusapikey=key)

pmd = pymedm.PMEDM(
    pup.year,
    pup.est_ind.index,
    pup.wt,
    pup.est_ind,
    pup.est_g1,
    pup.est_g2,
    pup.se_g1,
    pup.se_g2,
)

pmd.solve()

print(f"{float(pmd.res.state.value)}")

pandas.DataFrame(pmd.almat).to_parquet("almat.parquet")

pup.est_ind.to_parquet("est_ind.parquet")
pup.est_g2.to_parquet("est_g2.parquet")
pup.sporder.to_parquet("sporder.parquet")

query = "ESR=1&NAICSP=6111&OCCP=2300:2320&WKHP=40:999"

pums_segment_ids = livelike.acs.extract_pums_segment_ids(
    puma_fips, "person", query, censusapikey=key
)
pums_segment_ids.to_parquet("pums_segment_ids.parquet")
```
