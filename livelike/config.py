import os
import pathlib

import pandas

data_dir = pathlib.Path(__file__).parent / "data"

# changes to tract defs in 2010s
changes_2010 = dict(
    zip(
        [
            "36053940101",
            "36053940102",
            "36053940103",
            "36053940200",
            "36053940300",
            "36053940401",
            "36053940403",
            "36053940600",
            "36053940700",
            "36065940000",
            "36065940100",
            "36065940200",
            "36065023000",
            "04019002701",
            "04019002903",
            "04019410501",
            "04019410502",
            "04019410503",
            "04019470400",
            "04019470500",
            "06037930401",
        ],
        [
            "36053030101",
            "36053030102",
            "36053030103",
            "36053030200",
            "36053030300",
            "36053030401",
            "36053030403",
            "36053030600",
            "36053030402",
            "36065024800",
            "36065027400",
            "36065024900",
            "36065024900",
            "04019002704",
            "04019002906",
            "04019004118",
            "04019004121",
            "04019004125",
            "04019005200",
            "04019005300",
            "06037137000",
        ],
        strict=True,
    )
)

rel = pandas.DataFrame()
for d in ["10", "20"]:
    fname = f"20{d}_Census_Tract_to_20{d}_PUMA.zip"
    fpath = os.path.join(os.path.dirname(__file__), "data", f"{fname}")

    rel_suffix = "2020" if d == "20" else ""

    # u = f"https://www2.census.gov/geo/docs/maps-data/data/rel{rel_suffix}/20{d}_Census_Tract_to_20{d}_PUMA.txt"
    u = fpath
    rel_part = pandas.read_csv(u, dtype="str")

    if d == "10":
        # changes to county codes in 2010s
        rel_part.loc[
            (rel_part.STATEFP == "02") & (rel_part.COUNTYFP == "270"), "COUNTYFP"
        ] = "158"
        rel_part.loc[
            (rel_part.STATEFP == "46") & (rel_part.COUNTYFP == "113"), "COUNTYFP"
        ] = "102"
        rel_part.loc[
            (rel_part.STATEFP == "51") & (rel_part.COUNTYFP == "515"), "COUNTYFP"
        ] = "019"

    rel_part["puma"] = rel_part.STATEFP + rel_part.PUMA5CE
    rel_part["geoid"] = rel_part.STATEFP + rel_part.COUNTYFP + rel_part.TRACTCE
    rel_part["census"] = f"20{d}"

    if d == "10":
        # changes to tract codes in 2010s
        for g in changes_2010:
            rel_part.loc[rel_part["geoid"] == g, "geoid"] = changes_2010[g]

    rel = pandas.concat([rel, rel_part], axis=0)

rel = rel.groupby("census")

# universe constraints
universe_constraints = [
    "population",
    "housing_units",
    "occhu",
    "civ_noninst_pop",
    "group_quarters_pop",
]

# universe variable codes
universe_codes = [
    "B01001_001E",
    "B25001_001E",
    "B25003_001E",
    "B01001_001M",
    "B25001_001M",
    "B25003_001M",
]

# PUMS variable data types
pums_dtypes = dict(  # noqa C418
    {
        "SERIALNO": "str",
        "SPORDER": "int",
        "WGTP": "int",
        "WGTP1": "int",
        "WGTP2": "int",
        "WGTP3": "int",
        "WGTP4": "int",
        "WGTP5": "int",
        "WGTP6": "int",
        "WGTP7": "int",
        "WGTP8": "int",
        "WGTP9": "int",
        "WGTP10": "int",
        "WGTP11": "int",
        "WGTP12": "int",
        "WGTP13": "int",
        "WGTP14": "int",
        "WGTP15": "int",
        "WGTP16": "int",
        "WGTP17": "int",
        "WGTP18": "int",
        "WGTP19": "int",
        "WGTP20": "int",
        "WGTP21": "int",
        "WGTP22": "int",
        "WGTP23": "int",
        "WGTP24": "int",
        "WGTP25": "int",
        "WGTP26": "int",
        "WGTP27": "int",
        "WGTP28": "int",
        "WGTP29": "int",
        "WGTP30": "int",
        "WGTP31": "int",
        "WGTP32": "int",
        "WGTP33": "int",
        "WGTP34": "int",
        "WGTP35": "int",
        "WGTP36": "int",
        "WGTP37": "int",
        "WGTP38": "int",
        "WGTP39": "int",
        "WGTP40": "int",
        "WGTP41": "int",
        "WGTP42": "int",
        "WGTP43": "int",
        "WGTP44": "int",
        "WGTP45": "int",
        "WGTP46": "int",
        "WGTP47": "int",
        "WGTP48": "int",
        "WGTP49": "int",
        "WGTP50": "int",
        "WGTP51": "int",
        "WGTP52": "int",
        "WGTP53": "int",
        "WGTP54": "int",
        "WGTP55": "int",
        "WGTP56": "int",
        "WGTP57": "int",
        "WGTP58": "int",
        "WGTP59": "int",
        "WGTP60": "int",
        "WGTP61": "int",
        "WGTP62": "int",
        "WGTP63": "int",
        "WGTP64": "int",
        "WGTP65": "int",
        "WGTP66": "int",
        "WGTP67": "int",
        "WGTP68": "int",
        "WGTP69": "int",
        "WGTP70": "int",
        "WGTP71": "int",
        "WGTP72": "int",
        "WGTP73": "int",
        "WGTP74": "int",
        "WGTP75": "int",
        "WGTP76": "int",
        "WGTP77": "int",
        "WGTP78": "int",
        "WGTP79": "int",
        "WGTP80": "int",
        "PWGTP": "int",
        "PWGTP1": "int",
        "PWGTP2": "int",
        "PWGTP3": "int",
        "PWGTP4": "int",
        "PWGTP5": "int",
        "PWGTP6": "int",
        "PWGTP7": "int",
        "PWGTP8": "int",
        "PWGTP9": "int",
        "PWGTP10": "int",
        "PWGTP11": "int",
        "PWGTP12": "int",
        "PWGTP13": "int",
        "PWGTP14": "int",
        "PWGTP15": "int",
        "PWGTP16": "int",
        "PWGTP17": "int",
        "PWGTP18": "int",
        "PWGTP19": "int",
        "PWGTP20": "int",
        "PWGTP21": "int",
        "PWGTP22": "int",
        "PWGTP23": "int",
        "PWGTP24": "int",
        "PWGTP25": "int",
        "PWGTP26": "int",
        "PWGTP27": "int",
        "PWGTP28": "int",
        "PWGTP29": "int",
        "PWGTP30": "int",
        "PWGTP31": "int",
        "PWGTP32": "int",
        "PWGTP33": "int",
        "PWGTP34": "int",
        "PWGTP35": "int",
        "PWGTP36": "int",
        "PWGTP37": "int",
        "PWGTP38": "int",
        "PWGTP39": "int",
        "PWGTP40": "int",
        "PWGTP41": "int",
        "PWGTP42": "int",
        "PWGTP43": "int",
        "PWGTP44": "int",
        "PWGTP45": "int",
        "PWGTP46": "int",
        "PWGTP47": "int",
        "PWGTP48": "int",
        "PWGTP49": "int",
        "PWGTP50": "int",
        "PWGTP51": "int",
        "PWGTP52": "int",
        "PWGTP53": "int",
        "PWGTP54": "int",
        "PWGTP55": "int",
        "PWGTP56": "int",
        "PWGTP57": "int",
        "PWGTP58": "int",
        "PWGTP59": "int",
        "PWGTP60": "int",
        "PWGTP61": "int",
        "PWGTP62": "int",
        "PWGTP63": "int",
        "PWGTP64": "int",
        "PWGTP65": "int",
        "PWGTP66": "int",
        "PWGTP67": "int",
        "PWGTP68": "int",
        "PWGTP69": "int",
        "PWGTP70": "int",
        "PWGTP71": "int",
        "PWGTP72": "int",
        "PWGTP73": "int",
        "PWGTP74": "int",
        "PWGTP75": "int",
        "PWGTP76": "int",
        "PWGTP77": "int",
        "PWGTP78": "int",
        "PWGTP79": "int",
        "PWGTP80": "int",
        "ADJINC": "float",
        "RELSHIPP": "int",
        "RELP": "int",
        "VACS": "int",
        "TYPE": "int",
        "TYPEHUGQ": "int",
        "SEX": "int",
        "ESR": "int",
        "COW": "int",
        "OCCP": "str",
        "NAICSP": "str",
        "SCHG": "int",
        "WKHP": "int",
        "JWMNP": "int",
        "JWRIP": "int",
        "JWTRNS": "int",
        "HINCP": "int",
        "FINCP": "int",
        "OIP": "int",
        "PAP": "int",
        "RETP": "int",
        "SEMP": "int",
        "SSIP": "int",
        "SSP": "int",
        "WAGP": "int",
        "PERNP": "int",
        "PINCP": "int",
        "INTP": "int",
        "SCH": "int",
        "YBL": "int",
        "BLD": "int",
        "AGEP": "int",
        "RAC1P": "int",
        "HISP": "int",
        "HHT": "int",
        "TEN": "int",
        "R18": "int",
        "R60": "int",
        "POVPIP": "int",
        "HHL": "int",
        "POBP": "int",
        "VPS": "int",
        "SCHL": "int",
        "DIS": "int",
        "BDSP": "int",
        "NP": "int",
        "VEH": "int",
        "LNGI": "int",
        "GQ": "int",
        "ENG": "int",
        "RMSP": "int",
        "HFL": "int",
        "OCPIP": "int",
        "GRPIP": "int",
        "HINS1": "int",
        "HINS2": "int",
        "HINS3": "int",
        "HINS4": "int",
        "HINS5": "int",
        "HINS6": "int",
        "HINS7": "int",
        "HICOV": "int",
        "YRBLT": "int",
        "ACCESS": "int",
        "ACCESSINET": "int",
    }
)

# PUMS variables requiring income adjustment
pums_income_vars = [
    "HINCP",
    "FINCP",
    "OIP",
    "PAP",
    "RETP",
    "SEMP",
    "SSIP",
    "SSP",
    "WAGP",
    "PERNP",
    "PINCP",
    "INTP",
]

# PUMS helper functions needing a year argument
need_year = [
    "occhu",
    "group_quarters",
    "group_quarters_pop",
    "civ_noninst_pop",
    "year_built",
    "internet",
]

# P-MEDM constraints
constraints = pandas.read_csv(data_dir / "constraints.csv").fillna("")

# ordering scheme for prebuilt constraint themes
up_constraints_theme_order = [
    "universe",
    "worker",
    "student",
    "mobility",
    "demographic",
    "social",
    "economic",
    "housing",
]

# UrbanPop base constraints selection
up_base_constraints_selection = {
    "universe": True,
    "worker": True,
    "student": True,
    "mobility": True,
    "demographic": ["hhtype_hhsize"],
}

# UrbanPop expanded constraints selection
up_expanded_constraints_selection = {
    "universe": True,
    "worker": True,
    "student": True,
    "mobility": True,
    "demographic": [
        "sex_age",
        "hhtype",
        "hhtype_hhsize",
    ],
    "social": [
        "race",
        "hsplat",
    ],
    "economic": [
        "hhinc",
        "ipr",
    ],
    "housing": [
        "units",
        "year_built",
    ],
}

# geographic levels for constraint selection
geo_levels = {
    "bg": ["bg"],
    "trt": ["bg", "trt"],
}

# Census Microdata API request limit
cmd_api_request_limit = 50
