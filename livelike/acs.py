import pathlib
import urllib
import warnings

import geopandas as gpd
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
import requests
from libpysal import weights
from pygris.data import get_census
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from . import pums
from .config import (
    cmd_api_request_limit,
    constraints,
    geo_levels,
    need_year,
    pums_dtypes,
    pums_income_vars,
    rel,
    universe_codes,
    universe_constraints,
    up_base_constraints_selection,
    up_constraints_theme_order,
)
from .utils import _ensure_cache_folder_input

pd.options.mode.chained_assignment = None

# for evaluating whether we are using default prebuilt constraints
default_constraints = constraints.copy()


class puma:  # noqa N801
    """Baseline PUMA representation by ACS constraints
    with the hierarchy PUMA > Tract > Block Group.

    Parameters
    ----------
    fips : str
        Seven-digit PUMA FIPS code.
    constraints : pandas.DataFrame (config.constraints)
        Constraining variables for P-MEDM, formatted with a constraint
        name ('constraint') and ACS variable code ('code').
    constraints_selection : None | dict = None
        P-MEDM constraints to use. Key-value pairs with keys representing
        ACS variable themes and values representing specific subjects (tables).

        If the value passed is a ``bool`` type, it must be ``True``, which
        will include variables for all subjects in the theme. Passing in a
        ``False`` value will raise an error.

        If the value passed is a ``list`` type, only listed subjects will be
        included in the result.

        If using prebuilt constraints (``config.constraints`` - default), and
        no selection is provided, will default to
        ``config.up_base_constraints_selection``. If using custom constraints
        and no selection is provided, will use the provided constraints as-is.
    constraints_theme_order : None | list = None
        Ordering scheme for constraints ``theme`` column. If ``None`` (default) and
        ``config.constraints`` is passed to ``constraints`` parameter (default),
        orders by ``config.up_constraints_theme_order``.
        If a custom DataFrame is passed to ``constraints`` and an ordering scheme
        is not provided, orders themes by order of first occurrence.
    target_zone : str = 'bg'
        The target zone type for spatial allocation. Options include: ``'bg'``, which
        creates a block group (target) - tract (aggregate) hierarchy, and ``'trt'``,
        which creates a tract (target) and super-tract (aggregate) hierarchy.
    year : int = 2019
        ACS 5-Year Estimates vintage (default ``2019``). Only accepts years for
        which samples consist exclusively of 2010 PUMAs (2016 - 2019). 2020 is
        temporarily disabled.
    cache : bool = False
        Whether to cache ACS files locally. These are saved in Apache
        Parquet format. For more information see [https://parquet.apache.org/].
    keep_intermediates : bool = False
        Should intermediate P-MEDM household and person-level constraints
        (pre-aggregation) be kept?
    make_trt_geo : bool = True
        When ``target_zone='trt'``, extracts census tract boundaries from the
        TIGER Web Mapping Service. Default ``True`` and should not be modified
        except as accessed internally by other ``livelike`` functions.
    keep_geo : bool = False
        When creating a tract/supertract representation of a PUMA, keep the geometries
        used to perform regionalization of tracts into supertracts?
    make_super_trt : bool = True
        When ``target_zone='trt'``, creates supertracts as the aggregation zones
        (``g2``). Default ``True`` and should not be modified except as accessed
        internally by other ``livelike`` functions.
    make_super_trt_method : str = 'louvain'
        Method for for creating supertracts. Currently supports only
        ``'louvain'``. See ``make_supertracts()`` for more information.
    replicate : int = 0
        Replicate weights to use, from 1 - 80. The default value of 0 uses the base
        PUMS weights instead of replicate weights. If a value from 1 - 80 is entered,
        generates weights relative to the corresponding replicate index.
    random_state : int = 808
        Random starting seed for tract-supertract regionalization.
    censusapikey : None | str = None
        Optional Census API key.
    create_puma : bool = True
        Create the full ``puma`` object ready for use in ``pymedm.PMEDM()``
        upon initialization – ``True``. Otherwise – ``False`` –, following
        initialization build incrementally with the following methods (in order):
            1. Clean up constraints temporally: ``clean_constraints()``
            2. Extract initial P-MEDM constraints: ``build_individual_constraints()``
            3. Build tract input (either source or target): ``build_tract_input()``
            4. Build target input
                – if ``target_zone='bg''``
                    a. Build block group input: ``build_block_group_input()``
                – elif ``target_zone='trt''``
                    - if ``make_trt_geo=True``
                        a. Build tract geography: ``build_tract_geography()``
                    - if ``make_super_trt=True``
                        b. Build super tracts: ``build_super_tracts()``
                    - elif ``make_super_trt=False``
                        b. Bypass super tracts: ``bypass_super_tracts()``
            5. Final clean up and column validation: ``validate_columns()``
    verbose : bool = False
        Whether to print information about data construction.
    cache_folder : str | pathlib.Path = './livelike_acs_cache'
        The cache folder to store queried ACS inputs. This default value
        creates the folder in the current working directory if not present.
    """

    def __init__(
        self,
        fips: str,
        constraints: pd.DataFrame = constraints,
        constraints_selection: None | list = None,
        constraints_theme_order: None | list = None,
        target_zone: str = "bg",
        year: int = 2019,
        cache: bool = False,
        keep_intermediates: bool = False,
        make_trt_geo: bool = True,
        keep_geo: bool = False,
        make_super_trt: bool = True,
        make_super_trt_method: str = "louvain",
        replicate: int = 0,
        random_state: int = 808,
        censusapikey: None | str = None,
        create_puma: bool = True,
        verbose: bool = False,
        cache_folder: str | pathlib.Path = "./livelike_acs_cache",
    ):
        # Sanity checks ----------------------------
        # PUMA FIPS code must be a string and must be 7 characters
        if not isinstance(fips, str):
            raise TypeError(
                f"``fips`` argument must be str. Passed in: {type(fips).__name__}"
            )
        fips_len = len(fips)
        if fips_len != 7:
            raise ValueError(
                f"``fips`` argument must be 7 characters. Length: {fips_len}"
            )

        if not ((2016 <= year <= 2019) or (year >= 2023)):
            raise ValueError("Supported years are 2016 - 2019 and 2023+.")
        census = "2010" if year < 2020 else "2020"

        # must have a valid target zone
        if target_zone not in ["bg", "trt"]:
            raise ValueError("Target zone must be one of ``bg``, ``trt``.")

        # regionalization (tract aggregation) method must be supported
        # TODO: update this check before v2.0.0 release
        deprecated_methods = ["MaxP", "region_kmeans"]
        valid_super_trt_methods = ["louvain"] + deprecated_methods
        if make_super_trt_method not in valid_super_trt_methods:
            raise ValueError(
                "The value passed into `make_super_trt_method` is not "
                f"valid `{make_super_trt_method}`. See doctrings."
            )

        # check replicate number
        if (replicate < 0) | (replicate > 80):
            raise ValueError("Replicate number must be between 1 - 80.")
        replicate = replicate if replicate > 0 else None

        # Set attributes ---------------------------
        self.fips = fips
        self.constraints = constraints
        self.constraints_selection = constraints_selection
        self.constraints_theme_order = constraints_theme_order
        self.target_zone = target_zone
        self.cache = cache
        self.year = year
        self.keep_intermediates = keep_intermediates
        self.make_trt_geo = make_trt_geo
        self.keep_geo = keep_geo
        self.make_super_trt = make_super_trt
        self.make_super_trt_method = make_super_trt_method
        self.replicate = replicate
        self.random_state = random_state
        self.censusapikey = censusapikey
        self.verbose = verbose
        self.rel = rel.get_group(census)

        # validate cache input and create directory if it doesn't exist
        if cache:
            self.cache_folder = _ensure_cache_folder_input(cache_folder)
        else:
            self.cache_folder = cache_folder

        # check for default/prebuild constraints specified
        self.using_default_constraints = self.constraints.equals(default_constraints)

        if self.verbose:
            print(f"Using default constraints? {self.using_default_constraints}\n")

        # Isolate tract IDs in study area
        self.all_trt_geoids = self.rel[self.rel["puma"] == self.fips]["geoid"].values

        # Create PUMA object ready for ingestion into ``pymedm.PMEDM``
        if create_puma:
            self = self.clean_constraints()
            self = self.build_individual_constraints()
            self = self.build_tract_input()

            if self.target_zone == "bg":
                self = self.build_block_group_input()
            else:
                if make_trt_geo:
                    self = self.build_tract_geography()

                if self.make_super_trt:
                    self = self.build_super_tracts()
                else:
                    self = self.bypass_super_tracts()

            self = self.validate_columns()

        # Estimate group quarters for 2023+
        # when block groups (``bg``) are the target geometry
        if self.year >= 2023 and target_zone == "bg":
            self.estimate_group_quarters()

    def clean_constraints(self):
        """Clean up constraints temporally."""
        after_begin = self.constraints.begin_year <= self.year
        before_end = self.constraints.end_year >= self.year

        # Get year-specific constraints
        self.constraints = self.constraints.loc[after_begin & before_end].copy()

        ## filter by target zone
        geo_levels_tz = geo_levels[self.target_zone]
        self.constraints = self.constraints[
            self.constraints.geo_base_level.isin(geo_levels_tz)
        ]

        ## Sanity check --------------------------------------------------------
        ## ensure total population and total households present
        _constr_msg_pre = "Check the ``puma.{}`` attribute. "
        _constr_msg_post = "See ``livelike.config`` for reference."

        #   - in self.constraints.constraint
        if not (
            any(self.constraints.constraint == "population")
            & any(self.constraints.constraint == "housing_units")
            & any(self.constraints.constraint == "group_quarters_pop")
        ):
            raise ValueError(
                f"{_constr_msg_pre.format('constraints.constraint')}"
                "Constraints must include population totals "
                "(``population``, ``housing_units``, and ``group_quarters_pop``). "
                f"{_constr_msg_post}"
            )

        #   - in self.constraints_selection
        if self.constraints_selection is not None:
            if "universe" not in self.constraints_selection:
                raise ValueError(
                    f"{_constr_msg_pre.format('constraints_selection')}"
                    "Constraints selection must include ``universe`` key. "
                    f"{_constr_msg_post}"
                )
            else:
                uval = self.constraints_selection["universe"]
                _uval_msg = "The value of ``universe`` must"
                if uval is False:
                    raise ValueError(
                        f"{_constr_msg_pre.format('constraints_selection')}"
                        f"{_uval_msg} not be ``False``. "
                        f"{_constr_msg_post}"
                    )
                else:
                    if uval is True:
                        pass
                    elif not isinstance(uval, list):
                        raise TypeError(
                            f"{_constr_msg_pre.format('constraints_selection')}"
                            f"{_uval_msg} be a ``list``. "
                            f"{_constr_msg_post}"
                        )
                    else:
                        cstrs = ["population", "group_quarters_pop", "housing_units"]
                        if not all(cstr in uval for cstr in cstrs):
                            raise ValueError(
                                f"{_constr_msg_pre.format('constraints_selection')}"
                                f"{_uval_msg} include {cstrs}. "
                                f"{_constr_msg_post}"
                            )
            #   - ensure no False
            for k, v in self.constraints_selection.items():
                if isinstance(v, bool) and not (v):
                    raise ValueError(
                        f"The '{k}' constraint selection was passed in as '{v}', "
                        "but must be passed in as either 'True' or a list."
                    )

        #   - in self.constraints_theme_order
        if (
            self.constraints_theme_order is not None
            and "universe" not in self.constraints_theme_order
        ):
            raise ValueError(
                f"{_constr_msg_pre.format('constraints_theme_order')}"
                "Theme order must include ``universe``. "
                f"{_constr_msg_post}"
            )

        # ----------------------------------------------------------------------

        ## select constraints
        if self.constraints_selection is None:
            # No selection provided
            # Use base constraints with default/prebuilt constraints
            # Use all available constraints with custom constraints
            if self.using_default_constraints:
                self.constraints_selection = up_base_constraints_selection

            else:
                constraints_themes = pd.unique(self.constraints.theme)
                self.constraints_selection = dict(
                    zip(
                        constraints_themes,
                        [True] * len(constraints_themes),
                        strict=True,
                    )
                )

        self.constraints = select_constraints(
            self.constraints, self.constraints_selection
        )

        ## Replace any instances of "household-person" level with "person"
        self.constraints.level = self.constraints.level.replace(
            "household-person", "person"
        )

        ## ensure consistent constraint order
        ## by theme and ACS variable code
        if self.verbose:
            print(f"constraints : {self.constraints}\n")

        if self.constraints_theme_order is None:
            if self.using_default_constraints:
                self.constraints_theme_order = up_constraints_theme_order

            else:
                self.constraints_theme_order = pd.unique(self.constraints.theme)

        if self.verbose:
            print(f"constraints theme order : {self.constraints_theme_order}\n")

        self.constraints.theme = pd.Categorical(
            self.constraints.theme, categories=self.constraints_theme_order
        )

        if self.verbose:
            print(f"constraints.theme : {self.constraints.theme}\n")

        self.constraints = self.constraints.sort_values(["theme", "code"])

        return self

    def build_individual_constraints(self):
        """Extract initial P-MEDM constraints."""
        cind0 = build_constraints_ind(
            self.fips,
            self.constraints,
            year=self.year,
            cache=self.cache,
            replicate=self.replicate,
            keep_intermediates=self.keep_intermediates,
            censusapikey=self.censusapikey,
            verbose=self.verbose,
            cache_folder=self.cache_folder,
        )

        self.est_ind = cind0["cst"]
        self.wt = cind0["wt"]
        self.sporder = cind0["sporder"]

        # do not allow negative weights
        # P-MEDM compat for replicates with negative weights
        self.wt[self.wt < 0] = 0

        # preserve person and household intermediate constraints
        if self.keep_intermediates:
            self.est_household = cind0["chous"]
            self.est_person = cind0["cpers"]

        else:
            self.est_household = None
            self.est_person = None

        del cind0

        return self

    def build_tract_input(self):
        """Build tract input. Either for the source or target units."""

        # build tract inputs
        self.v_fmt = [format_acs_code(i, "E") for i in self.constraints.code]
        self.v_fmt_moe = [format_acs_code(i, "M") for i in self.constraints.code]

        self.ext_trt = build_acs_sf_inputs(
            v=self.v_fmt,
            level="trt",
            fips=self.all_trt_geoids,
            puma=self.fips,
            year=self.year,
            cache=self.cache,
            tag="est",
            censusapikey=self.censusapikey,
            verbose=self.verbose,
            cache_folder=self.cache_folder,
        )

        # remove unpopulated tract FIPS codes
        self.pop_trt_geoids = self.ext_trt.index.values

        self.ext_trt_moe = build_acs_sf_inputs(
            v=self.v_fmt_moe,
            level="trt",
            fips=self.pop_trt_geoids,
            puma=self.fips,
            year=self.year,
            cache=self.cache,
            populated=False,
            tag="moe",
            censusapikey=self.censusapikey,
            cache_folder=self.cache_folder,
        )
        self.ext_trt_se = self.ext_trt_moe / 1.645

        return self

    def build_block_group_input(self):
        """Build block group input"""

        self.ext_bg = build_acs_sf_inputs(
            v=self.v_fmt,
            level="bg",
            fips=self.pop_trt_geoids,
            puma=self.fips,
            year=self.year,
            cache=self.cache,
            tag="est",
            censusapikey=self.censusapikey,
            cache_folder=self.cache_folder,
        )

        self.ext_bg_moe = build_acs_sf_inputs(
            v=self.v_fmt_moe,
            level="bg",
            fips=self.pop_trt_geoids,
            puma=self.fips,
            year=self.year,
            cache=self.cache,
            tag="moe",
            populated=False,
            censusapikey=self.censusapikey,
            cache_folder=self.cache_folder,
        )

        populated_ext_bg_moe = self.ext_bg_moe.index.isin(self.ext_bg.index)
        self.ext_bg_moe = self.ext_bg_moe[populated_ext_bg_moe]
        self.ext_bg_se = self.ext_bg_moe / 1.645

        # sort by geoid
        self.ext_trt = self.ext_trt.sort_index()
        self.ext_bg = self.ext_bg.sort_index()

        self.est_g1 = self.ext_trt
        self.moe_g1 = self.ext_trt_moe
        self.se_g1 = self.ext_trt_se

        self.est_g2 = self.ext_bg
        self.moe_g2 = self.ext_bg_moe
        self.se_g2 = self.ext_bg_se

        self.topo = None  # TEMP
        self.geo = None  # TEMP
        self.g1 = None
        self.g2 = None

        return self

    def build_tract_geography(self):
        """Build tract geography."""

        self.geo_extract = extract_geographies(
            year=self.year,
            geo="trt",
            targets=self.ext_trt.index,
            cache=self.cache,
            puma=self.fips,
            censusapikey=self.censusapikey,
            verbose=self.verbose,
            cache_folder=self.cache_folder,
        )

        # sort index & remove unpopulated tracts
        self.geo_extract = self.geo_extract.sort_index().pipe(
            lambda df: df[df.index.isin(self.ext_trt.index)]
        )

        return self

    def build_super_tracts(self):
        """Build super tracts."""

        assert self.make_trt_geo, "Tract geometries are required."

        # assign labels and reorder columns
        # TODO: implement cache
        super_trt_labels = make_supertracts(
            geo=self.geo_extract,
            sf_data=self.ext_trt,
            random_state=self.random_state,
            method=self.make_super_trt_method,
        )
        self.geo_extract["super_trt"] = super_trt_labels
        cols = list(self.geo_extract.columns.copy())
        geocol, suptrt = cols[-2:]
        cols[-1] = geocol
        cols[-2] = suptrt
        self.geo_extract = self.geo_extract[cols].copy()

        # aggregate ests, standard errors, and MOEs
        agg_super_trt = aggregate_acs_sf_supertracts(
            self.ext_trt, self.ext_trt_se, super_trt_labels
        )

        topo = self.ext_trt.merge(self.geo_extract.loc[:, ["super_trt"]], on="GEOID")
        topo = topo["super_trt"].sort_values()
        topo = topo.reset_index()
        topo.columns = ["g2", "g1"]

        # TODO: remove `g1` and `g2` and just use `topo`
        self.g1 = pd.Series(topo["g1"])
        self.g2 = pd.Series(topo["g2"])

        self.topo = topo

        self.est_g1 = agg_super_trt["est"]
        self.moe_g1 = agg_super_trt["moe"]
        self.se_g1 = agg_super_trt["se"]

        reorder_g2 = topo["g2"].values
        self.est_g2 = self.ext_trt.reindex(reorder_g2)
        self.moe_g2 = self.ext_trt_moe.reindex(reorder_g2)
        self.se_g2 = self.ext_trt_se.reindex(reorder_g2)

        self.geo = self.geo_extract if self.keep_geo else None

        del self.geo_extract

        return self

    def bypass_super_tracts(self):
        """Bypass super tracts."""

        self.g1 = None
        self.g2 = None
        self.topo = None

        # placeholder for g1
        self.est_g1 = pd.DataFrame(columns=self.ext_trt.columns)
        self.moe_g1 = pd.DataFrame(columns=self.ext_trt_moe.columns)
        self.se_g1 = pd.DataFrame(columns=self.ext_trt_se.columns)

        self.est_g2 = self.ext_trt
        self.moe_g2 = self.ext_trt_moe
        self.se_g2 = self.ext_trt_se

        self.geo = self.geo_extract if hasattr(self, "geo_extract") else None

        return self

    def validate_columns(self):
        """Rename columns & validate value equivalency."""
        # rename geo columns
        for i in ["est_g1", "moe_g1", "se_g1", "est_g2", "moe_g2", "se_g2"]:
            getattr(self, i).columns = self.constraints.constraint

        # ensure individual/geo columns match
        assert all(self.est_ind.columns == self.est_g1.columns), (
            "Individual-level constraints do not match geographic constraints."
        )

        return self

    def estimate_group_quarters(self):
        # Estimate group quarters for 2023+
        # when block groups (``bg``) are the target geometry
        pop_in_households = build_acs_sf_inputs(
            v=["B11002_001E"],
            fips=self.est_g1.index.tolist(),
            puma=self.fips,
            populated=True,
            year=2023,
            level="bg",
            censusapikey=self.censusapikey,
        )
        gq_pop = pd.concat(
            [self.est_g2.population, -pop_in_households.B11002_001E], axis=1
        ).apply(sum, axis=1)

        # Approximate standard errors #
        rpop = get_vre_tables(
            fips=self.fips,
            year=self.year,
            table="B01001",
            cache=self.cache,
            cache_folder=self.cache_folder,
            verbose=self.verbose,
        )
        rpoh = get_vre_tables(
            fips=self.fips,
            year=self.year,
            table="B11002",
            cache=self.cache,
            cache_folder=self.cache_folder,
            verbose=self.verbose,
        )

        geoids = self.est_g2.index.values
        gq_pop_se = []
        for g in geoids:
            rp = rpop.loc[g]
            rph = rpoh.loc[g]

            gq_est = gq_pop.loc[g]
            gq_rep_est = rp - rph
            gq_se_est = np.sqrt(4 / 80 * np.sum((gq_rep_est - gq_est) ** 2))
            gq_pop_se += [gq_se_est]

        gq_pop_se = pd.Series(gq_pop_se, index=geoids)

        # replace zero SEs with ACS default value
        # TODO: more robust way to handle this?
        zero_se = self.se_g2.values[np.where(self.est_g2 == 0)].min()
        gq_pop_se.loc[gq_pop_se == 0] = zero_se

        self.est_g2.group_quarters_pop = gq_pop
        self.se_g2.group_quarters_pop = gq_pop_se

        return self


def select_constraints(constraints_: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """Selects P-MEDM constraints.

    Parameters
    ----------
    constraints_ : pandas.DataFrame (config.constraints)
        A DataFrame of P-MEDM constraints.
    selection : dict (config.up_base_constraints_selection)
         P-MEDM constraints to use. Key-value pairs with keys representing
        ACS variable themes and values representing specific subjects (tables).

        If the value passed is a ``bool`` type, a ``True`` value will include
        variables for all subjects in the theme, while a `False` value will
        bypass that theme (the same as omitting the theme from the selection).

        If the value passed is a ``list`` type, only listed subjects will be
        included in the result.

    Returns
    -------
    cst : pandas.DataFrame
        Selected P-MEDM constraints.
    """

    cst = pd.DataFrame()

    for k, v in selection.items():
        if not isinstance(v, (bool | list)):
            raise TypeError(f"Values of '{k}' must be ``bool`` or ``list`` type.")

        if isinstance(v, bool):
            if v:
                cst_ = constraints_[constraints_.theme == k]

        else:
            cst_ = constraints_[constraints_.subject.isin(v)]

        cst = pd.concat([cst, cst_], axis=0)

    return cst


def format_acs_code(x: str, var_type: str = "E") -> str:
    """Return formatted column IDs to match census API (ACS SF).

    Parameters
    ----------
    x : str
        Column ID.
    var_type : str = 'E'
        ACS variable type (Estimate (``'E'``) or Margin of Error (``'M'``)).

    Returns
    -------
    str
        Formatted column ID.
    """

    return x[0:-3] + "_" + x[-3:] + var_type


def parse_pums_variables(
    constraints_: pd.DataFrame,
    level: str,
    replicate: None | int = None,
) -> np.ndarray:
    """Parses PUMS variable names from P-MEDM target constraints.

    Parameters
    ----------
    constraints_ : pandas.DataFrame
        P-MEDM constraints.
    level : str
        PUMS level (``'person'`` or ``'household'``).
    replicate : None | int = None
        Replicate number.

    Returns
    -------
    pv : numpy.ndarray
        PUMS variable names.
    """

    cx = constraints_[constraints_["level"] == level]

    pums_cols = cx.columns[cx.columns.str.match("^pums")]
    pcx = cx.loc[:, pums_cols].fillna("")
    pv = np.unique(pcx.values.flatten())
    pv = pv[pv != ""]

    if replicate is not None:
        rep_str = [str(i) for i in range(1, replicate + 1)]
        rwt = "WGTP"
        if level == "person":
            rwt = "P" + rwt
        repwts = np.array([rwt + i for i in rep_str])
        pv = np.append(pv, repwts)

    return pv


def build_census_microdata_api_base_request(level: str, year: int = 2019) -> str:
    """Helper function for building base Census Microdata API request.

    Parameters
    ----------
    level : str
        PUMS level (``'person'`` or ``'household'``).
    year : int = 2019
        ACS 5-Year Estimates vintage.

    Returns
    -------
    request : str
        The base API request.
    """

    # base query
    request = "https://api.census.gov/data/" + str(year) + "/acs/acs5/pums?get="

    ## Build PUMS variable query
    pv0 = np.array(["SERIALNO"])

    # append household order if request is person-level
    if level == "person":
        pv0 = np.append(pv0, ["SPORDER"])
        pv0 = ",".join(pv0)
    else:
        # household level and only one var
        pv0 = pv0[0]

    request += pv0

    return request


def build_census_microdata_api_geo_request(fips: str, year: int = 2019) -> str:
    """Helper function for building Census Microdata API geo requests.

    Parameters
    ----------
    fips : str
        7-digit PUMA FIPS code (state + puma).
    year : int = 2019
        ACS 5-Year Estimates vintage.

    Returns
    -------
    georq : str
        The geo request.
    """

    # append geographic identifiers
    st = fips[:2]
    pu = str(int(fips[2:]))

    if 2016 <= year <= 2019:
        georq = f"&for=public%20use%20microdata%20area:{pu}&in=state:{st}"
    elif year >= 2023:
        georq = f"&for=state:{st}&PUMA={pu}"
    else:
        raise ValueError("Supported years are 2016 - 2019 and 2023+.")

    return georq


def extract_from_census_microdata_api(
    request: str,
    censusapikey: None | str = None,
    drop_geoids: bool = True,
) -> pd.DataFrame:
    """Performs a Census Microdata API request and converts the
    result to a ``pandas.DataFrame``.

    Parameters
    ----------
    request : str
        API request.
    censusapikey : None | str = None
        Optional Census API key.
    drop_geoids : bool = True
        Whether or not to drop state and PUMA identifiers.

    Returns
    -------
    extract : pandas.DataFrame
        The Census Microdata API extract.
    """

    # use census API key if given
    if censusapikey is not None:
        request += "&key=" + censusapikey
    
    # JSON to dataframe
    extract = pd.read_json(request)

    # fix header
    cols = extract.iloc[0].values
    extract = extract.iloc[1:]
    extract.columns = cols

    # drop geographic identifiers as needed
    if drop_geoids:
        year = int(request.split("/")[4])
        if 2016 <= year <= 2019:
            extract = extract.drop(["state", "public use microdata area"], axis=1)
        elif year >= 2023:
            extract = extract.drop(["state", "PUMA"], axis=1)
        else:
            raise ValueError("This should never happen. Something seriously wrong.")

    return extract


def update_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to update the data types of a PUMS
    extract to enable reclassification and tabulation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame consisting of PUMS variables.

    Returns
    -------
    df : pandas.DataFrame
        Properly typed ``df``.
    """

    df_dtypes = {k: v for k, v in pums_dtypes.items() if k in df.columns}

    return df.astype(df_dtypes)


def extract_pums_constraints(
    fips: str,
    level: str,
    constraints_: pd.DataFrame,
    year: int = 2019,
    replicate: int = None,
    censusapikey: None | str = None,
) -> pd.DataFrame:
    """Extracts PUMS variables for preparing P-MEDM constraints at the
    person or household levels via the Census Microdata API.

    Parameters
    ----------
    fips : str
        7-digit PUMA FIPS code (state + puma).
    level : str
        PUMS level (``'person'`` or ``'household'``).
    constraints_ : pandas.DataFrame
        P-MEDM constraints.
    year : int = 2019
        ACS 5-Year Estimates vintage.
    replicate : int = None
        Replicate number.
    censusapikey : None | str = None
        Optional Census API key. The default is ``None``

    Returns
    -------
    extract : pandas.DataFrame
        PUMS file consisting of variables needed to construct the P-MEDM constraints.
    """

    # base request
    request = build_census_microdata_api_base_request(level, year=year)

    # required pums constraints
    pvr = parse_pums_variables(constraints_, level, replicate=replicate)

    # base variables
    pv0 = np.array([], dtype="str")

    # append income adjustment factor if income vars required
    if any([i in pums_income_vars for i in pvr]):  # noqa C419
        pv0 = np.append(pv0, ["ADJINC"])

    # append vacancy status if request is household-level
    if level == "household":
        pv0 = np.append(pv0, ["VACS"])

    # append required variables
    pv0 = np.append(pv0, pvr)
    npart = int(np.ceil(len(pv0) / cmd_api_request_limit))
    pv0 = np.array_split(pv0, npart)

    extract = []
    sort_cols = ["SERIALNO"]
    if level == "person":
        sort_cols += ["SPORDER"]
    for v_ in pv0:
        pv_ = ",".join(v_)
        req_ = request + "," + pv_

        # append geographic identifiers
        req_ += build_census_microdata_api_geo_request(fips, year=year)

        # create extract
        ext_ = extract_from_census_microdata_api(req_, censusapikey=censusapikey)

        # update data types
        ext_ = update_data_types(ext_)

        ext_ = ext_.set_index(sort_cols)

        extract += [ext_]

    extract = pd.concat(extract, axis=1)

    ########################################################################
    # # ensure responses are ordered by serial/sporder
    # if level == "person":
    #     extract = extract.sort_values(by=["SERIALNO", "SPORDER"])

    # else:  # household
    #     extract = extract.sort_values(by=["SERIALNO"])
    extract = extract.sort_index()
    ########################################################################

    # reset index
    extract = extract.reset_index(drop=False)

    return extract


def extract_pums_segment_ids(
    fips: str,
    level: str,
    query: str,
    year: int = 2019,
    censusapikey: None | str = None,
) -> pd.DataFrame:
    """Extracts PUMS records matching a segment described by an API query.

    Parameters
    ----------
    fips : str
        7-digit PUMA FIPS code (state + puma).
    level : str
        PUMS level (``'person'`` or ``'household'``).
    query : str
        Query containing segment characteristics.
    year : int = 2019
        ACS 5-Year Estimates vintage.
    censusapikey : None | str = None
        Optional Census API key.

    Returns
    -------
    extract : pandas.DataFrame
        A DataFrame containing household identifiers (SERIALNO) and
        household structure if a person-level request (SPORDER).
    """

    levels = np.array(["person", "household"])
    assert any(levels == level), (
        "Argument ``level`` must be one of: ``person``, ``household``."
    )

    # base request
    request = build_census_microdata_api_base_request(level, year=year)

    # append geographic identifiers
    request += build_census_microdata_api_geo_request(fips, year=year)

    # append query
    request += "&" + query

    # create extract
    extract = extract_from_census_microdata_api(request, censusapikey=censusapikey)

    # if person-level, append person ID
    if level == "person":
        extract["p_id"] = extract["SERIALNO"] + extract["SPORDER"].str.zfill(2)

    return extract


def extract_pums_descriptors(
    fips: str,
    level: str,
    features: np.typing.ArrayLike,
    year: int = 2019,
    sample_weights: bool = False,
    censusapikey: None | str = None,
) -> pd.DataFrame:
    """Extracts descriptive features from the PUMS.

    Parameters
    ----------
    fips : str
        7-digit PUMA FIPS code (state + puma).
    level : str
        PUMS level (``'person'`` or ``'household'``).
    features : array-like
        Target feature names.
    sample_weights : bool = False
        Whether or not to append PUMS sample weights.
    year : int = 2019
        ACS 5-Year Estimates vintage (default ``2019``).
    censusapikey : None | str = None
        Optional Census API key.

    Returns
    -------
    extract : pandas.DataFrame
        A DataFrame containing household identifiers (SERIALNO) and
        household structure if a person-level request (SPORDER).
    """

    levels = np.array(["person", "household"])
    assert any(levels == level), (
        "Argument ``level`` must be one of: ``person``, ``household``."
    )

    if isinstance(features, np.ndarray):
        features = np.array(features, dtype="str")

    if sample_weights:
        if level == "person":
            features = np.append(np.array(["PWGTP"]), features)

        else:  # household
            features = np.append(np.array(["WGTP"]), features)

    # append income adjustment factor if income vars required
    if any([i in pums_income_vars for i in features]):  # noqa C419
        features = np.append(features, ["ADJINC"])

    features = ",".join(features)

    # base request
    request = build_census_microdata_api_base_request(level, year=year)

    # append features
    request += "," + features

    # append geographic identifiers
    request += build_census_microdata_api_geo_request(fips, year=year)

    # create extract
    extract = extract_from_census_microdata_api(request, censusapikey=censusapikey)

    # if person-level, append person ID
    if level == "person":
        extract["p_id"] = extract["SERIALNO"] + extract["SPORDER"].str.zfill(2)

    # update data types
    extract = update_data_types(extract)

    return extract


def build_acs_pums_inputs(
    fips: str,
    level: str,
    constraints_: pd.DataFrame,
    cache: bool = False,
    year: int = 2019,
    replicate: None | int = None,
    censusapikey: None | str = None,
    verbose: bool = False,
    cache_folder: str | pathlib.Path = "./livelike_acs_cache",
) -> dict[str, pd.DataFrame]:
    """Extracts the raw PUMS file and creates initial P-MEDM constraints from a
    table of study variables and household and person-level microdata (PUMS) files.

    Parameters
    ----------
    fips : str
        7-digit PUMA FIPS code (state + puma).
    level : str
        PUMS level (``'person'`` or ``'household'``).
    constraints_ : pandas.DataFrame
        P-MEDM constraints.
    cache : bool
        Whether to cache the constraints file, or read
        an existing constraints file from cache.
    year : int = 2019
        ACS 5-Year Estimates vintage.
    replicate : None | int = None
        Replicate number.
    censusapikey : None | str = None
        Optional Census API key.
    verbose : bool = False
        Whether to print status messages.
    cache_folder : str | pathlib.Path = './livelike_acs_cache'
        The cache folder to store queried ACS inputs. This default value
        creates the folder in the current working directory if not present.

    Returns
    -------
    out : dict[str, pandas.DataFrame]
        A dictionary containing the raw PUMS file extract (``'pums'``),
        and the initial constraints (``'constraints'``).
    """

    # validate cache input and create directory if it doesn't exist
    if cache:
        cache_folder = _ensure_cache_folder_input(cache_folder)

        # check for cached constraints
        cache_cst_path = cache_folder / f"constraints_{fips}_{level}.parquet"
        has_cached_cst_file = cache_cst_path.exists() and cache_cst_path.is_file()
    else:
        has_cached_cst_file = False

    if verbose:
        print(f"Cached {level} constraints file exists: {has_cached_cst_file}")

    if has_cached_cst_file and cache:
        if verbose:
            print(f"Using cached {level} constraints file: {cache_cst_path}")
        ext = pd.read_parquet(cache_cst_path)
    else:
        if verbose:
            print(f"Extracting {level} constraints from Census Microdata API.")
        ext = extract_pums_constraints(
            fips,
            level,
            constraints_,
            year=year,
            replicate=replicate,
            censusapikey=censusapikey,
        )
        if cache:
            if verbose:
                print(
                    f"Writing cached copy of {level} constraints at {cache_cst_path}."
                )
            ext.to_parquet(cache_cst_path)
    if verbose:
        print("\n")

    # target variables
    vx_names = pd.unique(constraints_[constraints_["level"] == level]["subject"])

    # constraints container
    cx = pd.DataFrame()

    if replicate is not None:
        # subset columns to current replicate weights
        ext_sub = ext.columns[
            (~ext.columns.str.contains("WGTP"))
            | (
                (ext.columns.str.contains("WGTP"))
                & (ext.columns.str.contains(f"{replicate}$"))
            )
        ]
        ext = ext.loc[:, ext_sub]

    for v in vx_names:
        # declare the function being sourced from ``pums.py``
        pums_func = getattr(pums, v)

        # run ``pums_func()`` for granular representation of variable values
        vx_repr = (
            pums_func(ext, year) if pums_func.__name__ in need_year else pums_func(ext)
        )

        # concatenate reseults to constraints container
        cx = pd.concat([cx, vx_repr], axis=1)

    cx.index = ext["SERIALNO"].astype("str")

    out = {"pums": ext, "constraints": cx}

    return out


def build_constraints_ind(
    fips: str,
    constraints_: pd.DataFrame,
    cache: bool = False,
    year: int = 2019,
    replicate: None | int = None,
    keep_intermediates: bool = False,
    censusapikey: None | str = None,
    verbose: bool = False,
    cache_folder: str | pathlib.Path = "./livelike_acs_cache",
) -> dict[str, pd.DataFrame]:
    """Extracts initial P-MEDM constraints at the household and
    person levels,  then merges them into a table of input
    household-level constraints for the P-MEDM problem.

    Parameters
    ----------
    fips : str
        7-digit PUMA FIPS code (state + puma).
    constraints_ : pandas.DataFrame
        P-MEDM constraints.
    year : int = 2019
        ACS 5-Year Estimates vintage (default 2019).
    replicate :  None | int = None
        Replicate number.
    keep_intermediates : bool = False
        Should intermediate P-MEDM household and person-level constraints
        (pre-aggregation) be kept? Default ``False``.
    censusapikey : None | str = None
        Optional Census API key.
    verbose : bool = False
        Whether to print status messages.
    cache_folder : str | pathlib.Path = './livelike_acs_cache'
        The cache folder to store queried ACS inputs. This default value
        creates the folder in the current working directory if not present.

    Returns
    -------
    constrs : dict[str, pandas.DataFrame]
        A dictionary containing the input constraints (``'cst'``), sample weights
        (``'wt'``), and person order within household (``'sporder'``).
    """

    ## household-level constraints
    in_hous = build_acs_pums_inputs(
        fips,
        "household",
        constraints_,
        cache=cache,
        year=year,
        replicate=replicate,
        censusapikey=censusapikey,
        verbose=verbose,
        cache_folder=cache_folder,
    )

    ## person-level constraints
    in_pers = build_acs_pums_inputs(
        fips,
        "person",
        constraints_,
        cache=cache,
        year=year,
        replicate=replicate,
        censusapikey=censusapikey,
        verbose=verbose,
        cache_folder=cache_folder,
    )

    # chous = in_hous["constraints"].reset_index().drop_duplicates()
    # chous = chous.set_index("SERIALNO")
    chous = in_hous["constraints"]
    cpers0_ = in_pers["constraints"]

    # multiply remainder of constraints by total population
    # constraint so that values are representative
    for c in cpers0_.columns:
        if c != "population":
            cpers0_[c] = cpers0_[c] * cpers0_["population"]

    # aggregate to match household level
    cpers = cpers0_.groupby(cpers0_.index).sum()

    ## final individual constraints
    cind = chous.merge(cpers, left_index=True, right_index=True, how="outer")

    # fill in remaining missing/blank values for vacant hh's
    cind = cind.fillna(0)

    ## ensure constraint level order is preserved
    ## and unused variables are dropped
    col_match = cind.columns.get_indexer(constraints_.constraint)
    cind = cind.loc[:, [cind.columns[i] for i in col_match if i != -1]]

    ## sample weights
    # for group quarters (`WGTP == 0``), we want to use `PWGTP` in place
    # of `WGTP` as there are expected to be n GQ residences linked to GQ persons
    r = str(replicate) if replicate is not None else ""

    wgtp_ = in_hous["pums"].loc[:, ["SERIALNO", f"WGTP{r}"]]
    wgtp_ = wgtp_.set_index("SERIALNO")

    gqs = cind.index[cind.group_quarters_pop > 0]

    rep_prs = in_pers["pums"][in_pers["pums"]["SPORDER"] == 1]
    wgtp_ = pd.concat([wgtp_, rep_prs.set_index("SERIALNO")[f"PWGTP{r}"]], axis=1)

    wgtp = np.where(
        wgtp_.index.isin(gqs),
        wgtp_.loc[:, f"PWGTP{r}"].values,
        wgtp_.loc[:, f"WGTP{r}"].values,
    )

    ## person order in household
    spo = in_pers["pums"].loc[:, ["SERIALNO", "SPORDER"]]
    spo.index = spo["SERIALNO"]
    spo = spo.drop("SERIALNO", axis=1)

    constrs = {"cst": cind, "wt": wgtp, "sporder": spo}

    if keep_intermediates:
        constrs.update({"chous": chous, "cpers": pd.concat([spo, cpers0_], axis=1)})

    return constrs


def build_fips_tree(fips: list[str]) -> dict[str, dict[str, list[str]]]:
    """
    Partitions FIPS codes (tract level or lower) as states : counties : census tracts.

    Parameters
    ----------
    fips : list[str]
        FIPS codes. Must be at least census tract level (11 characters or more).

    Returns
    -------
    ft : dict[str, dict[str, list[str]]]
        A nested dict with outer keys state FIPS codes, inner keys county
        FIPS codes, and inner values lists of tract FIPS codes.
    """

    fips_check = [bool(len(g) >= 11) for g in fips]
    if not all(fips_check):
        raise ValueError("All FIPS codes must be 11 characters or more.")

    # convert to tract IDs as needed
    fips = [g[11:] if len(g) > 11 else g for g in fips]

    # split FIPS codes by state/county/tract index
    ix = [0, 2, 5]
    tg = np.array(
        [[g[i:j] for i, j in zip(ix, ix[1:] + [None], strict=True)] for g in fips]
    )

    # tracts within counties
    cty_trt = {}
    for k, v in tg[:, 1:]:
        cty_trt.setdefault(k, []).append(v)

    # counties:tracts within states
    st_cty = np.unique(tg[:, :-1], axis=0)
    ft = {}
    for k, v in st_cty:
        if k in ft:
            ft[k].update({v: cty_trt[v]})

        else:
            ft[k] = {v: cty_trt[v]}

    return ft


def build_acs_sf_inputs(
    v: list[str],
    fips: list[str],
    puma: None | str = None,
    year: int = 2019,
    level: str = "trt",
    populated: bool = True,
    which_pop: int = 0,
    cache: bool = False,
    tag: None | str = None,
    censusapikey: None | str = None,
    verbose=False,
    cache_folder: str | pathlib.Path = "./livelike_acs_cache",
) -> pd.DataFrame:
    """Builds geographic constraints based on the ACS Summary File (SF)
    as inputs for P-MEDM.

    Parameters
    ----------
    v : list[str]
        ACS variable codes.
    fips : list[str]
       Target FIPS codes.
    puma : None | str = None
        7-digit PUMA FIPS code (state + puma).
    year : int = 2019
        Target year.
    level : str = 'trt'
        Target geography (one of block group ``bg`` or tract ``trt``).
    populated : bool = True
        Flag to keep populated units only (default ``True``).
    which_pop : int = 0
        Column index of the total population variable (default ``0``).
    cache : bool = False
        Whether to cache ACS SF data.
    tag : None | str = None
        Optional naming tag for cached data, appended to the end of the filename.
    censusapikey : None | str = None
        Optional Census API key.
    verbose : bool = False
        Whether to print status messages.
    cache_folder : str | pathlib.Path = './livelike_acs_cache'
        The cache folder to store queried ACS inputs. This default value
        creates the folder in the current working directory if not present.

    Returns
    -------
    ext : pandas.DataFrame
        ACS SF variable estimates for the target FIPS codes,
        indexed by FIPS code (``geoid``).
    """

    if level not in ["trt", "bg"]:
        raise ValueError("Argument ``level`` must be one of: ``tract``, ``bg``.")

    puma = f"_{puma}" if puma is not None else ""
    tag = f"_{tag}" if tag is not None else ""

    # validate cache input and create directory if it doesn't exist
    if cache:
        cache_folder = _ensure_cache_folder_input(cache_folder)

        # check for cached constraints
        cache_cst_path = cache_folder / f"constraints{puma}_{level}{tag}.parquet"
        has_cached_cst_file = cache_cst_path.exists() and cache_cst_path.is_file()
    else:
        has_cached_cst_file = False

    if verbose:
        print(f"Cached {level} constraints file exists: {has_cached_cst_file}")

    if has_cached_cst_file and cache:
        if verbose:
            print(f"Using cached {level}{tag} constraints file: {cache_cst_path}")
        ext = pd.read_parquet(cache_cst_path)
    else:
        if verbose:
            print(f"Extracting {level}{tag} constraints using pygris.")

        ext = pd.DataFrame()
        ft = build_fips_tree(fips)

        for st in ft:
            for cty in ft[st]:
                trt_join = ",".join(ft[st][cty])

                if level == "trt":
                    p_for = f"tract:{trt_join}"
                    p_in = f"state:{st}+county:{cty}"

                else:  # bg
                    p_for = "block group:*"
                    p_in = f"state:{st}+county:{cty}+tract:{trt_join}"

                params = {"for": p_for, "in": p_in}

                if censusapikey is not None:
                    params.update({"key": censusapikey})

                ext_part = get_census(
                    dataset="acs/acs5",
                    variables=v,
                    params=params,
                    year=year,
                    return_geoid=True,
                    guess_dtypes=True,
                ).set_index("GEOID")

                ext = pd.concat([ext, ext_part], axis=0)
        ext.index = ext.index.astype("str")
        ext = ext.sort_index()

        # remove unpopulated tracts
        if populated:
            ext = ext[ext.iloc[:, which_pop] > 0]

        if cache:
            if verbose:
                print(
                    f"Writing cached copy of {level}{tag} "
                    f"constraints at {cache_cst_path}."
                )
            ext.to_parquet(cache_cst_path)
    if verbose:
        print("\n")

    return ext


def parse_bg_fips_by_tract(
    year: int, targets: list[str], censusapikey: None | str = None
) -> np.ndarray:
    """
    Parses block group FIPS codes from tract FIPS codes
    using TIGER Web Mapping Service (WMS).

    Parameters
    ----------
    year : int
        Target year.
    targets : list[int]
        A list of target tract FIPS codes.
    censusapikey : None | str = None
        Optional Census API key.

    Returns
    -------
    geoids : numpy.ndarray
        Block group FIPS codes.
    """

    base = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        f"tigerWMS_ACS{year}/MapServer/10/query?"
    )

    gsearch = " OR GEOID LIKE".join([f"'{g}%'" for g in targets])
    params_ = {"where": f"GEOID LIKE {gsearch}"}

    params_.update({"outfields": "GEOID", "returnGeometry": "false", "f": "pjson"})
    if censusapikey is not None:
        params_.update({"key": censusapikey})

    params = urllib.parse.urlencode(params_)
    url = base + params

    rj = requests.get(url).json()
    geoids = pd.json_normalize(rj["features"]).values.flatten()

    return geoids


def extract_geographies(
    year: int,
    targets: np.typing.ArrayLike,
    geo: str = "bg",
    cache: bool = False,
    puma: None | str = None,
    censusapikey: None | str = None,
    verbose: bool = False,
    cache_folder: str | pathlib.Path = "./livelike_acs_cache",
) -> gpd.GeoDataFrame:
    """
    Extracts ACS geographies from the Tiger Web Mapping Service (WMS).
    Currently supports block groups and tracts.

    Parameters
    ----------
    year : int
        Target year.
    targets : array-like
        Target FIPS codes.
    geo : str = 'bg'
        Target geography (``bg`` for block groups, ``trt`` for tracts).
    cache : bool = False
        Whether to cache geographies.
    puma : None | str = None
        7-digit PUMA FIPS code (state + puma).
    censusapikey: None | str = None
        Optional Census API key.
    verbose : bool = False
        Whether to print status messages.
    cache_folder : str | pathlib.Path = './livelike_acs_cache'
        The cache folder to store queried geographies. This default value
        creates the folder in the current working directory if not present.

    Returns
    -------
    ext : geopandas.GeoDataFrame
          Target zones.
    """

    if geo not in ["bg", "trt"]:
        raise ValueError("Target zone type (``geo``) must be one of ``bg``, ``trt``.")

    puma = f"{puma}_" if puma is not None else ""

    # validate cache input and create directory if it doesn't exist
    if cache:
        cache_folder = _ensure_cache_folder_input(cache_folder)

        # check for cached constraints
        cache_geo_path = cache_folder / f"boundaries_{puma}{geo}.gpkg"
        has_cached_geo_file = cache_geo_path.exists() and cache_geo_path.is_file()
    else:
        has_cached_geo_file = False

    if verbose:
        print(f"Cached {puma}{geo} boundaries file exists: {has_cached_geo_file}")

    if has_cached_geo_file and cache:
        if verbose:
            print(f"Using cached {puma}{geo} boundaries file: {cache_geo_path}")
        ext = gpd.read_file(cache_geo_path).set_index("GEOID")

    else:
        if verbose:
            print(f"Extracting {puma}{geo} boundaries from TIGER WMS.")
        layer = 10 if geo == "bg" else 8
        base = (
            "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
            f"tigerWMS_ACS{year}/MapServer/{layer}/query?"
        )

        request_limit = 50
        if len(targets) > request_limit:
            nchunks = int(np.ceil(len(targets) / request_limit))
            targets = np.array_split(targets, nchunks)

        else:
            targets = np.array([targets])

        ext = gpd.GeoDataFrame()

        for geoids in targets:
            gsearch = ",".join([f"'{g}'" for g in geoids])
            params_ = {"where": f"GEOID IN ({gsearch})"}

            params_.update({"outfields": "*", "f": "geojson"})
            if censusapikey is not None:
                params_.update({"key": censusapikey})

            params = urllib.parse.urlencode(params_)
            url = base + params

            ext_part_ = requests.get(url).text
            ext_part = gpd.read_file(ext_part_)
            ext_part = ext_part.set_index("GEOID")

            ext = pd.concat([ext, ext_part], axis=0)
        if cache:
            if verbose:
                print(
                    f"Writing cached copy of {puma}{geo} "
                    f"boundaries at {cache_geo_path}.\n"
                )
            ext.to_file(cache_geo_path)

    return ext


def make_supertracts(
    geo: gpd.GeoDataFrame,
    sf_data: pd.DataFrame,
    exclude_universe_constraints: bool = True,
    random_state: int = 808,
    method: str = "louvain",
) -> pd.Series:
    """Makes supertracts from tract boundaries and ACS Summary File (SF)
    data  using the Max-P regionalization problem.

    Parameters
    ----------
    geo : geopandas.GeoDataFrame
        Input tract geographies.
    sf_data : pandas.DataFrame
        ACS SF variables for regionalization.
    exclude_universe_constraints : bool = True
        Whether or not to exclude ACS SF universe constraints (default ``True``)
    random_state : int = 808
        Random starting seed for regionalization.
    method : str = 'louvain'
        Method for for creating supertracts. Currently supports only ``'louvain'``.

    Returns
    -------
    super_trt : pandas.Series
        Supertract labels.
    """

    assert method == "louvain", "Argument ``method`` must be 'louvain'"

    if exclude_universe_constraints:
        sf_data = sf_data.loc[
            :,
            (~sf_data.columns.isin(universe_constraints))
            & (~sf_data.columns.isin(universe_codes)),
        ]

    # scale inputs
    scaler = StandardScaler()
    sf_data_z = scaler.fit_transform(sf_data)

    sf_data = sf_data.astype(float)
    sf_data.update(
        pd.DataFrame(
            sf_data_z,
            index=sf_data.index,
            columns=sf_data.columns,
        )
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The weights matrix is not fully connected",
            category=UserWarning,
        )

        # contiguity weights
        w = weights.Rook.from_dataframe(geo, use_index=False)

        # fix island polygons as needed
        if len(w.islands) > 0:
            w_knn1 = weights.KNN.from_dataframe(geo, k=1)
            w_knn1.remap_ids(list(range(geo.shape[0])))

            w = weights.attach_islands(w, w_knn1)

        wadj = w.to_adjlist(drop_islands=True)
        wcon = (
            pd.pivot(wadj, index="focal", columns="neighbor", values="weight")
            .fillna(0)
            .values
        )

        d = euclidean_distances(sf_data, squared=True)
        aff = 1 - (d / d.max())

        G = nx.Graph(aff * wcon)  # noqa: N806

        louv = nx_comm.louvain_communities(G, seed=random_state)

        reg = np.array([0] * geo.shape[0])
        for k in range(len(louv)):
            reg[list(louv[k])] = k

        # assign labels
        super_trt = pd.Series([str(i).zfill(2) for i in reg], index=geo.index)

        # reassign singletons -- nearest-neighbor weights again (temp)
        w_knn1 = weights.KNN.from_dataframe(geo, k=1)

        singletons = geo.index[~super_trt.duplicated(keep=False)]

        for g in singletons:
            nid = list(w_knn1[g].items())[0][0]
            super_trt[g] = super_trt[nid]

        return super_trt


def aggregate_acs_sf_supertracts(
    est_trt: pd.DataFrame,
    est_trt_se: pd.DataFrame,
    labels: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Aggregate estimates, standard errors, and MOEs.
    Called within ``pums.build_super_tracts()``.

    Parameters
    ----------
    est_trt : pd.DataFrame
        Tract-level estimates.
    est_trt_se : pd.DataFrame
        Tract-level standard errors.
    labels : pandas.Series
        Super-tract membership labels indexed by GEOID.

    Returns
    -------
    super_trt_sf : dict[str, pandas.DataFrame]
        Super-tract estimates, standard error, and margins of error.
    """

    ## Estimates
    super_trt_est = est_trt.groupby(labels).agg("sum")

    ## Standard Errors
    super_trt_se = est_trt_se.groupby(labels).agg(lambda x: np.sqrt(np.sum(x**2)))

    ## MOEs
    super_trt_moe = super_trt_se * 1.645

    super_trt_sf = {"est": super_trt_est, "se": super_trt_se, "moe": super_trt_moe}

    return super_trt_sf


def get_vre_tables(
    fips: str,
    year: int,
    table: str,
    cache: bool = False,
    cache_folder: str | pathlib.Path = "./livelike_acs_cache",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Fetches Variance Replicate Estimate (VRE) tables
    via Census FTP.

    Currently only block groups are supported.

    Parameters
    ----------
    fips : str
        7-digit PUMA FIPS code (state + puma).
    year : int, str
        Target year.
    table : str
        ACS SF table ID.
    cache : bool = False
        Whether to cache VRE tables.
    cache_folder : str | pathlib.Path = './livelike_acs_cache'
        The cache folder to store VRE tables. This default value
        creates the folder in the current working directory if not present.
    verbose : bool = False
        Whether to print status messages.

    Returns
    -------
    ext : pd.DataFrame
        VREs for the specified ACS SF table.
    """
    state = fips[:2]
    geo_prefix = "15000US" if year < 2020 else "1500000US"
    label = f"vre_{year}_{state}_{table}"

    # validate cache input and create directory if it doesn't exist
    if cache:
        cache_folder = _ensure_cache_folder_input(cache_folder)

        # check for cached VRE tables
        cache_vre_path = cache_folder / f"{label}.parquet"
        has_cached_vre_file = cache_vre_path.exists() and cache_vre_path.is_file()
    else:
        has_cached_vre_file = False

    if verbose:
        print(f"Cached {label} file exists: {has_cached_vre_file}")

    if has_cached_vre_file and cache:
        if verbose:
            print(f"Using cached {label} file: {cache_vre_path}")
        ext = pd.read_parquet(cache_vre_path)

    else:
        if verbose:
            print(f"Extracting {label}...")

        url = (
            "https://www2.census.gov/programs-surveys/acs/"
            f"replicate_estimates/{year}/data/5-year/150/{table}_{state}.csv.zip"
        )
        ext = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})

        ext = ext[ext.TITLE == "Total:"]
        ext.GEOID = ext.GEOID.str.replace(geo_prefix, "")
        ext = ext.set_index("GEOID")
        ext = ext.loc[:, ext.columns.str.contains("Var_Rep")]

        if cache:
            if verbose:
                print(f"Caching {label} at {cache_vre_path}.\n")
            ext.to_parquet(cache_vre_path)

    return ext
