import pathlib

import numpy as np
import pandas as pd

from livelike import acs
from livelike import homesim as hs

from .config import constraints
from .utils import _ensure_cache_folder_input


def make_pumas(
    fips: list[str],
    constraints: pd.DataFrame = constraints,
    constraints_selection: None | dict = None,
    constraints_theme_order: None | list = None,
    year: int = 2019,
    target_zone: str = "bg",
    keep_intermediates: bool = False,
    make_trt_geo: bool = True,
    keep_geo: bool = False,
    make_super_trt: bool = True,
    make_super_trt_method: str = "louvain",
    append_year: bool = False,
    censusapikey: None | str = None,
    cache: bool = False,
    cache_folder: str | pathlib.Path = "./livelike_acs_cache",
) -> dict[str, acs.puma]:
    """Creates ``livelike.acs.puma`` objects for a specified area of interest.

    Parameters
    ----------
    fips : list[str]
        A list of seven-digit FIPS codes for PUMAs in the area of interest.
    constraints : pandas.DataFrame (config.constraints)
        Constraining variables for P-MEDM, formatted with a constraint
        name ('constraint') and ACS variable code ('code').
    constraints_selection : None | dict = None
        P-MEDM constraints to use. Key-value pairs with keys representing
        ACS variable themes and values representing specific subjects (tables).

        If the value passed is a ``bool`` type, a ``True`` value will include
        variables for all subjects in the theme, while a ``False`` value will
        bypass that theme (the same as omitting the theme from the selection).

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
    year : int = 2019
        ACS 5-Year Estimates vintage.
    target_zone : str = 'bg'
        The target zone type for spatial allocation. Options include: ``'bg'``,
        which creates a block group (target) - tract (aggregate) hierarchy, and
        ``'trt'``, which creates a tract (target) and super-tract (aggregate)
        hierarchy.
    keep_intermediates : bool = False
        Should intermediate P-MEDM household and person-level constraints
        (pre-aggregation) be kept? Default ``False``.
    keep_geo : bool = False
        When creating a tract/supertract representation of a PUMA, keep the
        geometries used to perform regionalization of tracts into supertracts?
    make_trt_geo : bool = False
        When ``target_zone='trt'``, extracts census tract boundaries from the
        TIGER Web Mapping Service. Default ``True`` and should not be modified
        except as accessed internally by other ``livelike`` functions.
    make_super_trt : bool = True
        When ``target_zone='trt'``, creates supertracts as the aggregation zones
        (``g2``). Default ``True`` and should not be modified except as accessed
        internally by other ``livelike`` functions.
    make_super_trt_method : str = 'louvain'
        Method for for creating supertracts. Currently supports only
        ``'louvain'``. See ``acs.make_supertracts()`` for more information.
    append_year : bool = False
        Whether or not to append ``year`` to the output keys.
    censusapikey : None | str = None
        Optional Census API key.
    cache : bool = False
        Whether to cache ACS files locally. These are saved in Apache
        Parquet format. For more information see [https://parquet.apache.org/].
    cache_folder : str | pathlib.Path = './livelike_acs_cache'
        The cache folder to store queried ACS inputs. This default value
        creates the folder in the current working directory if not present.

    Returns
    -------
    mpu : dict[str, acs.puma]
        A dictionary with keys ``fips`` and values ``livelike.acs.puma``.
    """

    # validate cache input and create directory if it doesn't exist
    if cache:
        cache_folder = _ensure_cache_folder_input(cache_folder)

    mpu = dict.fromkeys(fips)

    for p in mpu:
        mpu[p] = acs.puma(
            p,
            constraints=constraints,
            constraints_selection=constraints_selection,
            constraints_theme_order=constraints_theme_order,
            year=year,
            target_zone=target_zone,
            keep_intermediates=keep_intermediates,
            keep_geo=keep_geo,
            make_trt_geo=make_trt_geo,
            make_super_trt=make_super_trt,
            make_super_trt_method=make_super_trt_method,
            censusapikey=censusapikey,
            cache=cache,
            cache_folder=cache_folder,
        )

    if append_year:
        for p in fips:
            pyr = p + "_" + str(year)

            mpu[pyr] = mpu.pop(p)

    return mpu


def make_replicate_pumas(
    fips: str,
    constraints: pd.DataFrame = constraints,
    constraints_selection: None | dict = None,
    constraints_theme_order: None | list = None,
    year: int = 2019,
    target_zone: str = "bg",
    nreps: int = 1,
    keep_intermediates: bool = False,
    censusapikey: None | str = None,
    cache_folder: str | pathlib.Path = "./livelike_acs_cache",
) -> dict[str, acs.puma]:
    """Creates ``livelike.acs.puma`` objects for a specified area of interest.
    Intermediary results are cached in Apache Parquet format.
    For more information see [https://parquet.apache.org/].

    Parameters
    ----------
    fips : str
        A single seven-digit FIPS code for a PUMA in the area of interest.
    constraints : pandas.DataFrame (config.constraints)
        Constraining variables for P-MEDM, formatted with a constraint
        name (``'constraint'``) and ACS variable code (``'code'``).
    constraints_selection : None | dict = None
        P-MEDM constraints to use. Key-value pairs with keys representing
        ACS variable themes and values representing specific subjects (tables).

        If the value passed is a ``bool`` type, a ``True`` value will include
        variables for all subjects in the theme, while a ``False`` value will
        bypass that theme (the same as omitting the theme from the selection).

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
    year : int = 1
        ACS 5-Year Estimates vintage.
    target_zone : str = 'bg'
        The target zone type for spatial allocation. Currently only block
        groups (``bg``) are supported.
    nreps : int = 1
        Number of replicate weights to use. Must be between 1 and 80.
    keep_intermediates : bool = False
        Should intermediate P-MEDM household and person-level constraints
        (pre-aggregation) be kept? Default ``False``.
    censusapikey : None | str = None
        Optional Census API key.
    cache_folder : str | pathlib.Path = './livelike_acs_cache'
        The cache folder to store queried ACS inputs. This default value
        creates the folder in the current working directory if not present.

    Returns
    -------
    mpu : dict[str, acs.puma]
        A dictionary with keys ``fips`` and values ``livelike.acs.puma``.
    """

    # TODO: is this still the case?
    if target_zone != "bg":
        raise ValueError("Only geographies of type ``bg`` are supported at this time.")

    if not 1 <= nreps <= 80:
        raise ValueError("Number of replicates must be between 1 and 80.")

    # validate cache input and create directory if it doesn't exist
    cache_folder = _ensure_cache_folder_input(cache_folder)

    mpu_ = []

    # replicate ``acs.puma`` instances must be generated in reverse order
    # to ensure all requireq replicated weights are cached
    reps = list(range(0, nreps + 1))[::-1]
    for r in reps:
        pup = acs.puma(
            fips,
            constraints=constraints,
            constraints_selection=constraints_selection,
            constraints_theme_order=constraints_theme_order,
            year=year,
            cache=True,
            target_zone=target_zone,
            replicate=r,
            keep_intermediates=keep_intermediates,
            censusapikey=censusapikey,
            cache_folder=cache_folder,
        )
        mpu_ += [pup]

    mpu_keys = [f"{fips}_{r}" for r in reps]
    mpu = dict(zip(mpu_keys, mpu_, strict=True))

    return mpu


def synthesize_multi(
    mpu: dict[str, acs.puma],
    pmds: dict[str, ...],
    nsim: int = 30,
    append_year: bool = False,
) -> pd.DataFrame:
    """Creates baseline residential synthetic populations from P-MEDM
    solutions for PUMAs in an area of interest.

    Parameters
    ----------
    mpu : dict[str, acs.puma]
        PUMA instances for the area of interest.
    pmds : dict[str, pymedm.PMEDM | pmedm_legacy.PMEDM]
        Solved P-MEDM problems for PUMAs in the area of interest. These
        ``PMEDM`` instances can be from either ``pymedm`` or ``pmedm_legacy``.
    nsim : int = 30
        Number of simulations to perform (default is ``30``).
    append_year : bool = False
        Whether or not to append ``year`` to the output keys.

    Returns
    -------
    syp : pandas.DataFrame
        A DataFrame consisting of the household counts (``count``)
        for each simulation in long format.
    """

    # TODO: enable parallel processing
    syp = pd.DataFrame()
    for p in mpu:
        _syp = hs.synthesize(
            pmds[p].almat,
            mpu[p].est_ind,
            mpu[p].est_g2,
            mpu[p].sporder,
            nsim=nsim,
            random_state=int(mpu[p].fips),  # TODO: parameterize
        )
        _syp["replicate"] = p
        if append_year:
            _syp["year"] = mpu[p].year
        syp = pd.concat([syp, _syp], axis=0)

    return syp


def extract_pums_segment_ids_multi(
    fips: list[str],
    level: str,
    query: str,
    year: int = 2019,
    append_year: bool = False,
    censusapikey: None | str = None,
) -> pd.DataFrame:
    """Extracts PUMS records matching a segment described by an API query
    for multiple PUMAs in an area of interest.

    Parameters
    ----------
    fips : list[str]
        7-digit PUMA FIPS codes in the area of interest (state + puma).
    level : str
        PUMS level (``'person'`` or ``'household'``).
    query : str
        Query containing segment characteristics.
    year : int = 2019
        ACS 5-Year Estimates vintage.
    append_year : bool = False
        Whether or not to append ``year`` to the output keys.
    censusapikey : None | str = None
        Optional Census API key.

    Returns
    -------
    extract : pandas.DataFrame
        A DataFrame containing household identifiers (SERIALNO) and household
        structure if a person-level request (SPORDER).
    """

    levels = np.array(["person", "household"])
    assert any(levels == level), (
        "Argument ``level`` must be one of: ``person``, ``household``."
    )

    extract = pd.DataFrame()

    for p in fips:
        ext = acs.extract_pums_segment_ids(
            p, level, query, year=year, censusapikey=censusapikey
        )

        extract = pd.concat([extract, ext], axis=0)

    if append_year:
        extract["year"] = [year] * extract.shape[0]

    return extract


def extract_pums_descriptors_multi(
    fips: list[str],
    level: str,
    features: np.typing.ArrayLike,
    year: int = 2019,
    append_year: bool = False,
    sample_weights: bool = False,
    censusapikey: None | str = None,
) -> pd.DataFrame:
    """Extracts descriptive features from the PUMS for PUMAs in an area of interest.

    Parameters
    ----------
    fips : list[str]
        7-digit PUMA FIPS codes in the area of interest (state + puma).
    level : str
        PUMS level ('person' or 'household').
    features : array-like
        Target feature names.
    year : int = 2019
        ACS 5-Year Estimates vintage.
    append_year : bool = False
        Whether or not to append ``year`` to the output keys.
    sample_weights : bool =False
        Whether or not to append PUMS sample weights.
    censusapikey : None | str = None
        Optional Census API key.

    Returns
    -------
    extract : pandas.DataFrame
        A DataFrame containing household identifiers (SERIALNO) and household
        structure if a person-level request (SPORDER).
    """

    levels = np.array(["person", "household"])
    assert any(levels == level), (
        "Argument ``level`` must be one of: ``person``, ``household``."
    )

    extract = pd.DataFrame()

    for p in fips:
        ext = acs.extract_pums_descriptors(
            p,
            level,
            features,
            year=year,
            sample_weights=sample_weights,
            censusapikey=censusapikey,
        )

        extract = pd.concat([extract, ext], axis=0)

    if append_year:
        extract["year"] = [year] * extract.shape[0]

    return extract
