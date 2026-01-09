import awkward as ak
import numpy as np
import pandas as pd


def _sum_rnd_sub_frc_remainders(_sub_frac: np.ndarray | pd.Series) -> int | np.int64:
    """Helper for sum+round fractional remainders to determine sample size."""
    try:
        # standard case
        dfc = _sub_frac.sum().round().astype("int")
    except AttributeError as e:
        if str(e) == "'int' object has no attribute 'round'":
            # fall back when empty pandas.Series passed in
            dfc = int(round(_sub_frac.sum()))
        else:
            raise
    return dfc


def trs(fracs: np.ndarray | pd.Series, random_state: int = 808) -> np.ndarray:
    """Population synthesis using the 'Truncate, Replicate, Sample'
    (TRS) method (Lovelace and Ballas 2013).

    Parameters
    ----------
    fracs : numpy.ndarray | pandas.Series
        Fractional individual counts (a 2D numpy array of individuals by location).
    random_state : int
        Random state value.

    Returns
    -------
    sub_int : numpy.ndarray
        A 2D numpy array of integerized individual counts.
    """

    if not isinstance(fracs, np.ndarray | pd.Series):
        raise TypeError(
            "``fracs`` parameter type expected numpy.ndarray or pandas.Series. "
            "Consult ``livelike.homesim.trs()`` documentation. "
            f"Input type: '{type(fracs).__name__}'"
        )

    # integer counts
    sub_int = np.floor(fracs).astype(int)

    # fractional remainders
    sub_frac = fracs - sub_int

    # fractional remainders must sum to 1
    sub_frac_norm = sub_frac / sub_frac.sum()

    # flatten for simpler indexing
    sub_int = sub_int.flatten()
    sub_frac_norm = sub_frac_norm.flatten()

    # sample size
    dfc = _sum_rnd_sub_frc_remainders(sub_frac)

    np.random.seed(random_state)
    n = np.arange(len(sub_frac_norm))
    sub_frac_norm = np.array(sub_frac_norm, dtype="float64")
    sub_cap = np.random.choice(n, size=dfc, p=sub_frac_norm, replace=False)

    sub_int[sub_cap] = sub_int[sub_cap] + 1
    sub_int = sub_int.reshape(fracs.shape)

    return sub_int


def synthesize(
    almat: np.ndarray,
    est_ind: pd.DataFrame,
    est_g2: pd.DataFrame,
    sporder: pd.DataFrame,
    nsim: int = 30,
    random_state: int = 0,
    longform: bool = True,
) -> pd.DataFrame | list[pd.DataFrame, ...]:
    """
    Synthesizes the baseline residential population by household ID.

    Parameters
    ----------
    almat : numpy.ndarray
        P-MEDM allocation matrix.
    est_ind : pandas.DataFrame
        P-MEDM individual constraints, probably from a ``livelike.acs.puma`` instance.
    est_g2 : pandas.DataFrame
        P-MEDM geographic constraints for the target zones,
        probably from a ``livelike.acs.puma`` instance.
    sporder : pandas.DataFrame
        PUMS household member line numbers, probably
        from a ``livelike.acs.puma`` instance.
    nsim : int = 30
        Number of simulations.
    random_state : int = 0
        Random state value to generate starting seeds for sims.
    longform : bool = True
        Use longform format.

    Returns
    -------
    sim_res : pandas.DataFrame | | list[pd.DataFrame, ...]
        If ``longform=True``, outputs a pandas.DataFrame consisting of the household
        counts (``count``) for each simulation in long format. Otherwise, outputs the
        raw simulation results as a list of the individual pandas.DataFrame objects.
    """

    geoids = est_g2.index.values

    # adjustments to harmonize pop/households
    nm = sporder.reset_index().groupby("SERIALNO").max().SPORDER
    nm = nm.reindex_like(est_ind)
    nm = nm.fillna(0)

    adj = est_ind["population"] / nm
    adj = adj.fillna(1)  # vacant
    almat_adj = almat * adj.values[:, None]

    # vacant housing units indicator (not directly in constraints)
    vachu = ((est_ind["occhu"] == 0) & (est_ind["group_quarters_pop"] == 0)).astype(
        "int"
    )
    vachu.name = "vacant"

    # create residential sampling scheme on
    # household type x household size
    res_cols_ = est_ind.columns[est_ind.columns.str.contains("^hht.*hhsize")].tolist()
    if len(res_cols_) == 0:
        raise ValueError(
            "Individual constraints must contain "
            "household type by household size columns."
        )
    res_cols_ += ["group_quarters_pop"]
    res_ind = est_ind.loc[:, res_cols_]
    res_ind = pd.concat([res_ind, vachu], axis=1)
    res_cols = res_ind.columns.tolist()

    tosamp_ = np.matmul(res_ind.values.T, np.array(almat_adj))

    # random states for each run
    np.random.seed(random_state)
    starts = generate_random_states(n=nsim, seed=random_state)

    sims = []

    for s in starts:
        tosamp = pd.DataFrame(
            trs(tosamp_, random_state=s),
            index=res_cols,
            columns=geoids,
        )

        serial = est_ind.index.values

        _syp_ = {}
        for g in geoids:
            gwhere = np.where(geoids == g)[0][0]

            gal = almat_adj[:, gwhere]
            gal = gal / gal.sum()

            gts = tosamp[[g]]

            gsyp = {}
            for v in gts.index.values:
                gvp = gal * res_ind.loc[:, v]
                gvp = gvp / gvp.sum()

                gvsamp = gts.loc[v].values[0]

                if gvsamp > 0:
                    np.random.seed(int(g[4:]))
                    gsyp[v] = np.random.choice(
                        serial, size=gvsamp, p=gvp, replace=True
                    ).tolist()
                else:
                    gsyp[v] = []

            _syp_[g] = gsyp

        syp_ = [
            ak.flatten(ak.Array([j for i, j in v.items()])) for k, v in _syp_.items()
        ]
        syp = pd.concat(
            [pd.Series(s).value_counts().reindex_like(est_ind).fillna(0) for s in syp_],
            axis=1,
        )
        syp.columns = geoids

        sims += [syp]

    sim_res = _make_longform(sims) if longform else sims if nsim > 1 else sims[0]

    return sim_res


def _make_longform(sims: list[pd.DataFrame, ...]) -> pd.DataFrame:
    """Helper to convert multiple simulated ``synthesize()`` results to longform."""

    sim_res = []

    for s in range(len(sims)):
        sx = pd.melt(sims[s], ignore_index=False)
        sx = sx.rename({"variable": "geoid", "value": "count"}, axis=1)
        sx["sim"] = s
        sx = sx.loc[:, ["sim", "geoid", "count"]]
        sx.index.name = "h_id"
        sx["count"] = sx["count"].astype("int")
        sx = sx[sx["count"] > 0]

        sim_res += [sx.reset_index()]

    return pd.concat(sim_res, axis=0).set_index("h_id")


def generate_random_states(n: int = 1, seed: int = 0) -> np.ndarray:
    """
    Generates random states for iterative sampling.

    Parameters
    ----------
    n : int
        Number of random states.
    seed : int
        Random starting seed.

    Returns
    -------
    np.ndarray
        Random integer array.
    """

    np.random.seed(seed)
    return np.random.randint(low=0, high=2**32 - 1, size=n, dtype="int64")
