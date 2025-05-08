from livelike import _attribution
import pandas as pd

def build_attributes(
    puma: pd.DataFrame, 
    level: str, 
    variables: list,
) -> pd.DataFrame:
    """
    Builds attributes for a PUMA at person or household level 
    on a defined list of variables. 

    Parameters
    ----------
    puma : livelike.acs.puma
        Symbolic representation of PUMA.
    level : str
        PUMS level (``'person'`` or ``'household'``).
    variables : list
        Variables of interest.

    Returns
    -------
    atts : pandas.DataFrame
        PUMA attributes at the level of interest.
    """
    if not level in ["person", "household"]:
        raise ValueError(
            "Argument ``level`` must be one of "
            "``'person'``, ``'household'``."
        )

    if getattr(puma, f"est_{level}") is None:
        raise ValueError(
            f"PUMS attributes ``puma.est_{level}`` "
            "cannot be ``None``. "
            "Use ``livelike.acs.puma(keep_intermediates=True)`` "
            "to retain these in ``puma``."
        )

    pums = getattr(puma, f"est_{level}")
    if level == "person":
        pums = pums.reset_index().set_index(["SERIALNO", "SPORDER"])

    atts = pd.concat(
        [getattr(_attribution, v)(pums) for v in variables],
        axis=1
    )
    return atts
