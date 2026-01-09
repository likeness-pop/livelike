"""livelike: A Population Synthesizer for High Demographic Resolution Analysis

@9vt
"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from .acs import (
    aggregate_acs_sf_supertracts,
    build_acs_pums_inputs,
    build_acs_sf_inputs,
    build_census_microdata_api_base_request,
    build_census_microdata_api_geo_request,
    build_constraints_ind,
    build_fips_tree,
    extract_from_census_microdata_api,
    extract_geographies,
    extract_pums_constraints,
    extract_pums_descriptors,
    extract_pums_segment_ids,
    format_acs_code,
    get_vre_tables,
    make_supertracts,
    parse_bg_fips_by_tract,
    parse_pums_variables,
    puma,
    select_constraints,
    update_data_types,
)
from .attribution import (
    build_attributes,
)
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
    up_base_attributes_household,
    up_base_attributes_person,
    up_base_constraints_selection,
    up_constraints_theme_order,
    up_expanded_attributes_household,
    up_expanded_attributes_person,
    up_expanded_constraints_selection,
)
from .est import (
    monte_carlo_estimate,
    tabulate_by_count,
    tabulate_by_serial,
    to_prop,
)
from .homesim import (
    generate_random_states,
    synthesize,
    trs,
)
from .multi import (
    extract_pums_descriptors_multi,
    extract_pums_segment_ids_multi,
    make_pumas,
    make_replicate_pumas,
    synthesize_multi,
)
from .pums import (
    age_cohort,
    age_simple,
    bedrooms,
    civ_noninst_pop,
    commute,
    disability,
    edu_attainment,
    emp_stat,
    foreign_born,
    grade,
    group_quarters,
    group_quarters_pop,
    health_ins,
    hhf,
    hhinc,
    hhsize_vehicles,
    hhtype,
    hhtype_hhsize,
    housing_units,
    hsplat,
    internet,
    intersect_dummies,
    ipr,
    language,
    lep,
    lingisol,
    minors,
    occhu,
    owncost,
    population,
    poverty,
    race,
    reclass_dummies,
    rentcost,
    rooms,
    school,
    seniors,
    sex,
    sex_age,
    sexcw,
    sexnaics,
    sexocc,
    tenure,
    tenure_vehicles,
    travel,
    units,
    veh_occ,
    vet_edu,
    veteran,
    worked,
    year_built,
)
from .sae import (
    estimate,
    summarize,
)
from .utils import (
    clear_acs_cache,
)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("livelike")
