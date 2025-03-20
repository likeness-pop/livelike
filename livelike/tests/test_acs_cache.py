import pytest
from pymedm import PMEDM

from livelike import acs
from livelike.utils import clear_acs_cache

# skip for now if no key available -- reimplement with caches later
if not pytest.USING_KEY:
    pytest.skip(allow_module_level=True)


acs_cache_testing = pytest.testing_cache / "acs_cache_testing"


clear_acs_cache(acs_cache_testing)


p = "4701602"
acs_puma_kwargs = {
    "constraints_selection": {"universe": True},
    "censusapikey": pytest.CENSUSAPIKEY,
    "cache": True,
    "verbose": True,
    "cache_folder": acs_cache_testing,
}


@pytest.xdist_acs_cache
def test_acs_cache():
    acs.puma(p, **acs_puma_kwargs)
    pup_from_cache = acs.puma(p, **acs_puma_kwargs)
    pmd = PMEDM(
        pup_from_cache.year,
        pup_from_cache.est_ind.index,
        pup_from_cache.wt,
        pup_from_cache.est_ind,
        pup_from_cache.est_g1,
        pup_from_cache.est_g2,
        pup_from_cache.se_g1,
        pup_from_cache.se_g2,
    )
    pmd.solve()

    clear_acs_cache(acs_cache_testing)

    observed = float(pmd.res.state.value)

    known = -0.16742593973532213
    assert pytest.approx(observed, abs=0.001) == known


@pytest.xdist_acs_cache_trt
def test_acs_cache_trt():
    acs.puma(p, target_zone="trt", **acs_puma_kwargs)
    pup_trt_from_cache = acs.puma(p, target_zone="trt", **acs_puma_kwargs)
    pmd_trt = PMEDM(
        pup_trt_from_cache.year,
        pup_trt_from_cache.est_ind.index,
        pup_trt_from_cache.wt,
        pup_trt_from_cache.est_ind,
        pup_trt_from_cache.est_g1,
        pup_trt_from_cache.est_g2,
        pup_trt_from_cache.se_g1,
        pup_trt_from_cache.se_g2,
        topo=pup_trt_from_cache.topo,
    )
    pmd_trt.solve()

    clear_acs_cache(acs_cache_testing)

    observed = float(pmd_trt.res.state.value)

    known = -0.10501877678825261
    assert pytest.approx(observed, abs=0.001) == known
