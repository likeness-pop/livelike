import pytest

from livelike import acs
from livelike.config import constraints

# target PUMA
p = "4701604"


class TestConstraintUniversePopulationErrors:
    def test_custom_sel_constr_error(self):
        _sel = {"universe": ["population", "group_quarters_pop"]}
        cst = acs.select_constraints(constraints, _sel)

        with pytest.raises(
            ValueError, match="Check the ``puma.constraints.constraint`` attribute."
        ):
            acs.puma(p, constraints=cst)

    def test_constr_sel_no_uni(self):
        _sel = {"universe": ["population", "group_quarters_pop", "housing_units"]}
        cst = acs.select_constraints(constraints, _sel)

        with pytest.raises(
            ValueError,
            match=(
                "Check the ``puma.constraints_selection`` attribute. "
                "Constraints selection must include ``universe`` key."
            ),
        ):
            acs.puma(
                p,
                constraints=cst,
                constraints_selection={
                    "economic": True,
                    "mobility": True,
                },
            )

    def test_constr_sel_false_uni(self):
        _sel = {"universe": ["population", "group_quarters_pop", "housing_units"]}
        cst = acs.select_constraints(constraints, _sel)

        with pytest.raises(
            ValueError,
            match=(
                "Check the ``puma.constraints_selection`` attribute. "
                "The value of ``universe`` must not be ``False``."
            ),
        ):
            acs.puma(
                p,
                constraints=cst,
                constraints_selection={
                    "economic": True,
                    "mobility": True,
                    "universe": False,
                },
            )

    def test_constr_sel_bad_type(self):
        _sel = {"universe": ["population", "group_quarters_pop", "housing_units"]}
        cst = acs.select_constraints(constraints, _sel)

        with pytest.raises(
            TypeError,
            match=(
                "Check the ``puma.constraints_selection`` attribute. "
                "The value of ``universe`` must be a ``list``."
            ),
        ):
            acs.puma(
                p,
                constraints=cst,
                constraints_selection={
                    "economic": True,
                    "mobility": True,
                    "universe": "group_quarters_pop",
                },
            )

    def test_constr_sel_no_pop(self):
        _sel = {"universe": ["population", "group_quarters_pop", "housing_units"]}
        cst = acs.select_constraints(constraints, _sel)

        with pytest.raises(
            ValueError,
            match=(
                "Check the ``puma.constraints_selection`` attribute. "
                "The value of ``universe`` must include "
            ),
        ):
            acs.puma(
                p,
                constraints=cst,
                constraints_selection={
                    "economic": True,
                    "mobility": True,
                    "universe": ["group_quarters_pop", "housing_units"],
                },
            )

    def test_constr_theme_order_no_uni(self):
        _sel = {"universe": ["population", "group_quarters_pop", "housing_units"]}
        cst = acs.select_constraints(constraints, _sel)
        custom_theme_order = ["mobility", "economic", "demographic"]

        with pytest.raises(
            ValueError,
            match=(
                "Check the ``puma.constraints_theme_order`` attribute. "
                "Theme order must include ``universe``."
            ),
        ):
            acs.puma(
                p,
                constraints=cst,
                constraints_selection={
                    "economic": True,
                    "mobility": True,
                    "universe": True,
                },
                constraints_theme_order=custom_theme_order,
            )
