from __future__ import annotations

from pathlib import Path

import pytest

from graphcreativity.engine import SimOptions, simulate
from graphcreativity.metrics import (
    compare_named_plans,
    compare_plans,
    critical_event_combinations,
    critical_removal_sets,
    effect_strength,
    minimal_sufficient_sets,
    robustness_radius,
    scenario_robustness,
    summarize,
)
from graphcreativity.model import (
    Edge,
    MeasureProgram,
    Model,
    ModelParams,
    Node,
    Plan,
    Scenario,
    load_model,
)

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "karkas.yaml"


@pytest.fixture(scope="module")
def model() -> Model:
    return load_model(CONFIG)


# ---------------------------------------------------------------------------
# §5 karkas.tex: статическая сила эффекта
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("measures", "expected"),
    [
        ((), 0.0),
        (("m1",), 0.6125),
        (("m3",), 0.7650),
        (("m4",), 0.3500),
        (("m3", "m4"), 1.1150),
        (("m1", "m3"), 1.3775),
        (("m1", "m4"), 0.9625),
    ],
)
def test_effect_strength_matches_karkas_table(model: Model, measures, expected):
    """Числа §5 — сумма путей длины <= 2, а не динамика."""
    assert effect_strength(model, measures) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Качественная структура каркаса
# ---------------------------------------------------------------------------

def _rows_by_set(model: Model) -> dict[tuple[str, ...], dict]:
    rows = compare_plans(model, "nominal")
    return {tuple(row["measures"]): row for row in rows}


def test_two_minimal_sufficient_sets(model: Model):
    rows = compare_plans(model, "nominal")
    mss = minimal_sufficient_sets(rows)
    assert mss == [("m1", "m3"), ("m3", "m4")]

    by_set = _rows_by_set(model)
    assert by_set[("m1", "m3")]["reached"]
    assert by_set[("m3", "m4")]["reached"]
    assert not by_set[("m1", "m4")]["reached"]
    # «почти достаточная» пара из §5: не дотягивает, но близко
    assert by_set[("m1", "m4")]["x_target_T"] > 0.9
    for single in (("m1",), ("m3",), ("m4",)):
        assert not by_set[single]["reached"]


def test_low_theta_makes_m1_sufficient(model: Model):
    """При θ=0.30 дешёвая m1 одна достигает цели — это граница применимости."""
    rows = compare_plans(model, "nominal", SimOptions(theta=0.30))
    mss = minimal_sufficient_sets(rows)
    assert ("m1",) in mss
    assert ("m1",) not in minimal_sufficient_sets(compare_plans(model, "nominal"))


def test_budget_excludes_sufficient_set(model: Model):
    """P1 стоит 50 млн; при бюджете 45 млн он не может быть «достаточным»."""
    rows = compare_plans(model, "nominal", SimOptions(budget_max=45.0))
    mss = minimal_sufficient_sets(rows)
    assert ("m3", "m4") not in mss
    assert ("m1", "m3") in mss


def test_critical_sets_are_distinct_objects(model: Model):
    rows = compare_plans(model, "nominal")
    removals = critical_removal_sets(rows, model.measure_ids)
    events = critical_event_combinations(model, "P1")
    assert removals == [("m3",), ("m1", "m4")]
    assert events == [
        ("grant_cut",),
        ("low_qualification",),
        ("market_downturn",),
    ]
    assert critical_event_combinations(model, "hybrid") == []


def test_p1_is_fragile_hybrid_is_robust(model: Model):
    p1 = scenario_robustness(model, "P1")
    hybrid = scenario_robustness(model, "hybrid")
    assert hybrid["success_rate"] == pytest.approx(1.0)
    assert p1["success_rate"] < hybrid["success_rate"]
    assert p1["success_rate"] <= 0.4
    assert robustness_radius(model, "P1") == 0
    assert robustness_radius(model, "hybrid") == len(model.atomic_scenarios)
    assert robustness_radius(model, "P2") == -1  # не проходит уже в номинале


def test_strict_equation_never_reaches_target(model: Model):
    """Буквальное (1 - x) делает x=1 асимптотой: цель недостижима."""
    result = simulate(model, "full", "nominal", SimOptions(saturation="strict"))
    assert max(result.states["X1"]) < 1.0
    assert result.first_goal_time() is None


def test_ceiling_mode_reaches_target(model: Model):
    result = simulate(model, "full", "nominal")
    assert result.first_goal_time() is not None
    assert result.first_goal_time() <= model.params.horizon
    assert result.states["X1"][-1] > 1.0


# ---------------------------------------------------------------------------
# Синтетические проверки механики
# ---------------------------------------------------------------------------

def _tiny_model(
    *,
    delay: float = 3.0,
    measure_cost: float = 1.0,
    budget_max: float = 10.0,
    horizon: int = 12,
) -> Model:
    params = ModelParams(
        horizon=horizon,
        theta=0.0,
        mu=0.0,
        x_max=1.6,
        saturation="ceiling",
        target_node="X",
        budget_max=budget_max,
    )
    nodes = {
        "m": Node(id="m", kind="measure", cost=measure_cost),
        "X": Node(
            id="X",
            kind="indicator",
            direction="up",
            base=0.0,
            target=1.0,
            rise=12.0,
        ),
    }
    return Model(
        params=params,
        nodes=nodes,
        edges=[Edge(src="m", dst="X", w=1.0, delay=delay)],
        plans={"P": Plan("P", programs={"m": MeasureProgram(start=0.0)})},
        scenarios={"nominal": Scenario("nominal")},
    )


def test_delay_shifts_response():
    model = _tiny_model(delay=3.0)
    result = simulate(model, "P", "nominal")
    first_positive = next(
        (t for t, value in enumerate(result.states["X"]) if value > 0), None
    )
    assert first_positive == 4  # a(0) доступно на шаге 3, x(4) — первый отклик


def test_per_node_theta_is_used_and_can_be_overridden():
    """Per-node θ работает, но явный SimOptions(theta=...) его перекрывает."""
    model = _tiny_model()
    model.nodes["X"].theta = 10.0

    blocked = simulate(model, "P", "nominal")
    assert blocked.states["X"][-1] == pytest.approx(0.0)

    overridden = simulate(model, "P", "nominal", SimOptions(theta=0.0))
    assert overridden.states["X"][-1] > 0.0


def test_named_plan_uses_its_own_starts(model: Model):
    """Гибрид стартует m3 с 3-го месяца, поэтому он не равен {m1,m3} с нуля."""
    named = {row["plan"]: row for row in compare_named_plans(model, "nominal")}
    subset = {
        tuple(row["measures"]): row for row in compare_plans(model, "nominal")
    }
    assert named["hybrid"]["t_goal"] == pytest.approx(30.0)
    assert subset[("m1", "m3")]["t_goal"] == pytest.approx(28.0)
    assert named["hybrid"]["x_target_T"] < subset[("m1", "m3")]["x_target_T"]


def test_no_measures_no_growth(model: Model):
    result = simulate(model, measures=[], scenario="nominal")
    assert result.states["X1"][-1] == pytest.approx(0.0)


def test_budget_violation_is_marked():
    model = _tiny_model(measure_cost=100.0, budget_max=10.0)
    result = simulate(model, "P", "nominal")
    assert result.feasible is False
    assert result.warnings


def test_summary_fields_consistent(model: Model):
    result = simulate(model, "hybrid", "nominal")
    summary = summarize(result)
    assert summary["reached"] is True
    assert summary["t_goal"] == result.first_goal_time()
    assert summary["cost_total"] == pytest.approx(43.0)
    assert summary["k_T"] == pytest.approx(
        summary["revenue_T"] / summary["cost_total"]
    )
    assert 0.0 <= summary["x_target_T"] <= model.params.x_max
