from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from graphcreativity.editing import (
    build_model,
    edge_rows,
    model_to_yaml,
    new_node_row,
    node_rows,
    plan_rows,
    save_model_yaml,
    scenario_rows,
    suggest_node_id,
    validate_new_node_id,
)
from graphcreativity.engine import simulate
from graphcreativity.metrics import compare_plans
from graphcreativity.model import MEASURE, load_model
from graphcreativity.plots import plot_dynamics

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "karkas.yaml"


@pytest.fixture(scope="module")
def model():
    return load_model(CONFIG)


def test_roundtrip_through_yaml(model, tmp_path):
    text = model_to_yaml(model)
    path = tmp_path / "edited.yaml"
    path.write_text(text, encoding="utf-8")
    loaded = load_model(path)

    assert set(loaded.nodes) == set(model.nodes)
    assert len(loaded.edges) == len(model.edges)
    for original, restored in zip(model.edges, loaded.edges):
        assert (original.src, original.dst) == (restored.src, restored.dst)
        assert original.w == pytest.approx(restored.w)
        assert original.delay == pytest.approx(restored.delay)
        assert original.optional == restored.optional

    assert loaded.params.theta == pytest.approx(model.params.theta)
    assert loaded.params.version == model.params.version
    assert loaded.plans["hybrid"].programs["m3"].start == pytest.approx(3.0)
    assert loaded.scenarios["combined_stress"].values["Y3"] == pytest.approx(0.5)
    assert loaded.scenarios["combined_stress"].atomic is False
    assert loaded.node("m1").owner == model.node("m1").owner
    assert loaded.node("X7").monetization["volume"] == pytest.approx(1_000_000.0)


def test_save_model_yaml_roundtrip(model, tmp_path):
    path = save_model_yaml(model, tmp_path / "nested" / "export.yaml")
    assert path.exists()
    loaded = load_model(path)
    assert set(loaded.nodes) == set(model.nodes)
    assert len(loaded.edges) == len(model.edges)


def test_edit_node_preserves_hidden_fields(model):
    rows = node_rows(model)
    for row in rows:
        if row["id"] == "X1":
            row["base"] = 110.0
            row["label"] = "Изменённый X1"

    edited = build_model(model, rows, edge_rows(model))
    assert edited.node("X1").base == pytest.approx(110.0)
    assert edited.node("X1").label == "Изменённый X1"
    assert edited.node("X1").theta == model.node("X1").theta
    assert edited.node("X1").x_max == model.node("X1").x_max


def test_model_without_measures_still_simulates(model):
    measure_ids = set(model.measure_ids)
    rows_n = [row for row in node_rows(model) if row["kind"] != MEASURE]
    rows_e = [
        row for row in edge_rows(model) if row["from"] not in measure_ids
    ]
    edited = build_model(model, rows_n, rows_e)

    assert edited.measure_ids == []
    assert all(not plan.programs for plan in edited.plans.values())
    result = simulate(edited, measures=[], scenario="nominal")
    assert result.states["X1"][-1] == pytest.approx(0.0)
    assert compare_plans(edited, "nominal", include_empty=True)

    fig = plot_dynamics(result)
    assert fig is not None
    plt.close(fig)


def test_edit_plan_programs_and_scenario_values(model):
    plans = plan_rows(model)
    plans["hybrid"]["m1"].update(
        {"start": 6.0, "duration": 12.0, "intensity": 0.5}
    )
    plans["hybrid"].pop("m3")

    scenarios = scenario_rows(model)
    scenarios["nominal"]["Y3"] = -0.25
    scenarios["nominal"]["Y2"] = 0.30

    edited = build_model(
        model,
        node_rows(model),
        edge_rows(model),
        plans,
        scenarios,
    )

    program = edited.plans["hybrid"].programs["m1"]
    assert program.start == pytest.approx(6.0)
    assert program.duration == pytest.approx(12.0)
    assert program.intensity == pytest.approx(0.5)
    assert "m3" not in edited.plans["hybrid"].programs
    assert edited.scenarios["nominal"].values["Y3"] == pytest.approx(-0.25)
    assert edited.scenarios["nominal"].values["Y2"] == pytest.approx(0.30)

    # Возвращённые строки не должны быть ссылками на объекты исходной модели.
    plans["hybrid"]["m1"]["start"] = 99.0
    assert model.plans["hybrid"].programs["m1"].start != pytest.approx(99.0)


def test_new_environment_is_zero_in_all_scenarios(model):
    scenarios = scenario_rows(model)
    scenarios["nominal"]["Y6"] = 0.0
    edited = build_model(
        model,
        node_rows(model),
        edge_rows(model),
        plan_rows(model),
        scenarios,
    )
    assert "Y6" not in edited.scenarios["nominal"].values

    row = new_node_row(model, "Y6", "environment", label="Новый фактор")
    edited = build_model(
        model,
        node_rows(model) + [row],
        edge_rows(model),
        plan_rows(model),
        scenarios,
    )
    assert edited.scenarios["nominal"].values["Y6"] == pytest.approx(0.0)
    assert edited.scenarios["grant_cut"].values["Y6"] == pytest.approx(0.0)


def test_edit_edge_weight_changes_dynamics(model):
    rows = edge_rows(model)
    for row in rows:
        if row["from"] == "m3" and row["to"] == "X1":
            row["w"] = 0.50

    edited = build_model(model, node_rows(model), rows)
    before = simulate(model, "hybrid", "nominal")
    after = simulate(edited, "hybrid", "nominal")
    assert after.states["X1"][-1] > before.states["X1"][-1]


def test_remove_node_cascades_edges_and_plans(model):
    rows_n = [row for row in node_rows(model) if row["id"] != "m4"]
    rows_e = [
        row
        for row in edge_rows(model)
        if row["from"] != "m4" and row["to"] != "m4"
    ]
    edited = build_model(model, rows_n, rows_e)

    assert "m4" not in edited.nodes
    assert all(edge.src != "m4" and edge.dst != "m4" for edge in edited.edges)
    assert "m4" not in edited.plans["P1"].programs
    assert "m3" in edited.plans["P1"].programs


def test_remove_environment_cascades_scenario_values(model):
    rows_n = [row for row in node_rows(model) if row["id"] != "Y2"]
    rows_e = [
        row
        for row in edge_rows(model)
        if row["from"] != "Y2" and row["to"] != "Y2"
    ]
    edited = build_model(model, rows_n, rows_e)

    assert "Y2" not in edited.nodes
    assert "Y2" not in edited.scenarios["grant_cut"].values
    assert edited.scenarios["grant_cut"].values.get("Y2") is None


def test_add_measure_and_edge(model):
    row = new_node_row(
        model,
        "m5",
        MEASURE,
        label="Тестовая мера",
        cost=5.0,
    )
    rows_e = edge_rows(model) + [
        {"from": "m5", "to": "X1", "w": 0.80, "delay": 0, "optional": False}
    ]
    edited = build_model(model, node_rows(model) + [row], rows_e)

    assert "m5" in edited.nodes
    assert edited.nodes["m5"].cost == pytest.approx(5.0)
    assert any(edge.src == "m5" and edge.dst == "X1" for edge in edited.edges)

    result = simulate(edited, measures=["m5"], scenario="nominal")
    assert result.states["X1"][-1] > 0


def test_unknown_edge_endpoint_rejected(model):
    rows_e = edge_rows(model) + [
        {"from": "m1", "to": "unknown", "w": 0.1, "delay": 1}
    ]
    with pytest.raises(ValueError, match="не найдена"):
        build_model(model, node_rows(model), rows_e)


def test_duplicate_edge_rejected(model):
    rows_e = edge_rows(model)
    rows_e.append(dict(rows_e[0]))
    with pytest.raises(ValueError, match="встречается дважды"):
        build_model(model, node_rows(model), rows_e)


def test_duplicate_node_rejected(model):
    rows_n = node_rows(model)
    rows_n.append(dict(rows_n[0]))
    with pytest.raises(ValueError, match="дважды"):
        build_model(model, rows_n, edge_rows(model))


def test_target_node_cannot_disappear(model):
    rows_n = [row for row in node_rows(model) if row["id"] != "X1"]
    rows_e = [
        row
        for row in edge_rows(model)
        if row["from"] != "X1" and row["to"] != "X1"
    ]
    with pytest.raises(ValueError, match="Целевая вершина"):
        build_model(model, rows_n, rows_e)


def test_node_id_helpers(model):
    assert validate_new_node_id("m5", model.nodes) == "m5"
    with pytest.raises(ValueError):
        validate_new_node_id("m 5", model.nodes)
    with pytest.raises(ValueError):
        validate_new_node_id("m1", model.nodes)

    assert suggest_node_id(model, MEASURE) == "m5"
    assert suggest_node_id(model, "indicator") == "X8"
    assert suggest_node_id(model, "environment") == "Y6"
