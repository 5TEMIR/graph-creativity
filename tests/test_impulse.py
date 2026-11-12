from __future__ import annotations

from pathlib import Path

import pytest

from graphcreativity.engine import SimOptions
from graphcreativity.impulse import (
    deviations,
    influence_matrix,
    respond,
    run_baseline,
    run_measure,
    run_state_shock,
    steps_of,
)
from graphcreativity.model import Model, load_model

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "industry.yaml"
HORIZON = 60


@pytest.fixture(scope="module")
def model() -> Model:
    return load_model(CONFIG)


@pytest.fixture(scope="module")
def options() -> SimOptions:
    return SimOptions(horizon=HORIZON)


@pytest.fixture(scope="module")
def baseline(model: Model, options: SimOptions):
    return run_baseline(model, options)


def test_steps_of_matches_grid(model: Model, options: SimOptions) -> None:
    assert steps_of(model, options) == HORIZON + 1


def test_baseline_is_flat(model: Model, baseline) -> None:
    for node in model.indicator_ids:
        assert all(value == 0.0 for value in baseline.states[node])


def test_respond_sees_nothing_in_identical_runs(
    model: Model, baseline
) -> None:
    responses = respond("i1", baseline, baseline, model.indicator_ids)
    assert all(item.first_month is None for item in responses)
    assert all(item.peak == 0.0 for item in responses)
    table = influence_matrix(responses, ["i1"], model.indicator_ids)
    assert table == [[0.0] * len(model.indicator_ids)]


def test_below_threshold_impulse_dies(
    model: Model, baseline, options: SimOptions
) -> None:
    """Импульс меньше порога не двигает ни одну другую вершину."""
    result = run_state_shock(model, "i1", 0.1, options)
    responses = respond("i1", result, baseline, model.indicator_ids)
    others = [item for item in responses if item.target != "i1"]
    assert all(item.peak == 0.0 for item in others)


def test_impulse_response_grows_with_size(
    model: Model, baseline, options: SimOptions
) -> None:
    peaks = []
    for size in (0.1, 0.4, 0.8):
        result = run_state_shock(model, "i2", size, options)
        item = respond("i2", result, baseline, ["i3"])[0]
        peaks.append(item.peak)
    assert peaks[0] == 0.0
    assert peaks[1] <= peaks[2]
    assert peaks[2] > 0.0


def test_measure_propagates_to_goal(
    model: Model, baseline, options: SimOptions
) -> None:
    result = run_measure(model, "p1", options)
    item = respond("p1", result, baseline, ["i3"])[0]
    assert item.first_month is not None
    assert item.peak > 0.0
    assert item.final > 0.0


def test_deviations_are_signed(
    model: Model, baseline, options: SimOptions
) -> None:
    result = run_state_shock(model, "i2", 0.8, options)
    delta = deviations(result, baseline, ["i2", "i3"])
    assert delta["i2"][0] > 0.0
    assert len(delta["i3"]) == steps_of(model, options)
