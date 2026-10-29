from __future__ import annotations

from pathlib import Path

import pytest

from graphcreativity.engine import simulate
from graphcreativity.metrics import summarize
from graphcreativity.model import load_model

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "article_example.yaml"


@pytest.fixture(scope="module")
def model():
    return load_model(CONFIG)


def _summary(model, plan: str) -> dict:
    return summarize(simulate(model, plan, "nominal"))


def test_article_example_matches_p1_p2_ranges(model):
    p1 = _summary(model, "P1")
    p2 = _summary(model, "P2")
    hybrid = _summary(model, "hybrid")

    # P1: 50-70 млн, эффект 30-40 %, плато 24-36 мес.
    assert p1["cost_total"] == pytest.approx(70.0)
    assert 30.0 <= p1["growth_pct_T"] <= 40.0
    assert p1["t_goal"] is not None
    assert 24.0 <= p1["t_goal"] <= 36.0

    # P2: 10-20 млн, эффект 15-25 %, но цели 25 % не хватает.
    assert p2["cost_total"] == pytest.approx(15.0)
    assert 15.0 <= p2["growth_pct_T"] <= 25.0
    assert p2["t_goal"] is None

    # Гибрид: сначала P2, затем акселератор P1; быстрее и выгоднее P1.
    assert hybrid["cost_total"] == pytest.approx(50.0)
    assert 30.0 <= hybrid["growth_pct_T"] <= 40.0
    assert hybrid["t_goal"] < p1["t_goal"]
    assert hybrid["k_T"] > p1["k_T"]


def test_p1_is_synergy_of_accelerator_and_patenting(model):
    m3 = summarize(simulate(model, measures=("m3",)))
    m4 = summarize(simulate(model, measures=("m4",)))
    p1 = _summary(model, "P1")

    assert m3["t_goal"] is None
    assert m4["t_goal"] is None
    assert p1["t_goal"] is not None
    assert p1["growth_pct_T"] > m3["growth_pct_T"]
    assert p1["growth_pct_T"] > m4["growth_pct_T"]
