from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphcreativity.data import (
    load_series,
    load_series_set,
    pearson,
    resolve_root,
    yoy_growth,
)


def _make_panel(tmp_path: Path, indicator: str = "123") -> Path:
    root = tmp_path / "rosstat"
    item = root / "by-id" / indicator
    item.mkdir(parents=True)
    (item / "meta.json").write_text(
        json.dumps({"id": indicator, "units": ["процент"], "title": "тест"}),
        encoding="utf-8",
    )
    rows = [
        ("country", "643", "28", "year", 2019, 10.0),
        ("country", "643", "28", "year", 2020, 12.0),
        ("country", "643", "28", "year", 2021, 15.0),
        ("country", "643", "29", "year", 2019, 5.0),
        ("country", "643", "29", "year", 2020, 6.0),
        ("region", "4500000000", "28", "year", 2019, 7.0),
        ("country", "643", "28", "quarter", 2019, 99.0),
    ]
    lines = ["level,okato,okved,unit,extra,period,kind,year,value"]
    for level, okato, okved, kind, year, value in rows:
        lines.append(f"{level},{okato},{okved},384,,1,{kind},{year},{value}")
    (item / "panel.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_load_series_filters_level_and_okved(tmp_path: Path) -> None:
    root = _make_panel(tmp_path)
    series = load_series(root, "123", "28")
    assert series.years() == [2019, 2020, 2021]
    assert series.values[2020] == 12.0
    assert series.unit == "процент"


def test_load_series_kind_filter(tmp_path: Path) -> None:
    root = _make_panel(tmp_path)
    series = load_series(root, "123", "28", kind="quarter")
    assert series.years() == [2019]
    assert series.values[2019] == 99.0


def test_load_series_missing_raises(tmp_path: Path) -> None:
    root = _make_panel(tmp_path)
    with pytest.raises(ValueError):
        load_series(root, "123", "99")
    with pytest.raises(FileNotFoundError):
        load_series(root, "999", "28")


def test_load_series_set_and_meta(tmp_path: Path) -> None:
    root = _make_panel(tmp_path)
    series = load_series_set(root, "123", ["28", "29"])
    assert sorted(series) == ["28", "29"]
    assert series["29"].mean([2019, 2020, 2021]) == pytest.approx(5.5)


def test_yoy_growth_skips_gaps() -> None:
    from graphcreativity.data import Series

    series = Series(
        indicator="x",
        okved="28",
        level="country",
        values={2019: 10.0, 2020: 12.0, 2022: 18.0},
    )
    growth = yoy_growth(series)
    assert growth == {2020: pytest.approx(0.2)}


def test_pearson_perfect() -> None:
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 2], [1, 2]) is None


def test_resolve_root_explicit(tmp_path: Path) -> None:
    root = _make_panel(tmp_path)
    assert resolve_root(tmp_path / "config.yaml", root) == root
