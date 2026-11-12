"""Загрузка панелей ЕМИСС для калибровки отраслевой модели.

Панели лежат вне репозитория движка, в каталоге вида
``data/rosstat/by-id/<id>/panel.csv``. Схема панели::

    level,okato,okved,unit,extra,period,kind,year,value

Здесь нет расчётов модели: только чтение, фильтрация и мелкие
статистики над рядом. Калибровка --- в ``scripts/calibrate_industry.py``.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from statistics import mean

__all__ = [
    "Series",
    "load_meta",
    "load_series",
    "load_series_set",
    "panel_path",
    "pearson",
    "resolve_root",
    "yoy_growth",
]


@dataclass(frozen=True)
class Series:
    """Один ряд «показатель x ОКВЭД x уровень»."""

    indicator: str
    okved: str
    level: str
    values: Mapping[int, float]
    unit: str = ""
    meta: Mapping[str, object] = field(default_factory=dict)

    def years(self) -> list[int]:
        return sorted(self.values)

    def get(self, year: int) -> float | None:
        return self.values.get(int(year))

    def mean(self, years: Iterable[int]) -> float | None:
        values = [self.values[int(y)] for y in years if int(y) in self.values]
        return mean(values) if values else None

    def growth(self, year_from: int, year_to: int) -> float | None:
        start = self.values.get(int(year_from))
        end = self.values.get(int(year_to))
        if start in (None, 0) or end is None:
            return None
        return end / start - 1.0


def resolve_root(
    config_path: str | Path,
    explicit: str | Path | None = None,
) -> Path:
    """Найти корень ``data/rosstat``.

    Порядок: явный аргумент, ``GC_DATA_ROOT``, затем кандидаты рядом с
    репозиторием движка. Ошибка --- если ни один путь не существует.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("GC_DATA_ROOT")
    if env:
        candidates.append(Path(env).expanduser())

    config_dir = Path(config_path).resolve().parent
    # configs/ -> graph-creativity -> repos -> devel -> docs/reports/master
    for parent in (config_dir, *config_dir.parents):
        candidates.append(parent / "docs" / "reports" / "master" / "data" / "rosstat")
        candidates.append(parent / "data" / "rosstat")
    candidates.append(Path.home() / "docs" / "reports" / "master" / "data" / "rosstat")

    for candidate in candidates:
        if (candidate / "by-id").is_dir():
            return candidate
    tried = ", ".join(str(c) for c in candidates[:3])
    raise FileNotFoundError(
        f"не найден каталог data/rosstat (искал: {tried}, ...); "
        "задайте --data-root или GC_DATA_ROOT"
    )


def panel_path(root: str | Path, indicator_id: str) -> Path:
    return Path(root) / "by-id" / str(indicator_id) / "panel.csv"


def load_meta(root: str | Path, indicator_id: str) -> dict:
    path = Path(root) / "by-id" / str(indicator_id) / "meta.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _unit_of(meta: Mapping[str, object], fallback: str = "") -> str:
    units = meta.get("units") or []
    if isinstance(units, str):
        return units
    return str(units[0]) if units else fallback


def load_series(
    root: str | Path,
    indicator_id: str,
    okved: str,
    *,
    level: str = "country",
    kind: str | None = None,
    period: str | int | None = None,
    meta: Mapping[str, object] | None = None,
) -> Series:
    """Прочитать один ряд и проверить, что он непустой.

    ``kind`` и ``period`` --- необязательные фильтры: у квартальных
    рядов (``58090``) ``kind=quarter``, у ряда по хозяйственным видам
    деятельности (``57715``) годовой период кодируется числом 118.
    """
    path = panel_path(root, indicator_id)
    if not path.exists():
        raise FileNotFoundError(
            f"нет панели {indicator_id}: {path}. "
            "Скачайте её через tools/fedstat_grid.mjs или tools/fedstat.py."
        )

    okved = str(okved)
    period = None if period is None else str(period)
    values: dict[int, float] = {}
    unit = ""
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("level") != level:
                continue
            if str(row.get("okved")) != okved:
                continue
            if kind is not None and row.get("kind") != kind:
                continue
            if period is not None and str(row.get("period")) != period:
                continue
            raw = row.get("value")
            if raw in (None, ""):
                continue
            values[int(row["year"])] = float(raw)
            unit = unit or row.get("unit", "")

    info = dict(meta) if meta is not None else load_meta(root, indicator_id)
    series = Series(
        indicator=str(indicator_id),
        okved=okved,
        level=level,
        values=values,
        unit=_unit_of(info, unit),
        meta=info,
    )
    if not values:
        raise ValueError(
            f"ряд {indicator_id} пуст для ОКВЭД {okved}, level={level}, "
            f"kind={kind}, period={period}"
        )
    return series


def load_series_set(
    root: str | Path,
    indicator_id: str,
    okveds: Sequence[str],
    **kwargs,
) -> dict[str, Series]:
    meta = kwargs.pop("meta", None)
    if meta is None:
        meta = load_meta(root, indicator_id)
    return {
        str(code): load_series(root, indicator_id, code, meta=meta, **kwargs)
        for code in okveds
    }


def yoy_growth(series: Series, years: Sequence[int] | None = None) -> dict[int, float]:
    """Темпы прироста к предыдущему году внутри ряда."""
    years = sorted(years) if years is not None else series.years()
    result: dict[int, float] = {}
    for previous, current in pairwise(years):
        if current - previous != 1:
            continue
        value = series.growth(previous, current)
        if value is not None:
            result[current] = value
    return result


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Коэффициент корреляции Пирсона без numpy."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = sum(v * v for v in dx) ** 0.5
    sy = sum(v * v for v in dy) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)
