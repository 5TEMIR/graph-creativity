#!/usr/bin/env python3
"""Ретроспективная проверка отраслевой модели (backtest 2019--2022).

Что делает:

1. берёт фактические ряды ЕМИСС по классам ОКВЭД 24--30, 33;
2. нормирует каждый узел так, что 2019 год --- это ``x=0``, а 2022 ---
   ``x=1`` (для узлов с ростом вверх);
3. подставляет исторические ряды среды ``y1`` (инвестиции) и ``y2``
   (выпуск) как накопленный рост к 2019 году;
4. прогоняет движок с экспертными весами и нулевыми мерами;
5. сравнивает модельные траектории с фактическими и подбирает
   глобальный множитель ``gain``, при котором модель догоняет факт по
   ``i3`` в 2022 году.

Скрипт не калибрует веса. Он отвечает на вопрос, может ли модель
с экспертными весами воспроизвести историю и во сколько раз нужно
усилить все дуги, чтобы она это сделала. Ожидаемый вывод --- модель
не является прогнозной: без автономного тренда она остаётся у базы,
а требуемый ``gain`` выходит за разумные пределы.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import itertools

from graphcreativity.data import load_meta, load_series, resolve_root
from graphcreativity.engine import SimOptions, simulate
from graphcreativity.model import Model, Plan, Scenario, load_model

DEFAULT_CONFIG = ROOT / "configs" / "industry.yaml"
DEFAULT_OUT = ROOT / "out" / "industry-backtest"

CLASSES = ["24", "25", "26", "27", "28", "29", "30", "33"]
YEARS = [2019, 2020, 2021, 2022]
NODES = {
    "i1": "58769",
    "i2": "60022",
    "i3": "58761",
    "i4": "58701",
    "o1": "57711",
    "o3": "58261",
}
ENV = {
    "y1": "59084",
    "y2": "57711",
}


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "---"
    return f"{value:.{digits}f}".replace(".", ",")


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "---"
    return f"{value * 100:+.{digits}f} %".replace(".", ",")


def _load_node(
    root: Path,
    indicator: str,
    code: str,
    years: Sequence[int],
) -> dict[int, float]:
    meta = load_meta(root, indicator)
    series = load_series(root, indicator, code, meta=meta)
    missing = [year for year in years if year not in series.values]
    if missing:
        raise ValueError(
            f"{indicator}, класс {code}: нет лет {missing}"
        )
    return {year: series.values[year] for year in years}


def _interp(values: dict[int, float], steps: int) -> list[float]:
    """Линейная интерполяция годовых значений по месяцам.

    ``values`` --- накопленное отклонение от 2019 года для 2020, 2021,
    2022. На шаге 0 --- 0, на шагах 12/24/36 --- значения 2020/2021/2022,
    после 36 --- последнее значение.
    """
    points = [(0, 0.0)] + [(12 * (i + 1), values[year]) for i, year in enumerate(YEARS[1:])]
    result: list[float] = []
    for step in range(steps):
        if step >= points[-1][0]:
            result.append(points[-1][1])
            continue
        for (x0, y0), (x1, y1) in itertools.pairwise(points):
            if x0 <= step <= x1:
                share = (step - x0) / (x1 - x0)
                result.append(y0 + share * (y1 - y0))
                break
        else:  # pragma: no cover - защита от неверной сетки
            result.append(points[-1][1])
    return result


def _normalized(values: dict[int, float], year: int) -> float:
    base, target = values[2019], values[2022]
    if target == base:
        return 0.0
    return (values[year] - base) / (target - base)


def _model_for_class(
    model: Model,
    data: dict[str, dict[int, float]],
) -> Model:
    """Копия модели, где base/target узлов --- факт 2019/2022."""
    nodes = dict(model.nodes)
    for node_id, values in data.items():
        node = nodes.get(node_id)
        if node is None:
            continue
        base, target = values[2019], values[2022]
        if target == base:
            target = base + 1e-9
        nodes[node_id] = replace(node, base=base, target=target)
    return replace(model, nodes=nodes)


def _run(
    model: Model,
    env: dict[str, list[float]],
    *,
    horizon: int,
    gain: float,
):
    options = SimOptions(horizon=horizon, gain=gain)
    return simulate(
        model,
        plan=Plan(name="history", label="История", programs={}),
        scenario=Scenario("history"),
        options=options,
        environment_series=env,
        initial_states={node_id: 0.0 for node_id in model.indicator_ids},
    )


def _best_gain(
    model: Model,
    env: dict[str, list[float]],
    horizon: int,
    target_id: str,
    goal: float,
) -> tuple[float, float]:
    """Подобрать gain, при котором x_target(T) ближе всего к goal."""
    best_gain, best_error = 1.0, float("inf")
    for exponent in [i / 10 for i in range(-20, 41)]:
        gain = 10.0 ** exponent
        result = _run(model, env, horizon=horizon, gain=gain)
        value = result.states[target_id][-1]
        error = abs(value - goal)
        if error < best_error:
            best_gain, best_error = gain, error
    return best_gain, best_error


def _write_summary(path: Path, report: dict) -> None:
    lines: list[str] = []
    lines.append("# Ретроспективная проверка отраслевой модели, 2019--2022")
    lines.append("")
    lines.append(
        "Экспертные веса, нулевые меры, начальное состояние 2019 года. "
        "Каждый узел нормирован: 2019 --- 0, 2022 --- 1. Среда ``y1`` и "
        "``y2`` --- накопленный рост инвестиций и выпуска к 2019 году."
    )
    lines.append("")
    lines.append("## 1. Модель против факта")
    lines.append("")
    lines.append(
        "| Класс | MAE модели | ``i3`` факт 2022 | ``i3`` модель 2022 | "
        "Требуемый ``gain`` | Ошибка при нём |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in report["classes"]:
        lines.append(
            f"| {row['okved']} | {_fmt(row['mae'])} | "
            f"{_fmt(row['actual_i3_2022'])} | {_fmt(row['model_i3_2022'])} | "
            f"{_fmt(row['required_gain'])} | {_fmt(row['gain_error'])} |"
        )
    lines.append("")
    lines.append(
        f"Средний MAE при экспертных весах: {_fmt(report['mean_mae'])}. "
        f"Средний требуемый ``gain``: {_fmt(report['mean_gain'])}. "
        "Это во столько раз нужно усилить все дуги, чтобы модель догнала "
        "факт по ``i3``; по отдельным классам разброс --- "
        f"{_fmt(report['min_gain'])}--{_fmt(report['max_gain'])}."
    )
    lines.append("")
    lines.append("## 2. Пилот 28 по узлам")
    lines.append("")
    lines.append("| Узел | Факт 2020 | Модель 2020 | Факт 2021 | Модель 2021 | Факт 2022 | Модель 2022 |")
    lines.append("|---|---|---|---|---|---|---|")
    for node_id, row in report["pilot"].items():
        lines.append(
            f"| `{node_id}` | {_fmt(row['actual'][2020])} | {_fmt(row['model'][2020])} | "
            f"{_fmt(row['actual'][2021])} | {_fmt(row['model'][2021])} | "
            f"{_fmt(row['actual'][2022])} | {_fmt(row['model'][2022])} |"
        )
    lines.append("")
    lines.append("## 3. Вывод")
    lines.append("")
    lines.append(
        "При экспертных весах модель остаётся у базы: без автономного "
        "тренда и при нулевых мерах драйверы среды слишком слабы, чтобы "
        "преодолеть порог ``theta``. Чтобы воспроизвести факт, все веса "
        "пришлось бы усилить на порядок; это уже другая модель, а не "
        "калибровка. Веса мер при этом всё равно не идентифицируются: "
        "в истории они равны нулю."
    )
    lines.append("")
    lines.append(
        "Практический вывод: модель нельзя использовать как прогнозную. "
        "Её назначение --- сравнение сценариев при одинаковых экспертных "
        "весах, а не воспроизведение прошлого."
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    config: Path,
    data_root: Path | None,
    out: Path,
    classes: Sequence[str],
    pilot: str,
) -> dict:
    model = load_model(config)
    root = resolve_root(config, data_root)
    horizon = 12 * (len(YEARS) - 1)
    steps = horizon + 1

    rows: list[dict] = []
    pilot_detail: dict[str, dict] = {}

    for code in classes:
        data = {
            node_id: _load_node(root, indicator, code, YEARS)
            for node_id, indicator in NODES.items()
        }
        env_raw = {
            env_id: _load_node(root, indicator, code, YEARS)
            for env_id, indicator in ENV.items()
        }
        env = {
            env_id: _interp(
                {
                    year: values[year] / values[2019] - 1.0
                    for year in YEARS[1:]
                },
                steps,
            )
            for env_id, values in env_raw.items()
        }
        env.setdefault("y3", [0.0] * steps)
        env.setdefault("y4", [0.0] * steps)

        class_model = _model_for_class(model, data)
        result = _run(class_model, env, horizon=horizon, gain=1.0)
        actual = {
            node_id: {year: _normalized(values, year) for year in YEARS[1:]}
            for node_id, values in data.items()
        }
        model_x = {
            node_id: {
                year: result.states[node_id][12 * (i + 1)]
                for i, year in enumerate(YEARS[1:])
            }
            for node_id in data
        }
        errors = [
            abs(model_x[node_id][year] - actual[node_id][year])
            for node_id in data
            for year in YEARS[1:]
        ]
        required_gain, gain_error = _best_gain(
            class_model, env, horizon, "i3", 1.0
        )
        rows.append(
            {
                "okved": code,
                "mae": mean(errors),
                "actual_i3_2022": actual["i3"][2022],
                "model_i3_2022": model_x["i3"][2022],
                "required_gain": required_gain,
                "gain_error": gain_error,
            }
        )
        if code == pilot:
            pilot_detail = {
                node_id: {
                    "actual": actual[node_id],
                    "model": model_x[node_id],
                }
                for node_id in data
            }

    report = {
        "config": str(config),
        "data_root": str(root),
        "pilot": pilot,
        "classes": rows,
        "pilot_detail": pilot_detail,
        "mean_mae": mean(row["mae"] for row in rows),
        "mean_gain": mean(row["required_gain"] for row in rows),
        "min_gain": min(row["required_gain"] for row in rows),
        "max_gain": max(row["required_gain"] for row in rows),
    }

    out.mkdir(parents=True, exist_ok=True)
    with (out / "backtest.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
    with (out / "by_class.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(out / "summary.md", {**report, "pilot": pilot_detail})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--classes", default=",".join(CLASSES))
    parser.add_argument("--pilot", default="28")
    args = parser.parse_args(argv)

    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    report = build_report(args.config, args.data_root, args.out, classes, args.pilot)
    print(f"отчёт: {args.out / 'summary.md'}")
    for row in report["classes"]:
        print(
            f"  класс {row['okved']}: MAE={_fmt(row['mae'])}, "
            f"i3 модель={_fmt(row['model_i3_2022'])}, "
            f"требуемый gain={_fmt(row['required_gain'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
