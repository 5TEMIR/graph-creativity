#!/usr/bin/env python3
"""Импульсный граф отраслевой модели: кто на кого влияет и когда.

Три вида импульса:

* фактор/исход (``i``, ``o``) --- начальное состояние +``size`` в месяц 0;
* мера (``p``) --- включена с нулевого месяца на всю кампанию;
* среда (``y``) --- ступенька +``size`` на все 60 месяцев.

Отклик --- отклонение от базового прогона (без мер, среда на нуле).
Главные величины --- месяц первого отклика (лаг) и пиковое отклонение.
Считается в двух режимах: при рабочем пороге ``theta = 0,35`` и при
``theta = 0`` (структура связей без порогового фильтра).

Пишет ``impulse.csv``, ``threshold.csv``, ``summary.md`` и тепловую
карту ``impulse_heatmap.png``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphcreativity.engine import SimOptions
from graphcreativity.impulse import (
    Response,
    respond,
    run_baseline,
    run_environment_shock,
    run_measure,
    run_state_shock,
)
from graphcreativity.model import Model, load_model

DEFAULT_CONFIG = ROOT / "configs" / "industry.yaml"
DEFAULT_OUT = ROOT / "out" / "industry-impulse"

HORIZON = 60
IMPULSES = tuple(round(0.05 * step, 2) for step in range(1, 21))
ENV_IMPULSES = (0.5, 1.0)
THETAS = {"model": None, "zero": 0.0}
TOL = 0.01


@dataclass(frozen=True)
class Run:
    theta: str
    kind: str
    source: str
    impulse: float
    response: Response


def _fmt(value: float, digits: int = 2) -> str:
    if abs(value) < 0.5 * 10**-digits:
        return "0"
    return f"{value:.{digits}f}".replace(".", ",")


def _lag(value: int | None) -> str:
    return "—" if value is None else str(value)


def _collect(model: Model) -> list[Run]:
    runs: list[Run] = []
    targets = model.indicator_ids
    for theta, value in THETAS.items():
        options = SimOptions(horizon=HORIZON, theta=value)
        base = run_baseline(model, options)
        for node in model.indicator_ids:
            for size in IMPULSES:
                result = run_state_shock(model, node, size, options)
                for item in respond(node, result, base, targets, tol=TOL):
                    runs.append(Run(theta, "фактор", node, size, item))
        for measure in model.measure_ids:
            result = run_measure(model, measure, options)
            for item in respond(measure, result, base, targets, tol=TOL):
                runs.append(Run(theta, "мера", measure, 1.0, item))
        for node in model.environment_ids:
            for size in ENV_IMPULSES:
                result = run_environment_shock(model, node, size, options)
                for item in respond(node, result, base, targets, tol=TOL):
                    runs.append(Run(theta, "среда", node, size, item))
    return runs


def _pick(
    runs: list[Run],
    theta: str,
    *,
    kind: str | None = None,
    impulse: float | None = None,
) -> dict[tuple[str, str], Response]:
    table: dict[tuple[str, str], Response] = {}
    for run in runs:
        if run.theta != theta:
            continue
        if kind is not None and run.kind != kind:
            continue
        if impulse is not None and abs(run.impulse - impulse) > 1e-9:
            continue
        table[(run.source, run.response.target)] = run.response
    return table


def _structure(
    runs: list[Run],
    theta: str,
    factor_impulse: float,
) -> dict[tuple[str, str], Response]:
    """Таблица структуры: импульс фактора задан, мера и среда — свои."""
    table: dict[tuple[str, str], Response] = {}
    for run in runs:
        if run.theta != theta:
            continue
        if run.kind == "фактор" and abs(run.impulse - factor_impulse) > 1e-9:
            continue
        if run.kind == "среда" and abs(run.impulse - 1.0) > 1e-9:
            continue
        table[(run.source, run.response.target)] = run.response
    return table


def _lag_table(
    sources: list[str],
    targets: list[str],
    table: dict[tuple[str, str], Response],
) -> list[str]:
    header = "| Источник | " + " | ".join(targets) + " |"
    rule = "|---|" + "---|" * len(targets)
    lines = [header, rule]
    for source in sources:
        cells: list[str] = []
        for target in targets:
            if source == target:
                cells.append("·")
                continue
            item = table.get((source, target))
            cells.append(_lag(item.first_month if item else None))
        lines.append(f"| `{source}` | " + " | ".join(cells) + " |")
    return lines


def _value_table(
    sources: list[str],
    targets: list[str],
    table: dict[tuple[str, str], Response],
) -> list[str]:
    header = "| Источник | " + " | ".join(targets) + " |"
    rule = "|---|" + "---|" * len(targets)
    lines = [header, rule]
    for source in sources:
        cells = [
            "·" if source == target else _fmt(table[(source, target)].peak)
            if (source, target) in table
            else "0"
            for target in targets
        ]
        lines.append(f"| `{source}` | " + " | ".join(cells) + " |")
    return lines


def _thresholds(
    runs: list[Run], model: Model
) -> list[tuple[str, float | None, int, str]]:
    """Минимальный импульс, который двигает хотя бы одну другую вершину."""
    rows: list[tuple[str, float | None, int, str]] = []
    for node in model.indicator_ids:
        best: float | None = None
        reached: set[str] = set()
        first = ""
        for run in runs:
            if run.theta != "model" or run.kind != "фактор" or run.source != node:
                continue
            if run.response.target == node or run.response.peak <= TOL:
                continue
            reached.add(run.response.target)
            if best is None or run.impulse < best:
                best = run.impulse
                first = run.response.target
        rows.append((node, best, len(reached), first))
    return rows


def _write_csv(path: Path, runs: list[Run]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "theta",
                "kind",
                "source",
                "impulse",
                "target",
                "peak",
                "peak_month",
                "final",
                "first_month",
                "sign",
            ]
        )
        for run in runs:
            item = run.response
            writer.writerow(
                [
                    run.theta,
                    run.kind,
                    run.source,
                    f"{run.impulse:.2f}",
                    item.target,
                    f"{item.peak:.6f}",
                    item.peak_month,
                    f"{item.final:.6f}",
                    "" if item.first_month is None else item.first_month,
                    item.sign,
                ]
            )


def _write_thresholds(
    path: Path, rows: list[tuple[str, float | None, int, str]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "min_impulse", "targets_reached", "first_target"])
        for source, value, reached, target in rows:
            writer.writerow(
                [source, "" if value is None else f"{value:.2f}", reached, target]
            )


def _heatmap(
    path: Path,
    sources: list[str],
    targets: list[str],
    zero: dict[tuple[str, str], Response],
    model: dict[tuple[str, str], Response],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def matrix(table: dict[tuple[str, str], Response]) -> np.ndarray:
        values = np.full((len(sources), len(targets)), np.nan)
        for row, source in enumerate(sources):
            for column, target in enumerate(targets):
                if source == target:
                    continue
                item = table.get((source, target))
                if item is not None and item.first_month is not None:
                    values[row, column] = item.first_month
        return values

    panels = [
        (matrix(zero), "θ = 0: структура, импульс 0,1", "viridis_r"),
        (matrix(model), "θ = 0,35: импульс 1,0", "viridis_r"),
    ]
    limit = max(np.nanmax(values) for values, _, _ in panels)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4), sharey=True)
    for axis, (values, title, cmap_name) in zip(axes, panels, strict=True):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e8e8e8")
        image = axis.imshow(
            values,
            cmap=cmap,
            vmin=0.0,
            vmax=limit,
            aspect="auto",
        )
        axis.set_xticks(range(len(targets)))
        axis.set_xticklabels(targets)
        axis.set_yticks(range(len(sources)))
        axis.set_yticklabels(sources)
        axis.set_title(title, fontsize=10)
        for row in range(len(sources)):
            for column in range(len(targets)):
                value = values[row, column]
                if not np.isnan(value):
                    axis.text(
                        column,
                        row,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > 0.6 * limit else "black",
                    )
                elif row == column:
                    axis.text(
                        column,
                        row,
                        "×",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="#666666",
                    )
    fig.colorbar(image, ax=axes, shrink=0.8, label="месяц первого отклика")
    fig.suptitle(
        "Импульсный граф: месяц первого отклика; "
        "× — сама вершина, серое — отклика нет",
        fontsize=11,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _findings(
    model: Model,
    zero: dict[tuple[str, str], Response],
    model_table: dict[tuple[str, str], Response],
    measures: dict[tuple[str, str], Response],
    environment: dict[tuple[str, str], Response],
    thresholds: list[tuple[str, float | None, int, str]],
) -> list[str]:
    findings: list[str] = []
    pairs = [
        (source, target)
        for source in model.indicator_ids
        for target in model.indicator_ids
        if source != target
    ]

    def alive(table: dict[tuple[str, str], Response]) -> int:
        return sum(
            1
            for key in pairs
            if key in table and table[key].peak > TOL
        )

    findings.append(
        f"1. Вне диагонали при θ = 0 работает {alive(zero)} пар "
        f"«источник–цель» из {len(pairs)}, при θ = 0,35 — {alive(model_table)}. "
        "Порог отсекает почти весь граф."
    )

    env_pairs = [
        (source, target)
        for (source, target), item in environment.items()
        if item.peak > TOL
    ]
    if not env_pairs:
        findings.append(
            "2. Среда не двигает ничего при ступеньке до +1,0: дуги "
            "`y1→i2` (0,35) и `y2→i3` (0,30) равны порогам ровно, поэтому "
            "`w·y − θ = 0`. Это не запас прочности, а ноль: модель "
            "оживает только от мер. Исключение --- класс 33 в "
            "ретроспективе, где инвестиции выросли на 125 % и `y1` "
            "перешёл порог."
        )
    else:
        findings.append(
            "2. Среда даёт отклик в парах: "
            + ", ".join(f"`{source}→{target}`" for source, target in env_pairs)
            + "."
        )

    sinks = [
        node
        for node in model.indicator_ids
        if not any(
            edge.src == node and not edge.optional for edge in model.edges
        )
    ]
    if sinks:
        findings.append(
            "3. Тупики графа: "
            + ", ".join(f"`{node}`" for node in sinks)
            + ". Исходящих дуг нет; обратные связи `o1→i2` и `o3→i2` "
            "опциональны и в базовом прогоне выключены. Поэтому прибыль и "
            "выпуск не возвращаются в НИОКР, а самоподдерживающегося роста "
            "в базовой модели нет по построению."
        )

    passed = [row for row in thresholds if row[1] is not None]
    if passed:
        findings.append(
            "4. Минимальный импульс, который проходит порог: "
            + "; ".join(
                f"`{source}` — {_fmt(value)}" for source, value, _, _ in passed
            )
            + ". Всё, что слабее, гаснет: отклик нелинейный."
        )

    top_measures = sorted(
        (
            (
                measure,
                sum(
                    measures[(measure, target)].peak
                    for target in model.indicator_ids
                    if (measure, target) in measures
                ),
            )
            for measure in model.measure_ids
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    coverage = []
    for measure in model.measure_ids:
        reached = [
            target
            for target in model.indicator_ids
            if (measure, target) in measures
            and measures[(measure, target)].peak > TOL
        ]
        coverage.append(
            f"`{measure}` — {len(reached)} ("
            + ", ".join(f"`{target}`" for target in reached)
            + ")"
        )
    findings.append(
        "5. Меры по сумме пиковых откликов: "
        + ", ".join(f"`{name}` — {_fmt(value)}" for name, value in top_measures)
        + ". Охват: "
        + "; ".join(coverage)
        + "."
    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    model = load_model(args.config)
    runs = _collect(model)
    thresholds = _thresholds(runs, model)
    args.out.mkdir(parents=True, exist_ok=True)

    _write_csv(args.out / "impulse.csv", runs)
    _write_thresholds(args.out / "threshold.csv", thresholds)

    sources = list(model.indicator_ids)
    zero = _structure(runs, "zero", 0.1)
    model_table = _structure(runs, "model", 1.0)
    measures = _pick(runs, "model", kind="мера")
    environment = _pick(runs, "model", kind="среда", impulse=1.0)
    _heatmap(
        args.out / "impulse_heatmap.png",
        sources,
        model.indicator_ids,
        zero,
        model_table,
    )

    lines = [
        "# Импульсный граф отрасли",
        "",
        "Отклик — отклонение от базового прогона (меры выключены, среда на",
        "нуле). Импульс по фактору — начальное состояние в месяц 0; по мере —",
        "включение с нулевого месяца; по среде — ступенька на все 60 месяцев.",
        "Это проверка структуры графа, а не прогноз и не калибровка.",
        "",
        "## 1. Структура при θ = 0, импульс 0,1",
        "",
        "Порог снят, виден весь граф факторов и исходов. Ячейка — номер",
        "месяца первого отклика; точка — сама вершина, прочерк — отклика нет.",
        "",
        *_lag_table(sources, model.indicator_ids, zero),
        "",
        "## 2. Рабочий режим θ = 0,35, импульс 1,0",
        "",
        "Тот же граф, но с порогом; импульс максимальный. Видно, что в",
        "рабочем режиме проходят только сильные дуги; каскад запаздывает,", 
        "потому что промежуточной вершине нужно сначала самой пройти порог.",
        "",
        *_lag_table(sources, model.indicator_ids, model_table),
        "",
        "## 3. Минимальный импульс, который двигает другую вершину",
        "",
        "| Источник | Минимальный импульс | Достигнуто вершин | Первая цель |",
        "|---|---|---|---|",
    ]
    for source, value, reached, target in thresholds:
        shown = "не проходит" if value is None else _fmt(value)
        lines.append(f"| `{source}` | {shown} | {reached} | {target or '---'} |")

    lines += [
        "",
        "## 4. Меры: пиковый отклик и лаг",
        "",
        "| Мера | " + " | ".join(model.indicator_ids) + " |",
        "|---|" + "---|" * len(model.indicator_ids),
    ]
    for measure in model.measure_ids:
        cells = []
        for target in model.indicator_ids:
            item = measures.get((measure, target))
            cells.append(
                "0" if item is None or item.peak <= TOL
                else f"{_fmt(item.peak)} / {_lag(item.first_month)}"
            )
        lines.append(f"| `{measure}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Формат: пиковый отклик / месяц. Полные ряды — `impulse.csv`.",
        "",
        "## 5. Среда: ступенька +1,0",
        "",
        "| Среда | " + " | ".join(model.indicator_ids) + " |",
        "|---|" + "---|" * len(model.indicator_ids),
    ]
    for env in model.environment_ids:
        cells = [
            _fmt(environment[(env, target)].peak)
            if (env, target) in environment
            else "0"
            for target in model.indicator_ids
        ]
        lines.append(f"| `{env}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 6. Пиковые отклики при θ = 0,35",
        "",
        *_value_table(sources, model.indicator_ids, model_table),
        "",
        "## 7. Что видно",
        "",
        *_findings(model, zero, model_table, measures, environment, thresholds),
        "",
        "Карта лагов — `impulse_heatmap.png`.",
        "",
    ]
    text = "\n".join(lines)
    (args.out / "summary.md").write_text(text, encoding="utf-8")

    active = sum(
        1
        for run in runs
        if run.theta == "zero" and run.response.peak > TOL
    )
    print(f"прогонов: {len(runs)}; ненулевых откликов при θ=0: {active}")
    print(f"отчёт: {args.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
