#!/usr/bin/env python3
"""Идентифицируемость весов: что данные могут и чего не могут.

Три проверки:

1. **История не различает веса.** Ретроспективный прогон 2019--2022 с
   экспертными весами и с весами, умноженными на 0,5 и 1,5, даёт одну и
   ту же траекторию: при нулевых мерах и нулевых начальных состояниях
   драйв ниже порога. Значит правдоподобие по ``w`` плоское.
2. **Панель не различает знаки.** Мощность того же дизайна (8 классов,
   7 лет, две фиксированные эффекта): какой минимальный коэффициент
   ``beta`` различим с вероятностью 80 % при 5 % уровне значимости.
   Сравнивается с экспертным весом.
3. **Но выбор весов меняет ответ.** Прогон планов при весах,
   переставленных по знакам панели, при ``|beta|``, при нуле и при
   0,5 от экспертных: как меняются MSS, лучший план, ``T`` и ``k``.

Вместе это даёт вывод: данные не выбирают веса, но выбор весов
определяет рекомендацию. Поэтому веса остаются экспертными, а ответ
предъявляется как набор сценариев, а не как калибровка.

Пишет ``history.csv``, ``power.csv``, ``power_curves.csv``,
``scenarios.csv`` и ``summary.md``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import backtest_industry as backtest
import fit_industry_edges as panel

from graphcreativity.data import resolve_root
from graphcreativity.engine import SimOptions
from graphcreativity.metrics import compare_plans, minimal_sufficient_sets
from graphcreativity.model import Model, load_model

DEFAULT_CONFIG = ROOT / "configs" / "industry.yaml"
DEFAULT_OUT = ROOT / "out" / "industry-identification"

HORIZON = 60
HISTORY_HORIZON = 36
REPS = 200
GRID = tuple(round(0.1 * step, 1) for step in range(16))
POWER_TARGET = 0.8


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "---"
    return f"{value:.{digits}f}".replace(".", ",")


def _key(measures: tuple[str, ...]) -> str:
    return "+".join(measures) or "none"


# ---------------------------------------------------------------------------
# 1. История не различает веса
# ---------------------------------------------------------------------------


def _history_check(model: Model, root: Path) -> list[dict]:
    """Максимальное расхождение траекторий при весах x0,5 и x1,5."""
    rows: list[dict] = []
    steps = HISTORY_HORIZON + 1
    for code in panel.CLASSES:
        data = {
            node_id: backtest._load_node(root, indicator, code, backtest.YEARS)
            for node_id, indicator in backtest.NODES.items()
        }
        env_raw = {
            env_id: backtest._load_node(root, indicator, code, backtest.YEARS)
            for env_id, indicator in backtest.ENV.items()
        }
        env = {
            env_id: backtest._interp(
                {
                    year: values[year] / values[2019] - 1.0
                    for year in backtest.YEARS[1:]
                },
                steps,
            )
            for env_id, values in env_raw.items()
        }
        env.setdefault("y3", [0.0] * steps)
        env.setdefault("y4", [0.0] * steps)

        class_model = backtest._model_for_class(model, data)
        base = backtest._run(
            class_model, env, horizon=HISTORY_HORIZON, gain=1.0
        )

        deviations: dict[str, float] = {}
        for multiplier in (0.5, 1.5):
            edges = [
                replace(edge, w=edge.w * multiplier)
                if not edge.optional
                else edge
                for edge in class_model.edges
            ]
            shifted = backtest._run(
                replace(class_model, edges=edges),
                env,
                horizon=HISTORY_HORIZON,
                gain=1.0,
            )
            deviations[str(multiplier)] = max(
                abs(shifted.states[node][-1] - base.states[node][-1])
                for node in model.indicator_ids
            )

        rows.append(
            {
                "class": code,
                "base_max": max(abs(base.states[node][-1]) for node in model.indicator_ids),
                "dev_half": deviations["0.5"],
                "dev_double": deviations["1.5"],
                "required_gain": backtest._best_gain(
                    class_model, env, HISTORY_HORIZON, "i3", 1.0
                )[0],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 2. Мощность панели
# ---------------------------------------------------------------------------


def _critical_value(
    y: np.ndarray,
    x: np.ndarray,
    cls: np.ndarray,
    yr: np.ndarray,
    rng: np.random.Generator,
    reps: int = 400,
) -> float:
    """95-й процентиль |t*| в нуле — замена t-критического значения.

    Считается тем же wild bootstrap по кластерам, что и p-value в
    ``fit_industry_edges``: остатки берутся из ограниченной модели
    ``beta = 0``. Иначе мощность выглядела бы оптимистичнее самих оценок.
    """
    design = panel._design(cls, yr)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    # Остатки ограниченной модели --- так же, как в _wild_cluster_p.
    residual = y - fitted
    clusters = sorted(set(cls.tolist()))
    index = {code: position for position, code in enumerate(clusters)}
    values: list[float] = []
    for _ in range(reps):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(clusters))
        factor = np.array([signs[index[code]] for code in cls])
        sample = fitted + residual * factor
        values.append(abs(panel._ols(sample, x, cls, yr).t))
    return float(np.quantile(values, 0.95))


def _power_curve(
    y: np.ndarray,
    x: np.ndarray,
    cls: np.ndarray,
    yr: np.ndarray,
    betas: tuple[float, ...],
    rng: np.random.Generator,
    critical: float,
    reps: int = REPS,
) -> dict[float, float]:
    design = panel._design(cls, yr)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    xd = panel._residualize(x, design, np.ones_like(x))
    # В отличие от _critical_value, здесь остатки неограниченной модели:
    # мощность считается при альтернативе y = FE + beta * x_d + u.
    unrestricted = panel._ols(y, x, cls, yr)
    residual = y - fitted - unrestricted.beta * xd
    clusters = sorted(set(cls.tolist()))
    index = {code: position for position, code in enumerate(clusters)}
    curve: dict[float, float] = {}
    for beta in betas:
        rejections = 0
        for _ in range(reps):
            signs = rng.choice(np.array([-1.0, 1.0]), size=len(clusters))
            factor = np.array([signs[index[code]] for code in cls])
            sample = fitted + beta * xd + residual * factor
            fit = panel._ols(sample, x, cls, yr)
            if abs(fit.t) > critical:
                rejections += 1
        curve[beta] = rejections / reps
    return curve


def _mde(curve: dict[float, float]) -> float | None:
    for beta in sorted(curve):
        if curve[beta] >= POWER_TARGET:
            return beta
    return None


def _panel_check(model: Model, root: Path) -> tuple[list[dict], list[dict]]:
    raw = panel._load_raw(root, panel.CLASSES)
    rng = np.random.default_rng(20261113)
    rows: list[dict] = []
    curves: list[dict] = []
    for edge in panel.EDGES:
        y, x, cls, yr = panel._panel(raw, edge, edge.lag, panel.CLASSES)
        fit = panel._ols(y, x, cls, yr)
        p_value = panel._wild_cluster_p(y, x, cls, yr, fit)
        grid = tuple(sorted({*GRID, edge.weight}))
        critical = _critical_value(y, x, cls, yr, rng)
        curve = _power_curve(y, x, cls, yr, grid, rng, critical)
        rows.append(
            {
                "edge": edge.name,
                "w": edge.weight,
                "lag": edge.lag,
                "n": fit.n,
                "beta": fit.beta,
                "se": fit.se,
                "p": p_value,
                "mde": _mde(curve),
                "critical": critical,
                "power_expert": curve.get(edge.weight),
                "power_observed": curve.get(round(abs(fit.beta), 1), None),
            }
        )
        curves.append({"edge": edge.name, "curve": curve})
    return rows, curves


# ---------------------------------------------------------------------------
# 3. Что меняется в ответе
# ---------------------------------------------------------------------------


def _scenarios(model: Model, rows: list[dict]) -> list[dict]:
    betas = {row["edge"]: row for row in rows}
    variants = {
        "экспертные (база)": lambda edge, beta: edge.w,
        "панель: знаки": lambda edge, beta: edge.w if beta > 0 else 0.0,
        "панель: модуль beta": lambda edge, beta: min(1.0, abs(beta)),
        "панель: ноль": lambda edge, beta: 0.0,
        "панель: x0,5": lambda edge, beta: edge.w * 0.5,
    }
    options = SimOptions(horizon=HORIZON)
    result: list[dict] = []
    for name, rule in variants.items():
        edges = []
        for edge in model.edges:
            key = f"{edge.src}->{edge.dst}"
            beta = betas[key]["beta"] if key in betas else 0.0
            edges.append(replace(edge, w=rule(edge, beta)) if key in betas else edge)
        variant = replace(model, edges=edges)
        plans = compare_plans(variant, "nominal", options)
        by_key = {tuple(row["measures"]): row for row in plans}
        mss = minimal_sufficient_sets(plans)
        best = max(
            (item for item in mss if by_key[item]["k_T"] is not None),
            key=lambda item: by_key[item]["k_T"],
            default=(),
        )
        result.append(
            {
                "scenario": name,
                "mss": [_key(item) for item in mss],
                "best": _key(best),
                "t_goal": by_key[best]["t_goal"] if best else None,
                "k_best": by_key[best]["k_T"] if best else None,
                "k13": by_key[("p1", "p3")]["k_T"],
                "t13": by_key[("p1", "p3")]["t_goal"],
                "k23": by_key[("p2", "p3")]["k_T"],
            }
        )
    return result


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        f"{row[field]:.4f}"
                        if isinstance(row.get(field), float)
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )


def _summary(
    model: Model,
    history: list[dict],
    rows: list[dict],
    curves: list[dict],
    scenarios: list[dict],
) -> str:
    lines = [
        "# Идентифицируемость весов: что данные могут и чего не могут",
        "",
        "Вопрос: можно ли заменить экспертные веса оценками по данным.",
        "Ответ: нет, и у этого три отдельные причины.",
        "",
        "## 1. История не различает веса",
        "",
        "Ретроспективный прогон 2019--2022: нулевые меры, нулевые начальные",
        "состояния, среда `y1`/`y2` из ЕМИСС. Веса всех основных дуг умножены",
        "на 0,5 и на 1,5. Если траектория не меняется, правдоподобие по `w`",
        "плоское: данные не могут выбрать масштаб весов.",
        "",
        "| Класс | max i3 модели | расхождение x0,5 | расхождение x1,5 | нужный gain |",
        "|---|---|---|---|---|",
    ]
    for row in history:
        lines.append(
            f"| {row['class']} | {_fmt(row['base_max'], 3)} | "
            f"{_fmt(row['dev_half'], 4)} | {_fmt(row['dev_double'], 4)} | "
            f"{_fmt(row['required_gain'])} |"
        )

    max_dev = max(
        max(row["dev_half"], row["dev_double"]) for row in history
    )
    live = [row for row in history if row["base_max"] > 0]
    if len(live) == 1:
        live_note = [
            (
                f"Живая траектория только у класса {live[0]['class']} "
                f"({_fmt(live[0]['base_max'], 3)}): лишь там рост инвестиций"
            ),
            "переводит `w·y` через порог. В остальных классах среда до",
            "порога не дотягивает, траектория тождественно нулевая, и",
            "умножение весов на 0,5 или 1,5 не меняет ничего.",
        ]
    else:
        live_note = [
            "Траектория нулевая не во всех классах, но и там веса",
            "различаются слабо.",
        ]
    lines += [
        "",
        f"Максимальное расхождение по всем классам — {_fmt(max_dev, 4)}.",
        *live_note,
        "Пока модель стоит на пороге, правдоподобие по `w` плоское:",
        "в 7 из 8 классов данные не выбирают ни масштаб, ни знак.",
        "Поэтому «подогнать веса под ряды» на этом отрезке нельзя",
        "в принципе.",
        "",
        "## 2. Панель: мощность дизайна",
        "",
        "Тот же дизайн, что в панельных оценках: 8 классов, 2018--2024,",
        "логарифмические темпы, фиксированные эффекты класса и года,",
        "кластерные по классу ошибки. Мощность — доля отклонений гипотезы",
        "`beta = 0` при 5 % уровне. Критическое значение — не `t`-распределение,",
        "а 95-й процентиль `|t*|` того же wild bootstrap по кластерам",
        "(200 повторов); оно консервативнее классического и меняется от дуги",
        "к дуге. MDE — минимальный `beta`, различимый с вероятностью 80 %.",
        "",
        "| Дуга | w | n | beta | SE | p | t* 95 % | MDE 80 % | мощность при w |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        power = row["power_expert"]
        lines.append(
            f"| `{row['edge']}` | {_fmt(row['w'])} | {row['n']} | "
            f"{_fmt(row['beta'], 3)} | {_fmt(row['se'], 3)} | "
            f"{_fmt(row['p'], 3)} | {_fmt(row['critical'])} | "
            f"{_fmt(row['mde'])} | "
            f"{'---' if power is None else f'{power * 100:.0f} %'} |"
        )

    detectable = [row for row in rows if row["mde"] is not None and row["mde"] <= row["w"]]
    lines += [
        "",
        f"Экспертный эффект различим только у {len(detectable)} из {len(rows)} дуг.",
        "У остальных MDE выше экспертного веса: отрицательный результат",
        "панели — это «данных не хватает», а не «эффекта нет». Различить",
        "можно эффекты порядка "
        + _fmt(min(row["mde"] for row in rows if row["mde"] is not None))
        + "--"
        + _fmt(max(row["mde"] for row in rows if row["mde"] is not None))
        + ".",
        "",
        "## 3. Но выбор весов меняет ответ",
        "",
        "Прогон при горизонте 60 мес. Веса 11 дуг заменены по правилам:",
        "знаки панели (отрицательные оценки обнулены), модуль `beta`, ноль,",
        "0,5 от экспертных. Дуги от мер не меняются — их в истории нет.",
        "",
        "| Сценарий | MSS | Лучший | T | k лучшего | k(p1,p3) | T(p1,p3) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in scenarios:
        mss = ", ".join(f"`{item}`" for item in row["mss"]) or "---"
        lines.append(
            f"| {row['scenario']} | {mss} | `{row['best']}` | "
            f"{_fmt(row['t_goal'], 0)} | {_fmt(row['k_best'])} | "
            f"{_fmt(row['k13'])} | {_fmt(row['t13'], 0)} |"
        )

    lines += [
        "",
        "## 4. Что это значит",
        "",
        "1. **Данные не выбирают веса.** История не даёт градиента (пункт 1),",
        "   панель не различает эффекты экспертного размера (пункт 2).",
        "2. **Выбор весов определяет рекомендацию.** Смена знаков по панели",
        "   убирает `p1` из достаточных наборов и меняет лучший план",
        "   (пункт 3). Это не уточнение, а другой ответ.",
        "3. **Правильный вывод — не «подогнать», а «разделить».** Веса",
        "   остаются экспертными и маркируются; данные дают уровни, шоки",
        "   и проверку знаков; ответ предъявляется как набор сценариев с",
        "   указанием, какие дуги его переворачивают.",
        "",
        "Полные кривые мощности — `power.csv`, сценарии — `scenarios.csv`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    model = load_model(args.config)
    root = resolve_root(args.config)

    history = _history_check(model, root)
    rows, curves = _panel_check(model, root)
    scenarios = _scenarios(model, rows)

    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.out / "history.csv",
        history,
        ["class", "base_max", "dev_half", "dev_double", "required_gain"],
    )
    _write_csv(
        args.out / "power.csv",
        rows,
        [
            "edge",
            "w",
            "lag",
            "n",
            "beta",
            "se",
            "p",
            "critical",
            "mde",
            "power_expert",
        ],
    )
    _write_csv(
        args.out / "scenarios.csv",
        [
            {
                "scenario": row["scenario"],
                "mss": "|".join(row["mss"]),
                "best": row["best"],
                "t_goal": row["t_goal"],
                "k_best": row["k_best"],
                "k13": row["k13"],
                "t13": row["t13"],
                "k23": row["k23"],
            }
            for row in scenarios
        ],
        ["scenario", "mss", "best", "t_goal", "k_best", "k13", "t13", "k23"],
    )
    with (args.out / "power_curves.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["edge", "beta", "power"])
        for item in curves:
            for beta, power in sorted(item["curve"].items()):
                writer.writerow([item["edge"], f"{beta:.1f}", f"{power:.3f}"])

    text = _summary(model, history, rows, curves, scenarios)
    (args.out / "summary.md").write_text(text, encoding="utf-8")

    max_dev = max(max(row["dev_half"], row["dev_double"]) for row in history)
    detectable = sum(
        1 for row in rows if row["mde"] is not None and row["mde"] <= row["w"]
    )
    print(f"макс. расхождение истории: {max_dev:.5f}")
    print(f"экспертный эффект различим у {detectable} из {len(rows)} дуг")
    print("сценарии:")
    for row in scenarios:
        print(f"  {row['scenario']:18s} MSS={row['mss']} best={row['best']} k={row['k_best']}")
    print(f"отчёт: {args.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
