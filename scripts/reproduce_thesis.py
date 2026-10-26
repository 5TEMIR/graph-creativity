#!/usr/bin/env python3
"""Воспроизвести таблицы и числа, которые используются в тексте ВКР.

Запуск:
    .venv/bin/python scripts/reproduce_thesis.py
    .venv/bin/python scripts/reproduce_thesis.py --config configs/karkas.yaml

Скрипт пишет в ``out/thesis``:
    section5.tex           — таблица §5 (статика + динамика)
    theta_sensitivity.csv  — как меняются минимальные наборы и планы по θ
    plan_scenario_matrix.csv — именованные планы × сценарии
    recommendation.md      — короткая сводка для ЛПР

Числа в тексте должны вставляться из этих файлов, а не набираться руками.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphcreativity.engine import SimOptions
from graphcreativity.metrics import (
    compare_named_plans,
    compare_plans,
    critical_event_combinations,
    effect_strength,
    minimal_sufficient_sets,
    pareto_front,
    scenario_robustness,
)
from graphcreativity.model import load_model

SETS = [
    ("m1", ("m1",)),
    ("m3", ("m3",)),
    ("m4", ("m4",)),
    ("m3+m4", ("m3", "m4")),
    ("m1+m3", ("m1", "m3")),
    ("m1+m4", ("m1", "m4")),
    ("all", ("m1", "m3", "m4")),
]

THETAS = [0.30, 0.31, 0.35, 0.40, 0.42, 0.425]


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "---"
    return f"{value:.{digits}f}".replace(".", ",")


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _subset_label(measures: tuple[str, ...]) -> str:
    if not measures:
        return r"$\varnothing$"
    return r"$\{" + ",".join(
        f"m_{{{measure[1:]}}}" if measure.startswith("m") else measure
        for measure in measures
    ) + r"\}$"


def build_section5(model, options: SimOptions | None = None):
    rows = compare_plans(model, "nominal", options, include_empty=False)
    by_set = {tuple(row["measures"]): row for row in rows}
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Отладочный расчёт §5: статическая сила эффекта и динамика. "
        r"Конфиг: \texttt{" + _latex_escape(model.params.version) + r"}, "
        r"$\theta=" + _fmt(model.params.theta) + r"$, режим "
        r"\texttt{" + model.params.saturation + r"}.}",
        r"\label{tab:section5}",
        r"\begin{tabular}{@{}l c c c c c c c@{}}",
        r"\toprule",
        r"Набор & $\sigma_1$ & $x_1(T)$ & Рост, \% & $T_{\text{goal}}$ & $C$ & $k$ & Цель\\",
        r"\midrule",
    ]
    for label, measures in SETS:
        row = by_set.get(tuple(sorted(measures)))
        if row is None:
            continue
        sigma = effect_strength(model, measures)
        goal = "да" if row["reached"] else "нет"
        admissible = bool(row["reached"] and row["feasible"])
        lines.append(
            " & ".join(
                [
                    _subset_label(tuple(sorted(measures))),
                    _fmt(sigma),
                    _fmt(row["x_target_T"]),
                    _fmt(row["growth_pct_T"], 1),
                    _fmt(row["t_goal"], 0),
                    _fmt(row["cost_total"], 1),
                    _fmt(row["k_T"] if admissible else None, 3),
                    goal,
                ]
            )
            + r"\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\noindent $\sigma_1$ --- сумма путей длины $\le 2$ от мер к $X_1$ "
        r"(статическая величина). Остальные столбцы --- результат динамики; "
        r"они не обязаны монотонно следовать за $\sigma_1$. $k$ приведён "
        r"только для наборов, достигающих цели в пределах бюджета; прочерк "
        r"означает, что набор не является допустимым решением.",
        "",
    ]
    return "\n".join(lines)


def build_theta_table(model):
    lines = [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Чувствительность к порогу $\theta$: минимальные достаточные "
        r"наборы и времена достижения цели для ключевых наборов (все меры "
        r"стартуют с 0-го месяца).}",
        r"\label{tab:theta}",
        r"\begin{tabular}{@{}c l c c c c c@{}}",
        r"\toprule",
        r"$\theta$ & Минимальные наборы & $T_{\{m_1\}}$ & "
        r"$T_{\{m_1,m_4\}}$ & $T_{\{m_3,m_4\}}$ & $T_{\{m_1,m_3\}}$ & "
        r"$T_{\{m_1,m_3,m_4\}}$\\",
        r"\midrule",
    ]
    for theta in THETAS:
        options = SimOptions(theta=theta)
        rows = compare_plans(model, "nominal", options)
        mss = minimal_sufficient_sets(rows)
        by_set = {tuple(row["measures"]): row for row in rows}
        mss_text = ", ".join(_subset_label(tuple(items)) for items in mss) or "---"
        m1 = by_set.get(("m1",), {})
        m14 = by_set.get(("m1", "m4"), {})
        p1 = by_set.get(("m3", "m4"), {})
        hy = by_set.get(("m1", "m3"), {})
        full = by_set.get(("m1", "m3", "m4"), {})
        lines.append(
            f"{_fmt(theta, 3)} & {mss_text} & "
            f"{_fmt(m1.get('t_goal'), 0)} & {_fmt(m14.get('t_goal'), 0)} & "
            f"{_fmt(p1.get('t_goal'), 0)} & {_fmt(hy.get('t_goal'), 0)} & "
            f"{_fmt(full.get('t_goal'), 0)}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_theta_csv(model, path: Path):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(
            [
                "theta",
                "minimal_sets",
                "m1_t_goal",
                "m1_x1",
                "m1_k",
                "m1m4_t_goal",
                "m1m4_x1",
                "m1m4_k",
                "P1_t_goal",
                "P1_x1",
                "P1_k",
                "hybrid_t_goal",
                "hybrid_x1",
                "hybrid_k",
                "full_t_goal",
                "full_x1",
                "full_k",
            ]
        )
        for theta in THETAS:
            rows = compare_plans(model, "nominal", SimOptions(theta=theta))
            by_set = {tuple(row["measures"]): row for row in rows}
            mss = minimal_sufficient_sets(rows)
            m1 = by_set.get(("m1",), {})
            m14 = by_set.get(("m1", "m4"), {})
            p1 = by_set.get(("m3", "m4"), {})
            hy = by_set.get(("m1", "m3"), {})
            full = by_set.get(("m1", "m3", "m4"), {})
            writer.writerow(
                [
                    f"{theta:.3f}",
                    " + ".join("{" + ",".join(items) + "}" for items in mss),
                    m1.get("t_goal"),
                    m1.get("x_target_T"),
                    m1.get("k_T"),
                    m14.get("t_goal"),
                    m14.get("x_target_T"),
                    m14.get("k_T"),
                    p1.get("t_goal"),
                    p1.get("x_target_T"),
                    p1.get("k_T"),
                    hy.get("t_goal"),
                    hy.get("x_target_T"),
                    hy.get("k_T"),
                    full.get("t_goal"),
                    full.get("x_target_T"),
                    full.get("k_T"),
                ]
            )


def write_scenario_matrix(model, path: Path):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(
            [
                "plan",
                "scenario",
                "reached",
                "feasible",
                "t_goal",
                "x1_T",
                "margin",
                "cost",
                "k",
                "success_share",
                "radius",
            ]
        )
        for plan_name, plan in model.plans.items():
            robustness = scenario_robustness(model, plan)
            for row in robustness["rows"]:
                writer.writerow(
                    [
                        plan_name,
                        row["scenario"],
                        row["reached"],
                        row["feasible"],
                        row["t_goal"],
                        f"{row['x_target_T']:.4f}",
                        f"{row['margin']:.4f}",
                        f"{row['cost_total']:.2f}",
                        "" if row["k_T"] is None else f"{row['k_T']:.4f}",
                        f"{robustness['success_share']:.3f}",
                        robustness["radius"],
                    ]
                )


def build_recommendation(model):
    named = compare_named_plans(model)
    feasible = [row for row in named if row["reached"] and row["feasible"]]
    if not feasible:
        return "Ни один именованный план не достигает цели в рамках бюджета.\n"
    best = max(feasible, key=lambda row: (row["k_T"] or 0.0, -row["t_goal"]))
    robustness = scenario_robustness(model, best["plan"])
    pareto = pareto_front(named)
    events = critical_event_combinations(model, best["plan"])
    lines = [
        f"# Рекомендация (конфиг {model.params.version}, θ={model.params.theta:g})",
        "",
        f"**{best['plan_label']}**",
        "",
        f"- Цель: к {best['t_goal']:g} мес.; запас x1(T) − 1 = {best['margin']:+.3f}.",
        f"- Стоимость: {best['cost_total']:.1f} млн руб. "
        f"при бюджете {model.params.budget_max:.1f} млн.",
        f"- k = R/C: {best['k_T']:.3f}.",
        f"- Устойчивость: {robustness['radius']} из "
        f"{len(model.atomic_scenarios)} одновременных событий; "
        f"доля сценариев с целью — {robustness['success_share'] * 100:.0f} %.",
        f"- Критические события: "
        + (", ".join("{" + ", ".join(c) + "}" for c in events) or "нет")
        + ".",
        f"- Парето-фронт: {', '.join(row['plan'] for row in pareto)}.",
        "",
        "## Кому и когда",
        "",
        "| Мера | Ответственный | Старт, мес. |",
        "|---|---|---|",
    ]
    plan = model.plan(best["plan"])
    for measure, program in plan.programs.items():
        node = model.node(measure)
        lines.append(
            f"| {node.short or measure} | {node.owner or 'не задан'} | {program.start:g} |"
        )
    lines += [
        "",
        "Примечание: это отладочный расчёт на демонстрационных параметрах, "
        "а не эмпирический результат предприятия.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "karkas.yaml"),
        help="YAML-конфиг модели",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "out" / "thesis"),
        help="каталог для таблиц и CSV",
    )
    args = parser.parse_args()

    model = load_model(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "section5.tex").write_text(build_section5(model), encoding="utf-8")
    (out / "theta_sensitivity.tex").write_text(build_theta_table(model), encoding="utf-8")
    write_theta_csv(model, out / "theta_sensitivity.csv")
    write_scenario_matrix(model, out / "plan_scenario_matrix.csv")
    (out / "recommendation.md").write_text(build_recommendation(model), encoding="utf-8")

    print(f"Модель: {model.params.version}, θ={model.params.theta:g}, "
          f"режим {model.params.saturation}")
    print(f"Записано в {out}: section5.tex, theta_sensitivity.tex, "
          "theta_sensitivity.csv, plan_scenario_matrix.csv, recommendation.md")
    print()
    print(build_recommendation(model))


if __name__ == "__main__":
    main()
