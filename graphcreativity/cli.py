"""Командный интерфейс: run / compare / graph."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .engine import SimOptions, simulate
from .metrics import (
    all_measure_sets,
    compare_named_plans,
    compare_plans,
    critical_event_combinations,
    critical_removal_sets,
    effect_strength,
    minimal_sufficient_sets,
    pareto_front,
    plan_from_set,
    summarize,
)
from .model import Model, load_model
from .plots import plot_compare, plot_dynamics, plot_graph, plot_indicators

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "karkas.yaml"


def _parse_pairs(text: str | None) -> dict[str, float]:
    if not text:
        return {}
    result: dict[str, float] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(f"ожидалось id=значение, получено {chunk!r}")
        key, value = chunk.split("=", 1)
        result[key.strip()] = float(value)
    return result


def _parse_measures(text: str | None) -> list[str]:
    if not text:
        return []
    return [m.strip() for m in text.split(",") if m.strip()]


def _options(args) -> SimOptions:
    return SimOptions(
        horizon=args.horizon,
        theta=args.theta,
        mu=args.mu,
        x_max=args.x_max,
        gain=args.gain,
        budget_max=args.budget_max,
        saturation=args.saturation,
        cost_mode=args.cost_mode,
        revenue_mode=args.revenue_mode,
        include_optional=args.include_optional,
    )


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _print_table(headers: list[str], rows: list[list], widths: list[int] | None = None) -> None:
    if widths is None:
        widths = [
            max(len(str(h)), *(len(str(row[i])) for row in rows)) if rows else len(str(h))
            for i, h in enumerate(headers)
        ]
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))


def _write_dynamics_csv(result, path: Path) -> None:
    model = result.model
    fields = ["month"]
    for node in model.indicators:
        fields += [f"{node.id}_norm", f"{node.id}_phys"]
    fields += [f"u_{m}" for m in model.measure_ids]
    fields += [f"Y_{e}" for e in model.environment_ids]
    fields += [f"sigma_{i}" for i in model.indicator_ids]
    fields += ["R", "C_flow", "C_cum", "k"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for idx, month in enumerate(result.time):
            row = {"month": month}
            for node in model.indicators:
                row[f"{node.id}_norm"] = result.states[node.id][idx]
                row[f"{node.id}_phys"] = result.physical[node.id][idx]
            for m in model.measure_ids:
                row[f"u_{m}"] = result.measures[m][idx]
            for e in model.environment_ids:
                row[f"Y_{e}"] = result.environment[e][idx]
            for i in model.indicator_ids:
                row[f"sigma_{i}"] = result.sigma[i][idx]
            row["R"] = result.revenue[idx]
            row["C_flow"] = result.cost_flow[idx]
            row["C_cum"] = result.cost_cum[idx]
            row["k"] = (
                result.revenue[idx] / result.cost_cum[idx]
                if result.cost_cum[idx] > 0
                else None
            )
            writer.writerow(row)


def _print_summary(summary: dict) -> None:
    print()
    print(f"План:      {summary['plan_label']} ({summary['plan']})")
    print(f"Сценарий:  {summary['scenario_label']} ({summary['scenario']})")
    print(f"Бюджет:    {'OK' if summary['feasible'] else 'ПРЕВЫШЕН'}")
    print(
        f"Цель:      {'достигнута на ' + str(summary['t_goal']) + ' мес.' if summary['reached'] else 'не достигнута'}"
    )
    target = summary["target_node"]
    print(
        f"{target}(T):  {summary['x_target_T']:.3f} норм. = "
        f"{summary['target_phys_T']:.2f} {summary['target_unit']}"
        + (f" ({summary['growth_pct_T']:+.1f} % к базе)" if summary.get("growth_pct_T") is not None else "")
    )
    print(f"Затраты:   {summary['cost_total']:.2f} млн руб.")
    print(
        f"Запас:     {summary['margin']:+.3f} норм. "
        f"({summary['x_target_T']:.3f} − {summary['target_level']:.3f})"
    )
    print(
        f"Модель:    {summary.get('model_version', '—')}, "
        f"θ={summary.get('theta', float('nan')):g}, "
        f"режим {summary.get('saturation', '—')}"
    )
    if summary.get("revenue_mode") == "rubles":
        print(f"Эффект:    {summary['revenue_T']:.2f} млн руб.")
    else:
        print(f"Эффект:    {summary['revenue_T']:.2f} % прироста")
    admissible = summary["reached"] and summary["feasible"]
    print(f"k = R/C:   {_fmt(summary['k_T'] if admissible else None, 3)}")
    for warning in summary.get("warnings") or []:
        print(f"! {warning}")


def cmd_run(args) -> int:
    model = load_model(args.config)
    if args.plan and args.plan not in model.plans:
        print(f"Нет плана {args.plan!r}. Доступны: {', '.join(model.plans)}")
        return 2
    plan = model.plan(args.plan) if args.plan else None
    scenario = model.scenario(args.scenario)
    options = _options(args)
    measures = _parse_measures(args.measures)
    starts = _parse_pairs(args.starts)

    result = simulate(
        model,
        plan=plan,
        scenario=scenario,
        options=options,
        measures=measures if plan is None else None,
        starts=starts,
    )
    summary = summarize(result)
    _print_summary(summary)
    strength = effect_strength(model, result.plan.measures)
    print(f"σ1 (статика §5, пути ≤2): {strength:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_dynamics_csv(result, out / "dynamics.csv")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "params": asdict(result.params),
                "effect_strength": strength,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if not args.no_plot:
        plot_dynamics(result, out / "dynamics.png")
        plot_indicators(result, out / "indicators.png")
    print(f"\nРезультаты: {out.resolve()}")
    return 0


def _row_for_table(row: dict) -> list:
    return [
        row["plan"],
        "+".join(row["measures"]) or "—",
        _fmt(row["cost_total"], 1),
        _fmt(row["t_goal"], 0),
        _fmt(row["x_target_T"], 2),
        _fmt(row["growth_pct_T"], 1),
        _fmt(row["k_T"] if row["reached"] and row["feasible"] else None, 3),
        "да" if row["reached"] else "нет",
        "да" if row["feasible"] else "нет",
    ]


def cmd_compare(args) -> int:
    model = load_model(args.config)
    scenario = model.scenario(args.scenario)
    options = _options(args)
    rows = compare_plans(
        model,
        scenario=scenario,
        options=options,
        include_empty=not args.no_empty,
    )
    def _sort_key(row):
        admissible = row["reached"] and row["feasible"]
        return (
            not admissible,
            not row["reached"],
            -(row["k_T"] or 0.0) if admissible else 0.0,
            -row["x_target_T"],
            row["cost_total"],
        )

    rows.sort(key=_sort_key)
    headers = ["план", "меры", "C", "T_goal", "x1(T)", "рост, %", "k", "цель", "бюджет"]
    _print_table(headers, [_row_for_table(row) for row in rows])

    mss = minimal_sufficient_sets(rows)
    critical = critical_removal_sets(rows, model.measure_ids)
    front = pareto_front(rows)
    events = {
        name: critical_event_combinations(model, name, options=options)
        for name in model.plans
    }
    print("\nМинимальные достаточные наборы (цель + бюджет):")
    print("  " + (", ".join("{ }" if not s else "{" + ", ".join(s) + "}" for s in mss) or "—"))
    print("Критические меры (удаление ломает полный комплект):")
    print("  " + (", ".join("{" + ", ".join(s) + "}" for s in critical) or "—"))
    print("Критические комбинации событий:")
    named_nominal = {
        row["plan"]: row for row in compare_named_plans(model, "nominal", options)
    }
    for name, combos in events.items():
        baseline_ok = bool(
            named_nominal.get(name, {}).get("reached")
            and named_nominal.get(name, {}).get("feasible")
        )
        if not baseline_ok:
            text = "план недопустим уже без событий (цель/бюджет)"
        else:
            text = ", ".join("{" + ", ".join(c) + "}" for c in combos) or "нет"
        print(f"  {name}: {text}")
    print("Парето-фронт по (T_goal, C, k):")
    print("  " + (", ".join(r["plan"] for r in front) or "—"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with (out / "plans.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in row.items()})
    (out / "summary.json").write_text(
        json.dumps(
            {
                "scenario": scenario.name,
                "minimal_sufficient_sets": [list(s) for s in mss],
                "critical_removal_sets": [list(s) for s in critical],
                "critical_event_combinations": {
                    name: [list(c) for c in combos]
                    for name, combos in events.items()
                },
                "pareto": [r["plan"] for r in front],
                "plans": rows,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if not args.no_plot:
        plot_compare(rows, out / "plans.png")
    print(f"\nРезультаты: {out.resolve()}")
    return 0


def cmd_graph(args) -> int:
    model = load_model(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_graph(model, path=out, include_optional=args.include_optional)
    print(f"Граф: {out.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gc",
        description="Симулятор графовой модели повышения креативности предприятия",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="путь к YAML-конфигу")
    common.add_argument("--scenario", default="nominal", help="имя сценария")
    common.add_argument("--horizon", type=int, help="период моделирования, мес.")
    common.add_argument("--theta", type=float, help="порог срабатывания")
    common.add_argument("--mu", type=float, help="деградация, 1/мес")
    common.add_argument("--x-max", type=float, dest="x_max", help="потолок насыщения")
    common.add_argument("--gain", type=float, help="множитель всех весов")
    common.add_argument("--budget-max", type=float, dest="budget_max", help="бюджет, млн руб.")
    common.add_argument(
        "--saturation",
        choices=["ceiling", "strict"],
        help="ceiling: (x_max-x); strict: (1-x)",
    )
    common.add_argument(
        "--cost-mode",
        choices=["one_time", "monthly", "spread"],
        dest="cost_mode",
    )
    common.add_argument(
        "--revenue-mode",
        choices=["percent", "rubles"],
        dest="revenue_mode",
    )
    common.add_argument("--include-optional", action="store_true", help="включить X1->X4")
    common.add_argument("--no-plot", action="store_true", help="не сохранять PNG")

    run = sub.add_parser("run", parents=[common], help="один план и сценарий")
    run.add_argument("--plan", help="имя плана из конфига")
    run.add_argument("--measures", help="свой набор мер: m1,m3")
    run.add_argument("--starts", help="месяцы запуска: m1=0,m3=3")
    run.add_argument("--out", default="out/run", help="каталог результатов")
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", parents=[common], help="все подмножества мер")
    compare.add_argument("--out", default="out/compare", help="каталог результатов")
    compare.add_argument("--no-empty", action="store_true", help="не показывать пустой набор")
    compare.set_defaults(func=cmd_compare)

    graph = sub.add_parser("graph", help="нарисовать топологию")
    graph.add_argument("--config", default=str(DEFAULT_CONFIG))
    graph.add_argument("--out", default="out/graph.png")
    graph.add_argument("--include-optional", action="store_true")
    graph.set_defaults(func=cmd_graph)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
