"""Показатели, сравнение планов и критические наборы.

Термины разведены явно:

* ``minimal_sufficient_sets`` — включительно-минимальные наборы МЕР,
  достаточные для цели и допустимые по бюджету;
* ``critical_removal_sets`` — минимальные наборы мер, удаление которых
  ломает полный комплект (двойственная задача к предыдущей);
* ``critical_event_combinations`` — минимальные комбинации неблагоприятных
  СОБЫТИЙ, при которых план перестаёт достигать цели. Это тот объект,
  который в теории называется критическими комбинациями событий.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

from .engine import SimOptions, SimulationResult, simulate
from .model import MeasureProgram, Model, Plan, Scenario


# ---------------------------------------------------------------------------
# Сводка одного прогона
# ---------------------------------------------------------------------------

def summarize(result: SimulationResult) -> dict:
    target = result.model.node(result.params.target_node)
    last = result.last_index
    cost = result.cost_cum[last] if result.cost_cum else 0.0
    revenue = result.revenue[last] if result.revenue else 0.0
    x_target = result.states[result.params.target_node][last]
    t_goal = result.first_goal_time()

    summary = {
        "plan": result.plan.name,
        "plan_label": result.plan.label or result.plan.name,
        "measures": list(result.plan.measures),
        "scenario": result.scenario.name,
        "scenario_label": result.scenario.label or result.scenario.name,
        "feasible": result.feasible,
        "reached": t_goal is not None,
        "t_goal": t_goal,
        "horizon": result.params.horizon,
        "target_node": result.params.target_node,
        "target_level": result.params.target_level,
        "x_target_T": x_target,
        "margin": x_target - result.params.target_level,
        "target_phys_T": result.physical[result.params.target_node][last],
        "target_unit": target.unit,
        "growth_pct_T": (
            result.revenue[last] if result.params.revenue_mode == "percent" else None
        ),
        "revenue_T": revenue,
        "cost_total": cost,
        "k_T": (revenue / cost) if cost > 0 else None,
        "revenue_mode": result.params.revenue_mode,
        "theta": result.params.theta,
        "saturation": result.params.saturation,
        "budget_max": result.params.budget_max,
        "model_version": result.model.params.version,
        "warnings": list(result.warnings),
    }
    for node in result.model.indicators:
        summary[f"{node.id}_norm"] = result.states[node.id][last]
        summary[f"{node.id}_phys"] = result.physical[node.id][last]
    return summary


# ---------------------------------------------------------------------------
# Статическая «сила эффекта» из §5 karkas.tex
# ---------------------------------------------------------------------------

def effect_strength(
    model: Model,
    active_measures: Iterable[str],
    target: str | None = None,
    max_path_len: int = 2,
    include_optional: bool = False,
) -> float:
    """Сумма произведений весов по путям длины <= max_path_len.

    Именно эта величина (а не динамика) приведена в первом столбце
    таблицы §5 karkas.tex: для {m1} получается
    0.60*0.70 + 0.35*0.55 = 0.6125 и т.д.
    """
    target = target or model.params.target_node
    pred = model.predecessors(include_optional)
    active = set(active_measures)

    def walk(node: str, depth: int, weight: float) -> float:
        if depth == 0:
            return 0.0
        total = 0.0
        for edge in pred.get(node, []):
            contribution = weight * edge.w
            if edge.src in active:
                total += contribution
            if edge.src in model.indicator_ids:
                total += walk(edge.src, depth - 1, contribution)
        return total

    return walk(target, max_path_len, 1.0)


# ---------------------------------------------------------------------------
# Перебор подмножеств мер
# ---------------------------------------------------------------------------

def all_measure_sets(model: Model) -> list[tuple[str, ...]]:
    ids = model.measure_ids
    sets: list[tuple[str, ...]] = []
    for size in range(len(ids) + 1):
        sets.extend(itertools.combinations(ids, size))
    return sets


def plan_from_set(
    model: Model,
    measures: Sequence[str],
    starts: dict[str, float] | None = None,
) -> Plan:
    starts = starts or {}
    programs = {
        measure: MeasureProgram(start=float(starts.get(measure, 0.0)))
        for measure in measures
    }
    name = "+".join(measures) if measures else "—"
    label = " + ".join(model.node(m).short or m for m in measures) if measures else "Без мер"
    return Plan(name=name, label=label, programs=programs)


def compare_plans(
    model: Model,
    scenario: Scenario | str = "nominal",
    options: SimOptions | None = None,
    *,
    include_empty: bool = True,
    measure_sets: Iterable[Sequence[str]] | None = None,
) -> list[dict]:
    """Прогнать все подмножества мер (все старты — с 0-го месяца)."""
    if isinstance(scenario, str):
        scenario = model.scenario(scenario)
    sets = list(measure_sets) if measure_sets is not None else all_measure_sets(model)
    if not include_empty:
        sets = [s for s in sets if s]
    rows: list[dict] = []
    for measures in sets:
        plan = plan_from_set(model, measures)
        result = simulate(model, plan, scenario, options)
        row = summarize(result)
        rows.append(row)
    return rows


def compare_named_plans(
    model: Model,
    scenario: Scenario | str = "nominal",
    options: SimOptions | None = None,
) -> list[dict]:
    """Прогнать именованные планы с их собственными стартами.

    Это важно для согласованности KPI и таблицы сравнения: гибрид
    стартует m3 с 3-го месяца, поэтому его нельзя сравнивать с
    подмножеством ``{m1, m3}``, у которого все старты нулевые.
    """
    if isinstance(scenario, str):
        scenario = model.scenario(scenario)
    return [
        summarize(simulate(model, plan, scenario, options))
        for plan in model.plans.values()
    ]


# ---------------------------------------------------------------------------
# Минимальные достаточные наборы
# ---------------------------------------------------------------------------

def _measure_key(row: dict) -> frozenset[str]:
    return frozenset(row.get("measures") or [])


def _minimal_by(
    rows: Iterable[dict],
    predicate,
) -> list[tuple[str, ...]]:
    by_set: dict[frozenset[str], bool] = {}
    for row in rows:
        key = _measure_key(row)
        by_set[key] = by_set.get(key, False) or bool(predicate(row))

    result: list[tuple[str, ...]] = []
    for measures, ok in by_set.items():
        if not ok:
            continue
        if any(
            subset and by_set.get(frozenset(subset), False)
            for subset in _proper_subsets(measures)
        ):
            continue
        result.append(tuple(sorted(measures)))
    return sorted(result, key=lambda s: (len(s), s))


def minimal_sufficient_sets(
    rows: Iterable[dict],
    *,
    require_feasible: bool = True,
) -> list[tuple[str, ...]]:
    """Включительно-минимальные достаточные наборы мер.

    По умолчанию «достаточный» означает одновременно ``reached`` и
    ``feasible``: набор, превышающий бюджет, не является допустимым
    решением. При ``require_feasible=False`` бюджет игнорируется — это
    нужно только для диагностики.
    """
    return _minimal_by(
        rows,
        lambda row: row.get("reached")
        and (row.get("feasible", True) or not require_feasible),
    )


def critical_removal_sets(
    rows: Iterable[dict],
    all_measures: Sequence[str],
) -> list[tuple[str, ...]]:
    """Минимальные наборы мер, удаление которых ломает полный комплект.

    Полный комплект должен быть достаточным и допустимым. Комбинация R
    критична, если full \\ R не достигает цели, а full \\ R' достигает
    для любого собственного подмножества R' ⊂ R.
    """
    by_set = {
        _measure_key(row): bool(row.get("reached")) and bool(row.get("feasible", True))
        for row in rows
    }
    full = frozenset(all_measures)
    if not by_set.get(full, False):
        return []
    critical: list[tuple[str, ...]] = []
    for size in range(1, len(full) + 1):
        for removal in itertools.combinations(sorted(full), size):
            removal_set = frozenset(removal)
            remaining = full - removal_set
            if by_set.get(remaining, False):
                continue
            if any(
                not by_set.get(full - frozenset(sub), False)
                for sub in _proper_subsets(removal_set)
            ):
                continue
            critical.append(tuple(sorted(removal_set)))
    return sorted(critical, key=lambda s: (len(s), s))


# Обратная совместимость со старым именем. Это именно removal sets,
# а не комбинации событий: для событий используйте
# ``critical_event_combinations``.
critical_combinations = critical_removal_sets


def _proper_subsets(items: Iterable[str]):
    items = tuple(sorted(items))
    for size in range(len(items)):
        yield from itertools.combinations(items, size)


# ---------------------------------------------------------------------------
# Критические комбинации событий
# ---------------------------------------------------------------------------

def combine_scenarios(events: Sequence[Scenario]) -> Scenario:
    """Суммировать отклонения нескольких событий в один сценарий."""
    values: dict[str, float] = {}
    for event in events:
        for node_id, value in event.values.items():
            values[node_id] = values.get(node_id, 0.0) + float(value)
    name = "+".join(event.name for event in events) or "nominal"
    label = " + ".join(event.label or event.name for event in events) or "Номинал"
    return Scenario(name=name, label=label, values=values)


def critical_event_combinations(
    model: Model,
    plan: Plan | str,
    options: SimOptions | None = None,
    events: Iterable[Scenario | str] | None = None,
) -> list[tuple[str, ...]]:
    """Минимальные комбинации событий, которые ломают план.

    Событие — атомарный сценарий (например, ``market_downturn``).
    Комбинация критична, если план успешен при всех её собственных
    подмножествах и неуспешен (или недопустим по бюджету) при ней самой.
    """
    if isinstance(plan, str):
        plan = model.plan(plan)
    if events is None:
        events = model.atomic_scenarios
    resolved: list[Scenario] = []
    for event in events:
        resolved.append(model.scenario(event) if isinstance(event, str) else event)
    resolved = [event for event in resolved if event.name != "nominal"]

    def succeeds(combo: Sequence[Scenario]) -> bool:
        scenario = combine_scenarios(combo) if combo else model.scenario("nominal")
        summary = summarize(simulate(model, plan, scenario, options))
        return bool(summary["reached"] and summary["feasible"])

    status: dict[frozenset[str], bool] = {}
    for size in range(len(resolved) + 1):
        for combo in itertools.combinations(resolved, size):
            status[frozenset(event.name for event in combo)] = succeeds(combo)

    if not status.get(frozenset(), False):
        return []

    result: list[tuple[str, ...]] = []
    for size in range(1, len(resolved) + 1):
        for combo in itertools.combinations(resolved, size):
            key = frozenset(event.name for event in combo)
            if status[key]:
                continue
            if all(status.get(frozenset(sub), False) for sub in _proper_subsets(key)):
                result.append(tuple(sorted(key)))
    return sorted(result, key=lambda s: (len(s), s))


def robustness_radius(
    model: Model,
    plan: Plan | str,
    options: SimOptions | None = None,
    events: Iterable[Scenario | str] | None = None,
) -> int:
    """Сколько одновременных неблагоприятных событий выдерживает план.

    Возвращает максимальное k, при котором план успешен при ЛЮБОЙ
    комбинации из k событий. ``-1`` означает, что план не проходит уже
    в номинальном сценарии.
    """
    if isinstance(plan, str):
        plan = model.plan(plan)
    if events is None:
        events = model.atomic_scenarios
    resolved: list[Scenario] = []
    for event in events:
        resolved.append(model.scenario(event) if isinstance(event, str) else event)
    resolved = [event for event in resolved if event.name != "nominal"]

    def succeeds(combo: Sequence[Scenario]) -> bool:
        scenario = combine_scenarios(combo) if combo else model.scenario("nominal")
        summary = summarize(simulate(model, plan, scenario, options))
        return bool(summary["reached"] and summary["feasible"])

    if not succeeds(()):
        return -1
    radius = 0
    for size in range(1, len(resolved) + 1):
        if all(succeeds(combo) for combo in itertools.combinations(resolved, size)):
            radius = size
        else:
            break
    return radius


# ---------------------------------------------------------------------------
# Парето-фронт
# ---------------------------------------------------------------------------

def pareto_front(rows: Iterable[dict]) -> list[dict]:
    """Планы, не доминируемые по (T_goal ↓, C ↓, k ↑).

    Учитываются только достаточные и допустимые по бюджету планы.
    """
    candidates = [
        row
        for row in rows
        if row.get("reached") and row.get("feasible") and row.get("k_T") is not None
    ]
    front: list[dict] = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            no_worse = (
                other["t_goal"] <= row["t_goal"]
                and other["cost_total"] <= row["cost_total"] + 1e-9
                and other["k_T"] >= row["k_T"] - 1e-12
            )
            strictly_better = (
                other["t_goal"] < row["t_goal"]
                or other["cost_total"] < row["cost_total"] - 1e-9
                or other["k_T"] > row["k_T"] + 1e-12
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda r: (r["t_goal"], r["cost_total"]))


# ---------------------------------------------------------------------------
# Устойчивость к сценариям
# ---------------------------------------------------------------------------

def scenario_robustness(
    model: Model,
    plan: Plan | str,
    scenarios: Iterable[Scenario | str] | None = None,
    options: SimOptions | None = None,
) -> dict:
    if isinstance(plan, str):
        plan = model.plan(plan)
    if scenarios is None:
        scenarios = list(model.scenarios.values())
    rows = []
    for scenario in scenarios:
        if isinstance(scenario, str):
            scenario = model.scenario(scenario)
        result = simulate(model, plan, scenario, options)
        summary = summarize(result)
        rows.append(
            {
                "scenario": scenario.name,
                "label": scenario.label or scenario.name,
                "reached": summary["reached"],
                "t_goal": summary["t_goal"],
                "x_target_T": summary["x_target_T"],
                "margin": summary["margin"],
                "cost_total": summary["cost_total"],
                "k_T": summary["k_T"],
                "feasible": summary["feasible"],
            }
        )
    success = sum(1 for row in rows if row["reached"] and row["feasible"])
    return {
        "plan": plan.name,
        "plan_label": plan.label or plan.name,
        # Доля сценариев, а не вероятность: вероятности сценариев не заданы.
        "success_share": success / len(rows) if rows else 0.0,
        # Старое имя оставлено для совместимости.
        "success_rate": success / len(rows) if rows else 0.0,
        "min_x_target_T": min((row["x_target_T"] for row in rows), default=0.0),
        "min_margin": min((row["margin"] for row in rows), default=0.0),
        "radius": robustness_radius(model, plan, options),
        "rows": rows,
    }


def measure_cost(model: Model, measures: Iterable[str]) -> float:
    return sum(model.node(m).cost for m in measures)
