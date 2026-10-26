"""Дискретный движок графовой модели.

Уравнение из karkas.tex (1), с явным потолком насыщения:

    sigma_i(t) = sum_p w_pi * a_p(t - tau_pi)
    x_i(t+1)   = x_i(t) + lambda_i * (x_max - x_i(t)) * [sigma_i - theta]_+
                 - mu_i * x_i(t)
    lambda_i   = ln(10) / rise_i

Все показатели X нормированы: 0 — база, 1 — цель. Для среды Y и мер M
в качестве a_p используются значения сценария и расписания соответственно.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable

from .model import (
    ENVIRONMENT,
    INDICATOR,
    MEASURE,
    MeasureProgram,
    Model,
    Plan,
    Scenario,
    estimate_plan_cost,
)


@dataclass
class SimOptions:
    """Переопределения параметров модели на один прогон."""

    horizon: int | None = None
    dt: float | None = None
    saturation: str | None = None
    theta: float | None = None
    mu: float | None = None
    x_max: float | None = None
    gain: float | None = None
    budget_max: float | None = None
    cost_mode: str | None = None
    revenue_mode: str | None = None
    include_optional: bool = False

    def resolve(self, model: Model) -> "ResolvedParams":
        p = model.params
        return ResolvedParams(
            horizon=int(self.horizon if self.horizon is not None else p.horizon),
            dt=float(self.dt if self.dt is not None else p.dt),
            saturation=str(self.saturation if self.saturation is not None else p.saturation),
            theta=float(self.theta if self.theta is not None else p.theta),
            mu=float(self.mu if self.mu is not None else p.mu),
            x_max=float(self.x_max if self.x_max is not None else p.x_max),
            gain=float(self.gain if self.gain is not None else p.gain),
            budget_max=float(
                self.budget_max if self.budget_max is not None else p.budget_max
            ),
            cost_mode=str(self.cost_mode if self.cost_mode is not None else p.cost_mode),
            revenue_mode=str(self.revenue_mode if self.revenue_mode is not None else p.revenue_mode),
            target_node=p.target_node,
            target_level=float(p.target_level),
            include_optional=bool(self.include_optional),
        )


@dataclass(frozen=True)
class ResolvedParams:
    horizon: int
    dt: float
    saturation: str
    theta: float
    mu: float
    x_max: float
    gain: float
    budget_max: float
    cost_mode: str
    revenue_mode: str
    target_node: str
    target_level: float
    include_optional: bool


@dataclass
class SimulationResult:
    model: Model
    plan: Plan
    scenario: Scenario
    params: ResolvedParams
    time: list[float] = field(default_factory=list)
    states: dict[str, list[float]] = field(default_factory=dict)
    physical: dict[str, list[float]] = field(default_factory=dict)
    measures: dict[str, list[float]] = field(default_factory=dict)
    environment: dict[str, list[float]] = field(default_factory=dict)
    sigma: dict[str, list[float]] = field(default_factory=dict)
    revenue: list[float] = field(default_factory=list)
    cost_flow: list[float] = field(default_factory=list)
    cost_cum: list[float] = field(default_factory=list)
    feasible: bool = True
    warnings: list[str] = field(default_factory=list)

    # --- удобные срезы --------------------------------------------------
    @property
    def last_index(self) -> int:
        return len(self.time) - 1

    @property
    def target_id(self) -> str:
        return self.params.target_node

    def state_at(self, node_id: str, t: int) -> float:
        return self.states[node_id][t]

    def physical_at(self, node_id: str, t: int) -> float:
        return self.physical[node_id][t]

    def first_goal_time(self) -> float | None:
        for t, value in zip(self.time, self.states[self.target_id]):
            if value >= self.params.target_level:
                return t
        return None

    def reached(self) -> bool:
        return self.first_goal_time() is not None


def _physical(node, x: float) -> float:
    """Нормированное состояние -> физическая единица."""
    if node.direction == "up":
        return node.base + x * (node.target - node.base)
    return node.base - x * (node.base - node.target)


def _percent_effect(node, x: float) -> float:
    """Прирост в процентах к базе (в «хорошую» сторону)."""
    if node.base == 0:
        return 0.0
    return 100.0 * x * abs(node.target - node.base) / node.base


def _rub_effect(node, x: float) -> float:
    """Текущий денежный эффект по показателю, млн руб."""
    r = float(node.monetization.get("r", 0.0))
    volume = float(node.monetization.get("volume", 0.0))
    if not r or not volume:
        return 0.0
    effect = x * abs(node.target - node.base)  # в физических единицах, «лучше»
    return r * effect * volume / 1_000_000.0


def _make_plan_from_measures(
    model: Model,
    measure_ids: Iterable[str],
    starts: dict[str, float] | None = None,
) -> Plan:
    starts = starts or {}
    programs = {
        measure: MeasureProgram(start=float(starts.get(measure, 0.0)))
        for measure in measure_ids
    }
    return Plan(name="custom", label="Пользовательский план", programs=programs)


def simulate(
    model: Model,
    plan: Plan | str | None = None,
    scenario: Scenario | str | None = None,
    options: SimOptions | None = None,
    *,
    measures: Iterable[str] | None = None,
    starts: dict[str, float] | None = None,
) -> SimulationResult:
    """Прогон одного плана в одном сценарии.

    plan может быть именем плана из конфига, объектом Plan или None
    (тогда используется measures/starts).
    """
    opts = (options or SimOptions()).resolve(model)
    if opts.dt <= 0:
        raise ValueError("dt должен быть > 0")

    # Явно заданные глобальные параметры перекрывают per-node значения.
    # Если параметр не задан, используется theta/mu/x_max узла, а затем
    # значение модели.
    theta_override = options.theta if options is not None else None
    mu_override = options.mu if options is not None else None
    x_max_override = options.x_max if options is not None else None

    if isinstance(plan, str):
        plan = model.plan(plan)
    elif plan is None:
        plan = _make_plan_from_measures(model, measures or [], starts)
    if isinstance(scenario, str):
        scenario = model.scenario(scenario)
    elif scenario is None:
        scenario = model.scenario("nominal") if "nominal" in model.scenarios else Scenario("nominal")

    horizon = opts.horizon
    dt = opts.dt
    steps = int(round(horizon / dt)) + 1
    time = [i * dt for i in range(steps)]

    indicators = model.indicator_ids
    measure_ids = model.measure_ids
    environment_ids = model.environment_ids
    pred = model.predecessors(opts.include_optional)

    # --- расписания мер -------------------------------------------------
    measure_series: dict[str, list[float]] = {}
    for measure_id in measure_ids:
        program = plan.programs.get(measure_id)
        if program is None:
            measure_series[measure_id] = [0.0] * steps
        else:
            measure_series[measure_id] = [
                program.value(t, int(math.ceil(horizon))) for t in time
            ]

    # --- среда ----------------------------------------------------------
    environment_series: dict[str, list[float]] = {
        env_id: [scenario.value(env_id)] * steps for env_id in environment_ids
    }

    # --- проверка бюджета ----------------------------------------------
    plan_cost = estimate_plan_cost(model, plan, opts.cost_mode, horizon)
    feasible = plan_cost <= opts.budget_max + 1e-9

    result = SimulationResult(
        model=model,
        plan=plan,
        scenario=scenario,
        params=opts,
        feasible=feasible,
    )
    if not feasible:
        result.warnings.append(
            f"План стоит {plan_cost:g} млн руб. при бюджете "
            f"{opts.budget_max:g} млн руб."
        )
    if opts.revenue_mode == "rubles":
        result.warnings.append(
            "Рублёвый режим монетизирует только показатели с заданными r_i; "
            "в текущей калибровке это X7, поэтому k несопоставим между планами."
        )

    # --- начальные состояния -------------------------------------------
    states = {node_id: [0.0] * steps for node_id in indicators}
    sigma = {node_id: [0.0] * steps for node_id in indicators}
    revenue = [0.0] * steps
    cost_flow = [0.0] * steps
    cost_cum = [0.0] * steps
    charged: set[str] = set()

    def source_value(edge_src: str, index: int) -> float:
        if index < 0:
            return 0.0
        if index >= steps:
            index = steps - 1
        if edge_src in states:
            return states[edge_src][index]
        if edge_src in measure_series:
            return measure_series[edge_src][index]
        if edge_src in environment_series:
            return environment_series[edge_src][index]
        return 0.0

    # --- основной цикл --------------------------------------------------
    for step in range(steps - 1):
        # стоимость мер на шаге step
        for measure_id, program in plan.programs.items():
            node = model.node(measure_id)
            is_active = program.value(time[step], int(math.ceil(horizon))) > 0
            if not is_active:
                continue
            if opts.cost_mode == "one_time":
                if measure_id not in charged:
                    cost_flow[step] += node.cost * program.intensity
                    charged.add(measure_id)
            elif opts.cost_mode == "monthly":
                cost_flow[step] += node.cost * program.intensity * dt
            elif opts.cost_mode == "spread":
                duration = program.duration or horizon
                cost_flow[step] += node.cost * program.intensity * dt / max(duration, 1e-9)

        # притоки и обновление состояний
        next_values: dict[str, float] = {}
        for indicator in indicators:
            node = model.node(indicator)
            total = 0.0
            for edge in pred.get(indicator, []):
                delay_steps = int(round(edge.delay / dt))
                total += (
                    edge.w
                    * opts.gain
                    * source_value(edge.src, step - delay_steps)
                )
            sigma[indicator][step] = total

            theta = (
                theta_override
                if theta_override is not None
                else (node.theta if node.theta is not None else opts.theta)
            )
            mu = (
                mu_override
                if mu_override is not None
                else (node.mu if node.mu is not None else opts.mu)
            )
            x_max = (
                x_max_override
                if x_max_override is not None
                else (node.x_max if node.x_max is not None else opts.x_max)
            )
            rise = node.rise or 1.0
            lam = math.log(10.0) * dt / rise
            drive = max(0.0, total - theta)

            x = states[indicator][step]
            cap = x_max if opts.saturation == "ceiling" else 1.0
            x_next = x + lam * (cap - x) * drive - mu * dt * x
            x_next = max(0.0, min(cap, x_next))
            next_values[indicator] = x_next

        for indicator, value in next_values.items():
            states[indicator][step + 1] = value

        # текущий эффект и накопленная стоимость
        target = model.node(opts.target_node)
        if opts.revenue_mode == "percent":
            revenue[step] = _percent_effect(target, states[opts.target_node][step])
        else:
            revenue[step] = sum(
                _rub_effect(model.node(indicator), states[indicator][step])
                for indicator in indicators
            )
        cost_cum[step] = (cost_cum[step - 1] if step > 0 else 0.0) + cost_flow[step]

    # последний момент: доход по конечному состоянию, стоимость без новых списаний
    last = steps - 1
    target = model.node(opts.target_node)
    if opts.revenue_mode == "percent":
        revenue[last] = _percent_effect(target, states[opts.target_node][last])
    else:
        revenue[last] = sum(
            _rub_effect(model.node(indicator), states[indicator][last])
            for indicator in indicators
        )
    cost_cum[last] = cost_cum[last - 1] if last > 0 else 0.0

    result.time = time
    result.states = states
    result.physical = {
        indicator: [_physical(model.node(indicator), x) for x in states[indicator]]
        for indicator in indicators
    }
    result.measures = measure_series
    result.environment = environment_series
    result.sigma = sigma
    result.revenue = revenue
    result.cost_flow = cost_flow
    result.cost_cum = cost_cum
    return result
