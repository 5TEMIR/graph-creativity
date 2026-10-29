"""Загрузка и валидация графовой модели.

Конфиг — единственный источник истины: вершины, дуги, параметры, планы
и сценарии. Здесь нет расчётов, только структура данных.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

INDICATOR = "indicator"
MEASURE = "measure"
ENVIRONMENT = "environment"
KINDS = {INDICATOR, MEASURE, ENVIRONMENT}


@dataclass
class ModelParams:
    horizon: int = 36
    dt: float = 1.0
    saturation: str = "ceiling"          # ceiling | strict
    theta: float = 0.45
    mu: float = 0.01
    x_max: float = 1.6
    gain: float = 1.0
    budget_max: float = 60.0
    cost_mode: str = "one_time"          # one_time | monthly | spread
    revenue_mode: str = "percent"        # percent | rubles
    target_node: str = "X1"
    target_level: float = 1.0
    version: str = "dev"                 # версия калибровки, печатается в отчётах


@dataclass
class Node:
    id: str
    kind: str
    label: str = ""
    short: str = ""
    unit: str = ""
    direction: str = "up"                # up | down
    base: float = 0.0
    target: float = 1.0
    rise: float | None = None            # мес. до плато (для X)
    theta: float | None = None
    mu: float | None = None
    x_max: float | None = None
    cost: float = 0.0                    # млн руб. (для мер)
    owner: str = ""                      # подразделение-ответственный (для мер)
    nominal: float = 0.0                 # номинал сценария (для Y)
    monetization: dict[str, float] = field(default_factory=dict)
    position: tuple[float, float] | None = None

    @property
    def is_indicator(self) -> bool:
        return self.kind == INDICATOR

    @property
    def is_measure(self) -> bool:
        return self.kind == MEASURE

    @property
    def is_environment(self) -> bool:
        return self.kind == ENVIRONMENT


@dataclass
class Edge:
    src: str
    dst: str
    w: float
    delay: float = 0.0
    source_class: str = ""
    mechanism: str = ""
    optional: bool = False


@dataclass
class MeasureProgram:
    """Расписание одной меры.

    Кампания меры — это полуинтервал ``[start, end)``. Внутри него мера
    либо работает непрерывно, либо применяется периодически:

    * ``interval`` не задан или равен 0 — мера активна непрерывно до ``end``;
    * ``interval > 0`` — применения начинаются в ``start``, ``start + interval``,
      ...; каждое длится ``duration`` месяцев и обрезается по ``end``.

    ``end=None`` означает конец периода моделирования. Если ``interval > 0``,
    ``duration`` должна быть положительной. ``duration`` не влияет на
    непрерывный режим.
    """

    start: float = 0.0
    end: float | None = None
    duration: float | None = None
    interval: float | None = None
    intensity: float = 1.0

    def campaign_end(self, horizon: int) -> float:
        """Конец кампании; None означает конец горизонта."""
        return float(horizon + 1) if self.end is None else float(self.end)

    def is_periodic(self) -> bool:
        return self.interval is not None and self.interval > 0

    def value(self, t: float, horizon: int) -> float:
        end = self.campaign_end(horizon)
        if t < self.start or t >= end:
            return 0.0
        if not self.is_periodic():
            return self.intensity

        interval = float(self.interval)
        duration = (
            float(self.duration)
            if self.duration is not None
            else interval
        )
        if duration <= 0:
            return 0.0

        # Проверяем последнее применение, начавшееся не позже t. Если
        # duration <= interval, более ранние применения уже закончились;
        # если duration > interval, последнее применение всё ещё активно.
        cycle = math.floor((t - self.start) / interval + 1e-12)
        cycle_start = self.start + cycle * interval
        return self.intensity if t < cycle_start + duration else 0.0

    def active_steps(self, horizon: int, dt: float = 1.0) -> int:
        """Число шагов сетки [0, horizon), на которых мера активна."""
        if dt <= 0:
            raise ValueError("dt должен быть > 0")
        steps = round(horizon / dt)
        return sum(
            1
            for index in range(steps)
            if self.value(index * dt, horizon) > 0
        )

    def spread_duration(self, horizon: int) -> float:
        """База для равномерного списания стоимости (месяцев)."""
        if self.is_periodic():
            duration = self.duration if self.duration is not None else self.interval
            return max(float(duration), 1e-9)
        end = min(self.campaign_end(horizon), float(horizon))
        return max(end - self.start, 1e-9)


@dataclass
class Plan:
    name: str
    label: str = ""
    programs: dict[str, MeasureProgram] = field(default_factory=dict)

    @property
    def measures(self) -> list[str]:
        return sorted(self.programs)

    def program(self, measure: str) -> MeasureProgram:
        return self.programs.get(measure, MeasureProgram(start=float("inf")))


@dataclass
class Scenario:
    name: str
    label: str = ""
    values: dict[str, float] = field(default_factory=dict)
    atomic: bool = True                  # атомарное событие для перебора комбинаций

    def value(self, node: str) -> float:
        return float(self.values.get(node, 0.0))


@dataclass
class Model:
    params: ModelParams
    nodes: dict[str, Node]
    edges: list[Edge]
    plans: dict[str, Plan]
    scenarios: dict[str, Scenario]
    raw: dict[str, Any] = field(default_factory=dict)

    # --- выборки вершин -------------------------------------------------
    @property
    def indicators(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.is_indicator]

    @property
    def measures(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.is_measure]

    @property
    def environments(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.is_environment]

    @property
    def indicator_ids(self) -> list[str]:
        return [n.id for n in self.indicators]

    @property
    def measure_ids(self) -> list[str]:
        return [n.id for n in self.measures]

    @property
    def environment_ids(self) -> list[str]:
        return [n.id for n in self.environments]

    @property
    def atomic_scenarios(self) -> list[Scenario]:
        """Сценарии-события для перебора критических комбинаций."""
        return [s for s in self.scenarios.values() if s.atomic and s.name != "nominal"]

    # --- дуги -----------------------------------------------------------
    def simulation_edges(self, include_optional: bool = False) -> list[Edge]:
        return [e for e in self.edges if include_optional or not e.optional]

    def predecessors(
        self, include_optional: bool = False
    ) -> dict[str, list[Edge]]:
        pred: dict[str, list[Edge]] = {node_id: [] for node_id in self.indicator_ids}
        for edge in self.simulation_edges(include_optional):
            pred.setdefault(edge.dst, []).append(edge)
        return pred

    # --- удобные ссылки -------------------------------------------------
    @property
    def target_node(self) -> Node:
        return self.nodes[self.params.target_node]

    def node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def plan(self, name: str) -> Plan:
        if name not in self.plans:
            options = ", ".join(sorted(self.plans))
            raise KeyError(f"План {name!r} не найден. Доступны: {options}")
        return self.plans[name]

    def scenario(self, name: str) -> Scenario:
        if name not in self.scenarios:
            options = ", ".join(sorted(self.scenarios))
            raise KeyError(f"Сценарий {name!r} не найден. Доступны: {options}")
        return self.scenarios[name]


def _parse_position(value: Any) -> tuple[float, float] | None:
    if not value:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"position должен быть парой [x, y], получено: {value!r}")


def _parse_program(value: Any) -> MeasureProgram:
    if isinstance(value, (int, float)):
        return MeasureProgram(start=float(value))
    if isinstance(value, dict):
        interval = value.get("interval", value.get("period"))
        return MeasureProgram(
            start=float(value.get("start", 0.0)),
            end=None if value.get("end") is None else float(value["end"]),
            duration=(
                None
                if value.get("duration") is None
                else float(value["duration"])
            ),
            interval=None if interval is None else float(interval),
            intensity=float(value.get("intensity", 1.0)),
        )
    raise ValueError(f"Не удалось разобрать расписание меры: {value!r}")


def load_model(path: str | Path) -> Model:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    params = ModelParams(**(raw.get("model") or {}))

    nodes: dict[str, Node] = {}
    for node_id, data in (raw.get("nodes") or {}).items():
        data = dict(data or {})
        kind = data.get("kind")
        if kind not in KINDS:
            raise ValueError(f"Вершина {node_id}: неизвестный kind={kind!r}")
        node = Node(
            id=str(node_id),
            kind=kind,
            label=str(data.get("label", node_id)),
            short=str(data.get("short", data.get("label", node_id))),
            unit=str(data.get("unit", "")),
            direction=str(data.get("direction", "up")),
            base=float(data.get("base", 0.0)),
            target=float(data.get("target", 1.0)),
            rise=None if data.get("rise") is None else float(data["rise"]),
            theta=None if data.get("theta") is None else float(data["theta"]),
            mu=None if data.get("mu") is None else float(data["mu"]),
            x_max=None if data.get("x_max") is None else float(data["x_max"]),
            cost=float(data.get("cost", 0.0)),
            owner=str(data.get("owner", "")),
            nominal=float(data.get("nominal", 0.0)),
            monetization=dict(data.get("monetization") or {}),
            position=_parse_position(data.get("position")),
        )
        if node.direction not in {"up", "down"}:
            raise ValueError(f"Вершина {node_id}: direction должен быть up/down")
        nodes[node.id] = node

    edges: list[Edge] = []
    for i, data in enumerate(raw.get("edges") or [], start=1):
        try:
            src = str(data["from"])
            dst = str(data["to"])
        except KeyError as exc:
            raise ValueError(f"Дуга #{i}: отсутствует поле {exc.args[0]!r}") from exc
        edges.append(
            Edge(
                src=src,
                dst=dst,
                w=float(data["w"]),
                delay=float(data.get("delay", 0.0)),
                source_class=str(data.get("source", "")),
                mechanism=str(data.get("mechanism", "")),
                optional=bool(data.get("optional", False)),
            )
        )

    plans: dict[str, Plan] = {}
    for name, data in (raw.get("plans") or {}).items():
        data = dict(data or {})
        programs = {
            measure: _parse_program(schedule)
            for measure, schedule in (data.get("measures") or {}).items()
        }
        plans[str(name)] = Plan(
            name=str(name),
            label=str(data.get("label", name)),
            programs=programs,
        )

    scenarios: dict[str, Scenario] = {}
    for name, data in (raw.get("scenarios") or {}).items():
        data = dict(data or {})
        scenarios[str(name)] = Scenario(
            name=str(name),
            label=str(data.get("label", name)),
            values={k: float(v) for k, v in (data.get("values") or {}).items()},
            atomic=bool(data.get("atomic", True)),
        )

    model = Model(
        params=params,
        nodes=nodes,
        edges=edges,
        plans=plans,
        scenarios=scenarios,
        raw=raw,
    )
    validate_model(model)
    return model


def estimate_plan_cost(
    model: Model,
    plan: Plan,
    cost_mode: str | None = None,
    horizon: int | None = None,
    dt: float | None = None,
) -> float:
    """Стоимость плана в млн руб. с учётом режима списания.

    ``one_time`` списывает стоимость один раз за план; ``monthly`` и
    ``spread`` считаются по фактически активным шагам сетки.
    """
    mode = cost_mode or model.params.cost_mode
    horizon = horizon if horizon is not None else model.params.horizon
    dt = float(dt if dt is not None else model.params.dt)
    total = 0.0
    for measure, program in plan.programs.items():
        node = model.node(measure)
        base = node.cost * program.intensity
        if mode == "one_time":
            total += base
        elif mode == "monthly":
            total += base * program.active_steps(horizon, dt) * dt
        elif mode == "spread":
            total += (
                base
                * program.active_steps(horizon, dt)
                * dt
                / program.spread_duration(horizon)
            )
        else:
            raise ValueError(f"Неизвестный cost_mode={mode!r}")
    return total


def validate_model(model: Model) -> None:
    errors: list[str] = []

    if model.params.saturation not in {"ceiling", "strict"}:
        errors.append("model.saturation должен быть ceiling или strict")
    if model.params.cost_mode not in {"one_time", "monthly", "spread"}:
        errors.append("model.cost_mode должен быть one_time, monthly или spread")
    if model.params.revenue_mode not in {"percent", "rubles"}:
        errors.append("model.revenue_mode должен быть percent или rubles")
    if model.params.horizon <= 0:
        errors.append("model.horizon должен быть > 0")
    if model.params.budget_max < 0:
        errors.append("model.budget_max не может быть отрицательным")
    if model.params.theta < 0:
        errors.append("model.theta не может быть отрицательным")
    if model.params.mu < 0:
        errors.append("model.mu не может быть отрицательным")
    if model.params.target_level <= 0:
        errors.append("model.target_level должен быть > 0")

    if model.params.target_node not in model.nodes:
        errors.append(f"Целевая вершина {model.params.target_node!r} не найдена")
    elif not model.node(model.params.target_node).is_indicator:
        errors.append("Целевая вершина должна быть показателем X")
    elif (
        model.params.saturation == "ceiling"
        and model.params.x_max <= model.params.target_level
    ):
        errors.append(
            "model.x_max должен быть больше target_level, иначе цель недостижима"
        )
    elif (
        model.params.saturation == "ceiling"
        and model.node(model.params.target_node).x_max is not None
        and model.node(model.params.target_node).x_max <= model.params.target_level
    ):
        errors.append(
            f"{model.params.target_node}: x_max должен быть больше target_level"
        )

    for node in model.nodes.values():
        if node.is_indicator and (node.rise is None or node.rise <= 0):
            errors.append(f"{node.id}: для показателя нужен rise > 0")
        if node.is_measure and node.cost < 0:
            errors.append(f"{node.id}: стоимость меры не может быть отрицательной")
        if node.theta is not None and node.theta < 0:
            errors.append(f"{node.id}: theta не может быть отрицательным")
        if node.mu is not None and node.mu < 0:
            errors.append(f"{node.id}: mu не может быть отрицательным")
        if node.x_max is not None and node.x_max <= 0:
            errors.append(f"{node.id}: x_max должен быть > 0")

    seen: set[tuple[str, str]] = set()
    for edge in model.edges:
        if edge.src not in model.nodes:
            errors.append(f"Дуга {edge.src}->{edge.dst}: источник не найден")
            continue
        if edge.dst not in model.nodes:
            errors.append(f"Дуга {edge.src}->{edge.dst}: приёмник не найден")
            continue
        if edge.src == edge.dst:
            errors.append(f"Дуга {edge.src}->{edge.dst}: петля не поддерживается")
        if not model.node(edge.dst).is_indicator:
            errors.append(f"Дуга {edge.src}->{edge.dst}: приёмником должен быть показатель X")
        if edge.delay < 0:
            errors.append(f"Дуга {edge.src}->{edge.dst}: delay < 0")
        key = (edge.src, edge.dst)
        if key in seen:
            errors.append(f"Дуга {edge.src}->{edge.dst} встречается дважды")
        seen.add(key)

    # Бюджет — не структурное ограничение модели: превышение
    # показывается как feasible=False в результате прогона. Иначе
    # редактор графа не позволил бы поднять стоимость меры.
    for plan_name, plan in model.plans.items():
        for measure, program in plan.programs.items():
            if measure not in model.nodes:
                errors.append(f"План {plan_name}: мера {measure!r} не найдена")
                continue
            if not model.node(measure).is_measure:
                errors.append(f"План {plan_name}: {measure!r} не является мерой")
                continue
            if program.start < 0:
                errors.append(
                    f"План {plan_name}, мера {measure}: начало не может быть < 0"
                )
            if program.end is not None and program.end <= program.start:
                errors.append(
                    f"План {plan_name}, мера {measure}: конец должен быть "
                    "позже начала"
                )
            if program.interval is not None and program.interval < 0:
                errors.append(
                    f"План {plan_name}, мера {measure}: периодичность "
                    "не может быть отрицательной"
                )
            if program.duration is not None and program.duration <= 0:
                errors.append(
                    f"План {plan_name}, мера {measure}: длительность "
                    "должна быть > 0"
                )
            if (
                program.interval is not None
                and program.interval > 0
                and program.duration is None
            ):
                errors.append(
                    f"План {plan_name}, мера {measure}: для периодической "
                    "меры нужно задать длительность"
                )

    for scenario_name, scenario in model.scenarios.items():
        for node_id in scenario.values:
            if node_id not in model.nodes:
                errors.append(f"Сценарий {scenario_name}: вершина {node_id!r} не найдена")
            elif not model.node(node_id).is_environment:
                errors.append(f"Сценарий {scenario_name}: {node_id!r} не является фактором среды Y")

    if errors:
        raise ValueError("Модель не прошла валидацию:\n  - " + "\n  - ".join(errors))
