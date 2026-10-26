"""Редактирование графа: строки UI <-> Model, экспорт YAML.

Здесь нет Streamlit. Модуль принимает простые списки словарей (как из
``st.data_editor``), собирает и валидирует новую модель, а также умеет
сериализовать её обратно в формат ``configs/karkas.yaml``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .model import (
    ENVIRONMENT,
    INDICATOR,
    KINDS,
    MEASURE,
    Edge,
    MeasureProgram,
    Model,
    Node,
    Plan,
    Scenario,
    validate_model,
)

NODE_FIELDS = [
    "id",
    "kind",
    "label",
    "base",
    "target",
    "rise",
    "cost",
    "owner",
    "direction",
    "short",
    "unit",
    "x",
    "y",
]

EDGE_FIELDS = ["from", "to", "w", "delay", "optional", "source", "mechanism"]

KIND_LABELS = {
    INDICATOR: "показатель X",
    MEASURE: "мера M",
    ENVIRONMENT: "среда Y",
}


# ---------------------------------------------------------------------------
# Model -> строки таблиц
# ---------------------------------------------------------------------------

def node_rows(model: Model) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in model.nodes.values():
        rows.append(
            {
                "id": node.id,
                "kind": node.kind,
                "label": node.label,
                "short": node.short,
                "unit": node.unit,
                "direction": node.direction,
                "base": node.base,
                "target": node.target,
                "rise": node.rise if node.rise is not None else 0.0,
                "cost": node.cost,
                "owner": node.owner,
                "x": node.position[0] if node.position is not None else None,
                "y": node.position[1] if node.position is not None else None,
            }
        )
    return rows


def edge_rows(model: Model) -> list[dict[str, Any]]:
    return [
        {
            "from": edge.src,
            "to": edge.dst,
            "w": edge.w,
            "delay": edge.delay,
            "optional": edge.optional,
            "source": edge.source_class,
            "mechanism": edge.mechanism,
        }
        for edge in model.edges
    ]


# ---------------------------------------------------------------------------
# Строки таблиц -> Model
# ---------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return bool(isinstance(value, float) and math.isnan(value))


def _as_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    if _is_missing(value):
        return default
    return str(value).strip()


def _as_bool(value: Any, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    return bool(value)


def _position(row: dict[str, Any]) -> tuple[float, float] | None:
    x = row.get("x")
    y = row.get("y")
    if _is_missing(x) or _is_missing(y):
        return None
    return float(x), float(y)


def build_model(
    base: Model,
    node_table: Iterable[dict[str, Any]],
    edge_table: Iterable[dict[str, Any]],
) -> Model:
    """Собрать новую модель из отредактированных таблиц.

    Параметры модели, планы и сценарии берутся из ``base``. Планы
    очищаются от удалённых/переклассифицированных мер, сценарии — от
    удалённых факторов среды. Бюджет здесь не проверяется: превышение
    бюджета — это результат прогона, а не ошибка графа.
    """
    nodes: dict[str, Node] = {}
    for row in node_table:
        node_id = _as_str(row.get("id"))
        if not node_id:
            continue
        if node_id in nodes:
            raise ValueError(f"Вершина {node_id!r} встречается в таблице дважды.")

        kind = _as_str(row.get("kind"))
        if kind not in KINDS:
            raise ValueError(
                f"Вершина {node_id}: неизвестный тип {kind!r}. "
                f"Допустимо: {', '.join(sorted(KINDS))}."
            )

        direction = _as_str(row.get("direction"), "up")
        if direction not in {"up", "down"}:
            raise ValueError(
                f"Вершина {node_id}: direction должен быть up или down, "
                f"получено {direction!r}."
            )

        original = base.nodes.get(node_id)
        rise = _as_float(row.get("rise"), 0.0)
        label = _as_str(row.get("label")) or node_id
        short = _as_str(row.get("short")) or label

        nodes[node_id] = Node(
            id=node_id,
            kind=kind,
            label=label,
            short=short,
            unit=_as_str(row.get("unit")),
            direction=direction,
            base=_as_float(row.get("base"), 0.0),
            target=_as_float(row.get("target"), 0.0),
            rise=rise if rise > 0 else None,
            # Скрытые от таблицы поля сохраняем у существующих вершин.
            theta=original.theta if original is not None else None,
            mu=original.mu if original is not None else None,
            x_max=original.x_max if original is not None else None,
            cost=_as_float(row.get("cost"), 0.0),
            owner=_as_str(row.get("owner"), original.owner if original is not None else ""),
            nominal=original.nominal if original is not None else 0.0,
            monetization=dict(original.monetization) if original is not None else {},
            position=_position(row),
        )

    if not nodes:
        raise ValueError("В графе не осталось ни одной вершины.")

    edges: list[Edge] = []
    for row in edge_table:
        src = _as_str(row.get("from"))
        dst = _as_str(row.get("to"))
        if not src and not dst:
            continue
        if not src or not dst:
            raise ValueError("У дуги должны быть заполнены обе вершины: «из» и «в».")
        if src not in nodes:
            raise ValueError(f"Дуга {src}->{dst}: вершина-источник {src!r} не найдена.")
        if dst not in nodes:
            raise ValueError(f"Дуга {src}->{dst}: вершина-приёмник {dst!r} не найдена.")

        w = _as_float(row.get("w"), float("nan"))
        if not math.isfinite(w):
            raise ValueError(f"Дуга {src}->{dst}: вес должен быть числом.")
        delay = _as_float(row.get("delay"), 0.0)

        original_edge = next(
            (e for e in base.edges if e.src == src and e.dst == dst), None
        )
        edges.append(
            Edge(
                src=src,
                dst=dst,
                w=w,
                delay=delay,
                source_class=_as_str(
                    row.get("source"),
                    original_edge.source_class if original_edge is not None else "",
                ),
                mechanism=_as_str(
                    row.get("mechanism"),
                    original_edge.mechanism if original_edge is not None else "",
                ),
                optional=_as_bool(row.get("optional"), False),
            )
        )

    plans: dict[str, Plan] = {}
    for name, plan in base.plans.items():
        programs = {
            measure: MeasureProgram(
                start=program.start,
                duration=program.duration,
                intensity=program.intensity,
            )
            for measure, program in plan.programs.items()
            if measure in nodes and nodes[measure].kind == MEASURE
        }
        plans[name] = Plan(name=name, label=plan.label, programs=programs)

    scenarios: dict[str, Scenario] = {}
    for name, scenario in base.scenarios.items():
        values = {
            node_id: value
            for node_id, value in scenario.values.items()
            if node_id in nodes and nodes[node_id].kind == ENVIRONMENT
        }
        scenarios[name] = Scenario(
            name=name,
            label=scenario.label,
            values=values,
            atomic=scenario.atomic,
        )

    model = Model(
        params=base.params,
        nodes=nodes,
        edges=edges,
        plans=plans,
        scenarios=scenarios,
        raw=base.raw,
    )
    validate_model(model)
    return model


# ---------------------------------------------------------------------------
# Model -> YAML
# ---------------------------------------------------------------------------

def model_to_dict(model: Model) -> dict[str, Any]:
    return {
        "model": asdict(model.params),
        "nodes": {
            node.id: {
                "kind": node.kind,
                "label": node.label,
                "short": node.short,
                "unit": node.unit,
                "direction": node.direction,
                "base": node.base,
                "target": node.target,
                "rise": node.rise,
                "theta": node.theta,
                "mu": node.mu,
                "x_max": node.x_max,
                "cost": node.cost,
                "owner": node.owner,
                "nominal": node.nominal,
                "monetization": dict(node.monetization),
                "position": list(node.position) if node.position is not None else None,
            }
            for node in model.nodes.values()
        },
        "edges": [
            {
                "from": edge.src,
                "to": edge.dst,
                "w": edge.w,
                "delay": edge.delay,
                "source": edge.source_class,
                "mechanism": edge.mechanism,
                "optional": edge.optional,
            }
            for edge in model.edges
        ],
        "plans": {
            name: {
                "label": plan.label,
                "measures": {
                    measure: {
                        "start": program.start,
                        "duration": program.duration,
                        "intensity": program.intensity,
                    }
                    for measure, program in plan.programs.items()
                },
            }
            for name, plan in model.plans.items()
        },
        "scenarios": {
            name: {
                "label": scenario.label,
                "values": dict(scenario.values),
                "atomic": scenario.atomic,
            }
            for name, scenario in model.scenarios.items()
        },
    }


def model_to_yaml(model: Model) -> str:
    header = (
        "# Экспорт графа из Streamlit-редактора.\n"
        f"# Дата: {datetime.now().isoformat(timespec='seconds')}\n"
        "# Комментарии исходного конфига не сохраняются.\n\n"
    )
    payload = yaml.safe_dump(
        model_to_dict(model),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )
    return header + payload


def save_model_yaml(model: Model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model_to_yaml(model), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Мелкие помощники для формы добавления вершины
# ---------------------------------------------------------------------------

def suggest_node_id(model: Model, kind: str) -> str:
    if kind == MEASURE:
        prefix = "m"
    elif kind == ENVIRONMENT:
        prefix = "Y"
    elif kind == INDICATOR:
        prefix = "X"
    else:
        raise ValueError(f"Неизвестный тип вершины: {kind!r}")

    numbers = []
    for node in model.nodes.values():
        if node.id.startswith(prefix):
            tail = node.id[len(prefix):]
            if tail.isdigit():
                numbers.append(int(tail))
    return f"{prefix}{max(numbers, default=0) + 1}"


def validate_new_node_id(node_id: str, existing_ids: Iterable[str]) -> str:
    node_id = node_id.strip()
    if not node_id:
        raise ValueError("ID вершины не может быть пустым.")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", node_id):
        raise ValueError(
            "ID должен начинаться с латинской буквы и содержать только "
            "латинские буквы, цифры и подчёркивания."
        )
    if node_id in set(existing_ids):
        raise ValueError(f"Вершина с ID {node_id!r} уже существует.")
    return node_id


def default_position(model: Model, kind: str) -> tuple[float, float]:
    if kind == MEASURE:
        x = 0.0
    elif kind == ENVIRONMENT:
        x = 11.0
    else:
        x = 5.6

    ys = [
        node.position[1]
        for node in model.nodes.values()
        if node.kind == kind and node.position is not None
    ]
    y = max(ys) + 1.8 if ys else 0.0
    return x, y


def new_node_row(
    model: Model,
    node_id: str,
    kind: str,
    *,
    label: str = "",
    short: str = "",
    unit: str = "",
    direction: str = "up",
    base: float = 0.0,
    target: float = 0.0,
    rise: float = 0.0,
    cost: float = 0.0,
    owner: str = "",
) -> dict[str, Any]:
    x, y = default_position(model, kind)
    label = label or node_id
    return {
        "id": node_id,
        "kind": kind,
        "label": label,
        "short": short or label,
        "unit": unit,
        "direction": direction,
        "base": base,
        "target": target,
        "rise": rise,
        "cost": cost,
        "owner": owner,
        "x": x,
        "y": y,
    }
