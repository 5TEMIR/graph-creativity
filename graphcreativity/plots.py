"""Графики: динамика, сравнение планов, топология графа."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch

from .engine import SimulationResult
from .model import Model

COLORS = {
    "X1": "#1F4E79",
    "X2": "#2E75B6",
    "X3": "#548235",
    "X4": "#BF8F00",
    "X7": "#7030A0",
}
NEGATIVE = "#A61C00"
POSITIVE = "#1F4E79"
OPTIONAL = "#808080"
KIND_COLORS = {
    "measure": "#DDEEDC",
    "indicator": "#DCE9F7",
    "environment": "#FBE6CC",
}
KIND_EDGE = {
    "measure": "#548235",
    "indicator": "#1F4E79",
    "environment": "#BF8F00",
}


def _effect_pct(node, physical: float) -> float:
    if node.base == 0:
        return 0.0
    delta = physical - node.base if node.direction == "up" else node.base - physical
    return 100.0 * delta / node.base


def _target_pct(node) -> float:
    return 100.0 * abs(node.target - node.base) / node.base if node.base else 0.0


def _save(fig, path: str | Path | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_dynamics(result: SimulationResult, path: str | Path | None = None):
    """Главный график: цель, все показатели, меры, экономика."""
    model = result.model
    t = result.time
    target = model.node(result.params.target_node)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.suptitle(
        f"Динамика: {result.plan.label or result.plan.name} · "
        f"сценарий «{result.scenario.label or result.scenario.name}»",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Целевой показатель в процентах к базе
    ax = axes[0, 0]
    pct = [_effect_pct(target, v) for v in result.physical[target.id]]
    ax.plot(t, pct, color=COLORS.get(target.id, POSITIVE), lw=2.4, label=target.short or target.id)
    ax.axhline(_target_pct(target), color="#C00000", ls="--", lw=1.2, label="цель")
    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    t_goal = result.first_goal_time()
    if t_goal is not None:
        ax.axvline(t_goal, color="#548235", ls=":", lw=1.4)
        ax.annotate(
            f"цель достигнута: {t_goal:g} мес.",
            xy=(t_goal, _target_pct(target)),
            xytext=(6, -24),
            textcoords="offset points",
            fontsize=9,
            color="#375623",
        )
    ax.set_title(f"{target.label} — прирост к базе, %")
    ax.set_xlabel("месяц")
    ax.set_ylabel("% к базе")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")

    # 2. Все показатели в нормированном виде
    ax = axes[0, 1]
    for node in model.indicators:
        ax.plot(
            t,
            result.states[node.id],
            lw=2 if node.id == target.id else 1.6,
            color=COLORS.get(node.id),
            label=node.short or node.id,
        )
    ax.axhline(1.0, color="#C00000", ls="--", lw=1.1, label="цель x=1")
    ax.set_title("Все показатели, нормированные (0 — база, 1 — цель)")
    ax.set_xlabel("месяц")
    ax.set_ylabel("x, доля пути до цели")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper left")

    # 3. Меры и среда
    ax = axes[1, 0]
    for measure_id in model.measure_ids:
        series = result.measures.get(measure_id, [])
        ax.step(t, series, where="post", lw=1.9, label=measure_id)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Профили мер u_j(t)")
    ax.set_xlabel("месяц")
    ax.set_ylabel("интенсивность")
    ax.grid(alpha=0.25)
    if model.measure_ids:
        ax.legend(fontsize=9, loc="upper right")
    env_text = ", ".join(
        f"{env_id}={result.environment.get(env_id, [0])[0]:+.2f}"
        for env_id in model.environment_ids
    )
    if env_text:
        ax.text(
            0.02,
            0.04,
            "среда: " + env_text,
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )

    # 4. Экономика
    ax = axes[1, 1]
    ax.plot(t, result.revenue, color="#548235", lw=2.0, label="R(t)")
    ax.set_xlabel("месяц")
    ax.set_ylabel("R, % прироста" if result.params.revenue_mode == "percent" else "R, млн руб.")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(t, result.cost_cum, color="#A61C00", lw=2.0, label="C накопл.")
    ax2.set_ylabel("C, млн руб.")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    last = result.last_index
    k = (result.revenue[last] / result.cost_cum[last]) if result.cost_cum[last] > 0 else None
    if k is not None:
        ax.text(
            0.03,
            0.72,
            f"k = R/C = {k:.2f}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color="#1F4E79",
        )
    ax.set_title("Эффект R(t), затраты C(t) и критерий k")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)


def plot_indicators(result: SimulationResult, path: str | Path | None = None):
    """Small multiples: каждый показатель в физических единицах."""
    model = result.model
    nodes = model.indicators
    ncols = 3
    nrows = math.ceil(len(nodes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows), squeeze=False)
    fig.suptitle(
        f"Показатели в физических единицах · {result.plan.label or result.plan.name} · "
        f"«{result.scenario.label or result.scenario.name}»",
        fontsize=13,
        fontweight="bold",
    )
    for idx, node in enumerate(nodes):
        ax = axes[idx // ncols][idx % ncols]
        values = result.physical[node.id]
        ax.plot(result.time, values, color=COLORS.get(node.id, POSITIVE), lw=2.2)
        ax.axhline(node.base, color="#777777", ls=":", lw=1.0, label="база")
        ax.axhline(node.target, color="#C00000", ls="--", lw=1.0, label="цель")
        ax.set_title(f"{node.short or node.id} ({node.unit})", fontsize=10)
        ax.set_xlabel("месяц")
        ax.grid(alpha=0.25)
        final = values[-1]
        ax.annotate(
            f"{final:.1f}",
            xy=(result.time[-1], final),
            xytext=(-4, 8),
            textcoords="offset points",
            fontsize=9,
            ha="right",
            fontweight="bold",
        )
        if idx == 0:
            ax.legend(fontsize=8, loc="best")
    for idx in range(len(nodes), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def plot_graph(
    model: Model,
    active_measures: Iterable[str] | None = None,
    path: str | Path | None = None,
    include_optional: bool = False,
):
    """Топология каркаса с весами и знаками дуг."""
    active = set(active_measures or [])
    positions: dict[str, tuple[float, float]] = {}
    for idx, node in enumerate(model.nodes.values()):
        if node.position is not None:
            positions[node.id] = node.position
        else:
            angle = 2 * math.pi * idx / max(len(model.nodes), 1)
            positions[node.id] = (5 * math.cos(angle), 5 * math.sin(angle))

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    for edge in model.edges:
        if edge.optional and not include_optional:
            continue
        x1, y1 = positions[edge.src]
        x2, y2 = positions[edge.dst]
        color = POSITIVE if edge.w >= 0 else NEGATIVE
        style = "--" if edge.optional else "-"
        lw = 0.7 + 2.6 * min(abs(edge.w), 1.0)
        alpha = 1.0
        if edge.src in model.measure_ids and active and edge.src not in active:
            alpha = 0.25
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            connectionstyle="arc3,rad=0.06",
            linewidth=lw,
            linestyle=style,
            color=color,
            alpha=alpha,
            shrinkA=22,
            shrinkB=22,
            zorder=1,
        )
        ax.add_patch(arrow)
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy) or 1.0
        # сдвигаем подпись в сторону от линии, чтобы не перекрывать дугу
        lx = x1 + 0.42 * dx - 0.20 * dy / norm
        ly = y1 + 0.42 * dy + 0.20 * dx / norm
        ax.text(
            lx,
            ly,
            f"{edge.w:+.2f}",
            fontsize=8.5,
            ha="center",
            va="center",
            color=color,
            alpha=alpha,
            zorder=3,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
        )

    for node in model.nodes.values():
        x, y = positions[node.id]
        is_active = node.id in active
        ax.scatter(
            [x],
            [y],
            s=2600 if node.id == model.params.target_node else 1800,
            c=KIND_COLORS[node.kind],
            edgecolors=KIND_EDGE[node.kind] if not is_active else "#375623",
            linewidths=3.5 if is_active else 1.4,
            zorder=2,
        )
        ax.text(x, y, node.id, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)

    groups = {
        "меры M": [n.id for n in model.measures],
        "показатели X": [n.id for n in model.indicators],
        "среда Y": [n.id for n in model.environments],
    }
    for label, ids in groups.items():
        xs = [positions[i][0] for i in ids]
        ys = [positions[i][1] for i in ids]
        if xs:
            ax.text(
                sum(xs) / len(xs),
                max(ys) + 1.15,
                label,
                ha="center",
                fontsize=10,
                fontweight="bold",
                color="#444444",
            )

    ax.set_title("Топология каркаса: толщина — |w|, синий — плюс, красный — минус", fontsize=12)
    ax.legend(
        handles=[
            Patch(facecolor=KIND_COLORS["measure"], edgecolor=KIND_EDGE["measure"], label="мера"),
            Patch(facecolor=KIND_COLORS["indicator"], edgecolor=KIND_EDGE["indicator"], label="показатель"),
            Patch(facecolor=KIND_COLORS["environment"], edgecolor=KIND_EDGE["environment"], label="среда"),
        ],
        loc="lower left",
        fontsize=8,
    )
    ax.set_xlim(min(x for x, _ in positions.values()) - 1.5, max(x for x, _ in positions.values()) + 1.5)
    ax.set_ylim(min(y for _, y in positions.values()) - 1.5, max(y for _, y in positions.values()) + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, path)


def plot_compare(rows: list[dict], path: str | Path | None = None):
    """Сравнение подмножеств мер: k и плоскость T_goal/стоимость."""
    feasible = [r for r in rows if r.get("feasible")]
    admissible = [
        r for r in feasible if r.get("reached") and r.get("k_T") is not None
    ]
    admissible.sort(key=lambda r: -(r.get("k_T") or 0))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.suptitle("Сравнение комплексов мер", fontsize=13, fontweight="bold")

    ax = axes[0]
    labels = [r["plan"] for r in admissible]
    values = [r.get("k_T") or 0.0 for r in admissible]
    ax.barh(range(len(labels)), values, color="#548235")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("k = R/C, %/млн руб.")
    ax.set_title("Критерий эффект/затраты (только допустимые планы)")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    for r in feasible:
        if not r.get("reached"):
            continue
        ax.scatter(
            r["cost_total"],
            r["t_goal"],
            s=60 + 260 * min(abs(r.get("k_T") or 0), 3),
            color="#1F4E79",
            alpha=0.8,
            edgecolors="white",
        )
        ax.annotate(r["plan"], (r["cost_total"], r["t_goal"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("стоимость, млн руб.")
    ax.set_ylabel("T_goal, мес.")
    ax.set_title("Достигнувшие цели: быстрее и дешевле — лучше")
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)
