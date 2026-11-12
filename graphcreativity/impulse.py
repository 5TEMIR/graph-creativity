"""Импульсные отклики графа: кто на кого влияет и с какой задержкой.

Модуль не меняет движок. Импульс задаётся одним из трёх способов:

* начальным состоянием вершины (``initial_states``) --- для факторов
  и исходов ``i``/``o``;
* включением меры с нулевого месяца --- для ``p``;
* ступенькой среды (``environment_series``) --- для ``y``.

Отклик --- отклонение траектории от базового прогона (нулевые меры,
нулевая среда). Он не является прогнозом: это проверка структуры графа,
а не калибровка.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .engine import SimOptions, SimulationResult, simulate
from .model import Model

DEFAULT_TOL = 1e-3


@dataclass(frozen=True)
class Response:
    """Отклик одной вершины на один импульс."""

    source: str
    target: str
    peak: float
    peak_value: float
    peak_month: int
    final: float
    first_month: int | None

    @property
    def sign(self) -> int:
        if self.peak_value > 0:
            return 1
        if self.peak_value < 0:
            return -1
        return 0


def steps_of(model: Model, options: SimOptions | None = None) -> int:
    """Число шагов сетки --- так же, как в движке."""
    opts = (options or SimOptions()).resolve(model)
    if opts.dt <= 0:
        raise ValueError("dt должен быть > 0")
    return round(opts.horizon / opts.dt) + 1


def run_baseline(
    model: Model,
    options: SimOptions | None = None,
) -> SimulationResult:
    """Базовый прогон: без мер, среда на нуле."""
    return simulate(model, None, "nominal", options)


def run_state_shock(
    model: Model,
    node: str,
    size: float,
    options: SimOptions | None = None,
) -> SimulationResult:
    """Импульс фактору или исходу: состояние ``node`` в месяц 0 равно ``size``."""
    return simulate(
        model,
        None,
        "nominal",
        options,
        initial_states={node: size},
    )


def run_measure(
    model: Model,
    measure: str,
    options: SimOptions | None = None,
) -> SimulationResult:
    """Импульс мере: она включена с нулевого месяца на всю кампанию."""
    return simulate(model, None, "nominal", options, measures=[measure])


def run_environment_shock(
    model: Model,
    node: str,
    size: float,
    options: SimOptions | None = None,
) -> SimulationResult:
    """Ступенька среды: ``y`` равен ``size`` все месяцы."""
    steps = steps_of(model, options)
    return simulate(
        model,
        None,
        "nominal",
        options,
        environment_series={node: [float(size)] * steps},
    )


def deviations(
    result: SimulationResult,
    reference: SimulationResult,
    nodes: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    """Отклонение траекторий от базового прогона по каждой вершине."""
    targets = list(nodes) if nodes is not None else list(result.states)
    return {
        node: [
            value - reference.states[node][index]
            for index, value in enumerate(result.states[node])
        ]
        for node in targets
    }


def _first_crossing(values: Iterable[float], tol: float) -> int | None:
    for index, value in enumerate(values):
        if abs(value) > tol:
            return index
    return None


def respond(
    source: str,
    result: SimulationResult,
    reference: SimulationResult,
    targets: Sequence[str],
    *,
    tol: float = DEFAULT_TOL,
) -> list[Response]:
    """Сводка отклика ``result`` относительно ``reference`` по всем целям."""
    delta = deviations(result, reference, targets)
    responses: list[Response] = []
    for target in targets:
        values = delta[target]
        peak_month = max(range(len(values)), key=lambda i: abs(values[i]))
        responses.append(
            Response(
                source=source,
                target=target,
                peak=abs(values[peak_month]),
                peak_value=values[peak_month],
                peak_month=peak_month,
                final=values[-1],
                first_month=_first_crossing(values, tol),
            )
        )
    return responses


def influence_matrix(
    responses: Iterable[Response],
    sources: Sequence[str],
    targets: Sequence[str],
) -> list[list[float]]:
    """Матрица «источник × цель» по пиковому отклонению."""
    table = {(item.source, item.target): item.peak for item in responses}
    return [
        [table.get((source, target), 0.0) for target in targets]
        for source in sources
    ]


def first_response(responses: Iterable[Response]) -> int | None:
    """Самый ранний отклик во всём наборе."""
    months = [item.first_month for item in responses if item.first_month is not None]
    return min(months) if months else None
