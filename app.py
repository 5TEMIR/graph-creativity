"""Streamlit-приложение: графовая модель креативности.

Интерфейс исследователя: все подмножества мер, σ₁, чувствительность
и редактирование графа. Основной расчёт идёт по базовой среде.

Запуск:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from graphcreativity.editing import (
    EDGE_FIELDS,
    KIND_LABELS,
    build_model,
    edge_rows,
    model_to_yaml,
    new_node_row,
    node_rows,
    plan_rows,
    save_model_yaml,
    suggest_node_id,
    validate_new_node_id,
)
from graphcreativity.engine import SimOptions, simulate
from graphcreativity.metrics import (
    compare_named_plans,
    compare_plans,
    critical_removal_sets,
    effect_strength,
    minimal_sufficient_sets,
    pareto_front,
    summarize,
)
from graphcreativity.model import (
    ENVIRONMENT,
    INDICATOR,
    MEASURE,
    MeasureProgram,
    Plan,
    load_model,
)
from graphcreativity.plots import (
    plot_compare,
    plot_dynamics,
    plot_graph,
    plot_indicators,
)

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"

st.set_page_config(
    page_title="Креативность · графовая модель",
    page_icon="🕸️",
    layout="wide",
)


def show_fig(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def fmt(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    return f"{value:.{digits}f}{suffix}"


def num(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _row_by_id(node_id: str) -> dict | None:
    return next(
        (
            row
            for row in st.session_state.get("node_rows", [])
            if str(row.get("id", "")).strip() == node_id
        ),
        None,
    )


def _finish_graph_edit(message: str) -> None:
    st.session_state.graph_epoch += 1
    st.session_state.graph_flash = message
    st.rerun()


CUSTOM_PLAN = "__custom__"


def _is_all_measures_plan(plan: Plan) -> bool:
    return plan.name == "full" or plan.label.strip().casefold() == "все меры"


def _editable_plan_names(model) -> list[str]:
    return [
        name
        for name, plan in model.plans.items()
        if not _is_all_measures_plan(plan)
    ]


def _default_plan_name(model) -> str:
    names = _editable_plan_names(model)
    if "hybrid" in names:
        return "hybrid"
    return names[0] if names else CUSTOM_PLAN


def _base_scenario_name(model) -> str:
    """Сценарий, который Streamlit использует для основного расчёта."""
    if "nominal" in model.scenarios:
        return "nominal"
    return next(iter(model.scenarios), "nominal")


def _base_environment_values(model, scenario_name: str) -> dict[str, float]:
    if scenario_name not in model.scenarios:
        return {}
    return dict(model.scenario(scenario_name).values)


def _initial_custom_programs(model, plan_table: dict) -> dict[str, dict]:
    default_name = _default_plan_name(model)
    source_name = default_name if default_name in plan_table else None
    if source_name is None and "full" in plan_table:
        source_name = "full"
    if source_name is None:
        return {}
    return {
        measure: dict(program)
        for measure, program in plan_table[source_name].items()
        if measure in model.measure_ids
    }


def _sync_custom_state(model) -> dict[str, dict]:
    """Согласовать выбранные меры своего комплекса и их расписания."""
    valid = set(model.measure_ids)
    programs = st.session_state.get("custom_plan_programs", {})
    if not isinstance(programs, dict):
        programs = {}
    programs = {
        measure: dict(program)
        for measure, program in programs.items()
        if measure in valid and isinstance(program, dict)
    }
    if "custom_plan_measures" not in st.session_state:
        st.session_state.custom_plan_measures = list(programs)
    selected = [
        measure
        for measure in st.session_state.custom_plan_measures
        if measure in valid
    ]
    for measure in selected:
        programs.setdefault(
            measure,
            {"start": 0.0, "duration": None, "intensity": 1.0},
        )
    programs = {measure: programs[measure] for measure in selected}
    st.session_state.custom_plan_programs = programs
    st.session_state.custom_plan_measures = selected
    return programs


def _build_active_plan(model) -> Plan:
    name = st.session_state.get("active_plan_name")
    if name != CUSTOM_PLAN and (
        name not in model.plans or _is_all_measures_plan(model.plans[name])
    ):
        name = _default_plan_name(model)
        st.session_state.active_plan_name = name
    if name == CUSTOM_PLAN:
        programs = {}
        for measure, data in _sync_custom_state(model).items():
            duration = data.get("duration")
            programs[measure] = MeasureProgram(
                start=float(data.get("start", 0.0)),
                duration=None if duration is None else float(duration),
                intensity=float(data.get("intensity", 1.0)),
            )
        return Plan(name="custom", label="Свой комплекс мер", programs=programs)
    return model.plan(name)


@st.dialog("Показатель")
def edit_indicator_dialog(node_id: str, epoch: int) -> None:
    row = _row_by_id(node_id)
    if row is None:
        st.error(f"Вершина {node_id} не найдена.")
        return

    with st.form(f"indicator_dialog_{node_id}_{epoch}"):
        st.caption(f"ID: `{node_id}` · тип: показатель X")
        label = st.text_input("Название", value=str(row.get("label") or ""))
        c_short, c_unit = st.columns(2)
        short = c_short.text_input("Короткое имя", value=str(row.get("short") or ""))
        unit = c_unit.text_input("Единица измерения", value=str(row.get("unit") or ""))
        direction = st.selectbox(
            "Направление",
            ["up", "down"],
            index=0 if str(row.get("direction", "up")) == "up" else 1,
            format_func=lambda d: "up — больше лучше" if d == "up" else "down — меньше лучше",
        )
        c_base, c_target, c_rise = st.columns(3)
        base = c_base.number_input(
            "База", value=float(row.get("base") or 0.0), format="%.2f"
        )
        target = c_target.number_input(
            "Цель", value=float(row.get("target") or 0.0), format="%.2f"
        )
        rise = c_rise.number_input(
            "rise, мес.",
            value=float(row.get("rise") or 0.0),
            min_value=0.0,
            step=1.0,
            help="Время до плато; 0 — оставить автоматический расчёт по умолчанию.",
        )
        with st.expander("Положение на графе"):
            c_x, c_y = st.columns(2)
            x = c_x.number_input(
                "x", value=float(row.get("x") or 0.0), step=1.0
            )
            y = c_y.number_input(
                "y", value=float(row.get("y") or 0.0), step=1.0
            )

        if st.form_submit_button("Сохранить", type="primary"):
            row.update(
                {
                    "label": label.strip() or node_id,
                    "short": short.strip() or label.strip() or node_id,
                    "unit": unit.strip(),
                    "direction": direction,
                    "base": float(base),
                    "target": float(target),
                    "rise": float(rise),
                    "x": float(x),
                    "y": float(y),
                }
            )
            _finish_graph_edit(f"Показатель {node_id} обновлён.")


@st.dialog("Мера")
def edit_measure_dialog(
    node_id: str,
    plan_name: str,
    plan_label: str,
    epoch: int,
    horizon: int,
    *,
    custom: bool,
) -> None:
    row = _row_by_id(node_id)
    if row is None:
        st.error(f"Вершина {node_id} не найдена.")
        return

    if custom:
        plan_data = st.session_state.setdefault("custom_plan_programs", {})
    else:
        plan_data = st.session_state.plan_rows.setdefault(plan_name, {})
    program = plan_data.get(node_id)
    included = isinstance(program, dict)
    start = round(float(program.get("start", 0))) if included else 0
    intensity = float(program.get("intensity", 1.0)) if included else 1.0
    duration = program.get("duration") if included else None
    until_end = duration is None
    duration_value = round(float(duration)) if duration is not None else horizon

    with st.form(f"measure_dialog_{node_id}_{plan_name}_{epoch}"):
        st.caption(f"ID: `{node_id}` · мера M · комплекс: {plan_label}")
        label = st.text_input("Название", value=str(row.get("label") or ""))
        c_short, c_unit = st.columns(2)
        short = c_short.text_input("Короткое имя", value=str(row.get("short") or ""))
        unit = c_unit.text_input("Единица измерения", value=str(row.get("unit") or ""))
        c_cost, c_owner = st.columns(2)
        cost = c_cost.number_input(
            "Стоимость, млн руб.",
            value=float(row.get("cost") or 0.0),
            min_value=0.0,
            step=1.0,
        )
        owner = c_owner.text_input(
            "Ответственный", value=str(row.get("owner") or "")
        )

        widget_suffix = f"{node_id}_{plan_name}_{epoch}_{horizon}"
        if included:
            start_value = st.slider(
                "Начало, мес.",
                min_value=0,
                max_value=max(1, horizon),
                value=min(max(start, 0), max(1, horizon)),
                step=1,
                key=f"measure_start_{widget_suffix}",
            )
            intensity_value = st.slider(
                "Интенсивность",
                min_value=0.0,
                max_value=1.0,
                value=min(max(intensity, 0.0), 1.0),
                step=0.05,
                key=f"measure_intensity_{widget_suffix}",
                help="Доля полного объёма меры: 1.0 — полный объём.",
            )
            until_end_value = st.checkbox(
                "Действует до конца периода",
                value=until_end,
                key=f"measure_until_end_{widget_suffix}",
            )
            duration_input = st.number_input(
                "Продолжительность, мес.",
                min_value=1.0,
                max_value=float(max(1, horizon)),
                value=float(min(max(duration_value, 1), max(1, horizon))),
                step=1.0,
                key=f"measure_duration_{widget_suffix}",
                help="Не используется, если мера действует до конца периода.",
            )
        else:
            if custom:
                st.info(
                    "Мера не входит в свой комплекс. Отметьте её в списке "
                    "мер выше, чтобы задать расписание."
                )
            else:
                st.info(
                    "Мера не входит в этот именованный план. Состав плана "
                    "фиксирован; для выбора мер используйте «Свой комплекс мер»."
                )
            start_value = 0
            intensity_value = 1.0
            until_end_value = True
            duration_input = float(max(1, horizon))

        with st.expander("Положение на графе"):
            c_x, c_y = st.columns(2)
            x = c_x.number_input(
                "x", value=float(row.get("x") or 0.0), step=1.0
            )
            y = c_y.number_input(
                "y", value=float(row.get("y") or 0.0), step=1.0
            )

        if st.form_submit_button("Сохранить", type="primary"):
            row.update(
                {
                    "label": label.strip() or node_id,
                    "short": short.strip() or label.strip() or node_id,
                    "unit": unit.strip(),
                    "cost": float(cost),
                    "owner": owner.strip(),
                    "x": float(x),
                    "y": float(y),
                }
            )
            if included:
                plan_data[node_id] = {
                    "start": float(start_value),
                    "duration": None if until_end_value else float(duration_input),
                    "intensity": float(intensity_value),
                }
                if custom:
                    st.session_state.custom_plan_programs = plan_data
                else:
                    st.session_state.plan_rows[plan_name] = plan_data
            target = "своего комплекса мер" if custom else f"плана «{plan_label}»"
            _finish_graph_edit(f"Мера {node_id} обновлена для {target}.")


@st.dialog("Фактор среды")
def edit_environment_dialog(node_id: str, epoch: int) -> None:
    row = _row_by_id(node_id)
    if row is None:
        st.error(f"Вершина {node_id} не найдена.")
        return

    values = st.session_state.environment_values
    current = float(values.get(node_id, 0.0))

    with st.form(f"environment_dialog_{node_id}_{epoch}"):
        st.caption(f"ID: `{node_id}` · фактор среды Y")
        label = st.text_input("Название", value=str(row.get("label") or ""))
        c_short, c_unit = st.columns(2)
        short = c_short.text_input("Короткое имя", value=str(row.get("short") or ""))
        unit = c_unit.text_input("Единица измерения", value=str(row.get("unit") or ""))
        direction = st.selectbox(
            "Направление",
            ["up", "down"],
            index=0 if str(row.get("direction", "up")) == "up" else 1,
            format_func=lambda d: "up — больше лучше" if d == "up" else "down — меньше лучше",
            help="Для факторов среды направление справочное и не влияет на σ.",
        )
        low = min(-1.0, current)
        high = max(1.0, current)
        deviation = st.slider(
            "Отклонение от номинала",
            min_value=low,
            max_value=high,
            value=min(max(current, low), high),
            step=0.05,
            key=f"environment_deviation_{node_id}_{epoch}",
            help="Безразмерное отклонение: 0 — номинал, отрицательное — неблагоприятное.",
        )
        with st.expander("Положение на графе"):
            c_x, c_y = st.columns(2)
            x = c_x.number_input(
                "x", value=float(row.get("x") or 0.0), step=1.0
            )
            y = c_y.number_input(
                "y", value=float(row.get("y") or 0.0), step=1.0
            )

        if st.form_submit_button("Сохранить", type="primary"):
            row.update(
                {
                    "label": label.strip() or node_id,
                    "short": short.strip() or label.strip() or node_id,
                    "unit": unit.strip(),
                    "direction": direction,
                    "x": float(x),
                    "y": float(y),
                }
            )
            values[node_id] = float(deviation)
            st.session_state.environment_values = values
            _finish_graph_edit(f"Фактор {node_id} обновлён.")


# ---------------------------------------------------------------------------
# Загрузка модели
# ---------------------------------------------------------------------------

configs = sorted(
    list(CONFIG_DIR.glob("*.yaml")) + list(CONFIG_DIR.glob("*.yml"))
)
if not configs:
    st.error("В configs/ нет YAML-файлов.")
    st.stop()

default_config_index = next(
    (i for i, path in enumerate(configs) if path.name == "karkas.yaml"),
    0,
)

with st.sidebar:
    st.title("🕸️ Графовая модель")
    st.caption("Повышение креативности предприятия")
    config_path = st.selectbox(
        "Конфиг",
        configs,
        index=default_config_index,
        format_func=lambda p: p.name,
        help="Модель, дуги, параметры и планы — всё в YAML.",
    )
    try:
        base_model = load_model(config_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Ошибка загрузки модели:\n\n{exc}")
        st.stop()

    active_scenario_name = _base_scenario_name(base_model)

    # Состояние редактора графа. При смене конфига сбрасываем его.
    if st.session_state.get("graph_config") != str(config_path):
        st.session_state.graph_config = str(config_path)
        st.session_state.graph_epoch = st.session_state.get("graph_epoch", 0) + 1
        st.session_state.node_rows = node_rows(base_model)
        st.session_state.edge_rows = edge_rows(base_model)
        st.session_state.plan_rows = plan_rows(base_model)
        st.session_state.environment_values = _base_environment_values(
            base_model, active_scenario_name
        )
        st.session_state.active_plan_name = _default_plan_name(base_model)
        st.session_state.custom_plan_programs = _initial_custom_programs(
            base_model, st.session_state.plan_rows
        )
        st.session_state.custom_plan_measures = list(
            st.session_state.custom_plan_programs
        )
        st.session_state.last_valid_model = base_model

    # Совместимость с сессией, открытой до появления редакторов планов.
    st.session_state.setdefault("plan_rows", plan_rows(base_model))
    st.session_state.setdefault(
        "environment_values",
        _base_environment_values(base_model, active_scenario_name),
    )
    st.session_state.setdefault("active_plan_name", _default_plan_name(base_model))
    st.session_state.setdefault(
        "custom_plan_programs",
        _initial_custom_programs(base_model, st.session_state.plan_rows),
    )
    st.session_state.setdefault(
        "custom_plan_measures",
        list(st.session_state.custom_plan_programs),
    )

    try:
        model = build_model(
            base_model,
            st.session_state.node_rows,
            st.session_state.edge_rows,
            st.session_state.plan_rows,
            {active_scenario_name: st.session_state.environment_values},
        )
        st.session_state.last_valid_model = model
        graph_error: str | None = None
    except Exception as exc:  # noqa: BLE001
        model = st.session_state.get("last_valid_model", base_model)
        graph_error = str(exc)

    if graph_error:
        st.warning(
            "Граф содержит ошибку — расчёты идут по последней валидной "
            "версии. Исправьте её на вкладке «Граф»."
        )
    else:
        st.caption(
            f"Граф: {len(model.nodes)} вершин, {len(model.edges)} дуг. "
            "Редактирование — на вкладке «Граф»."
        )

    st.caption(
        "Активный комплекс мер и отклонения факторов среды настраиваются "
        "на вкладке «Граф»."
    )

    st.divider()
    st.subheader("Параметры расчёта")
    horizon = st.slider(
        "Период моделирования, мес.",
        min_value=12,
        max_value=60,
        value=int(model.params.horizon),
        step=1,
    )
    theta = st.slider(
        "Порог θ",
        min_value=0.0,
        max_value=0.8,
        value=float(model.params.theta),
        step=0.005,
        help="Порог срабатывания: слабый комплекс не даёт эффекта. "
        "При θ=0.30 мера m1 одна достигает цели только к 36-му месяцу — "
        "это нижняя граница условия (C)(i).",
    )
    budget_max = st.slider(
        "Бюджет, млн руб.",
        min_value=0.0,
        max_value=150.0,
        value=float(model.params.budget_max),
        step=1.0,
        help="Жёсткое ограничение на суммарную стоимость мер.",
    )

    mu = st.slider(
        "Деградация μ, 1/мес.",
        min_value=0.0,
        max_value=0.05,
        value=float(model.params.mu),
        step=0.001,
        format="%.3f",
    )
    x_max = st.slider(
        "Потолок x_max",
        min_value=1.0,
        max_value=2.0,
        value=float(model.params.x_max),
        step=0.05,
        help="1.6 — верхняя граница эффекта P1 из статьи (40 % при цели 25 %).",
    )
    gain = st.slider(
        "Множитель весов",
        min_value=0.5,
        max_value=1.5,
        value=float(model.params.gain),
        step=0.05,
        help="Глобальный gain для всех дуг: удобно для сканирования чувствительности.",
    )
    saturation = st.radio(
        "Режим насыщения",
        ["ceiling", "strict"],
        format_func=lambda s: "ceiling: x_max − x (рабочий)"
        if s == "ceiling"
        else "strict: 1 − x (исторический)",
        horizontal=False,
    )
    include_optional = st.checkbox(
        "Включить обратную связь X1 → X4",
        value=False,
        help="Опциональная дуга 16: успех финансирует НИОКР.",
    )
    revenue_mode = st.radio(
        "Эффект R",
        ["percent", "rubles"],
        format_func=lambda m: "проценты прироста X1"
        if m == "percent"
        else "рубли через X7 (частичная оценка)",
        horizontal=False,
    )

    with st.expander("Как читать модель"):
        st.markdown(
            r"""
Нормировка: $x=0$ — база, $x=1$ — цель, $x>1$ — перевыполнение.

$$
\sigma_i(t)=\sum_{p} w_{pi}\,a_p(t-\tau_{pi}),\qquad
x_i(t+1)=x_i(t)+\lambda_i(x_{\max}-x_i(t))[\sigma_i-\theta]_+-\mu_i x_i(t)
$$

$\lambda_i=\ln 10/\tau_i^{rise}$. Меры — ступеньки $u_j(t)$,
среда — отклонение от номинала.

При режиме «проценты» $R$ — прирост $X_1$; режим «рубли» монетизирует
только $X_7$, поэтому для сравнения планов используется процентный режим.
"""
        )

# Активный комплекс мер и базовая среда.
_editable_plans = _editable_plan_names(model)
if st.session_state.get("active_plan_name") not in _editable_plans + [CUSTOM_PLAN]:
    st.session_state.active_plan_name = _default_plan_name(model)
_sync_custom_state(model)
plan = _build_active_plan(model)
scenario = model.scenario(active_scenario_name)

options = SimOptions(
    horizon=horizon,
    theta=theta,
    mu=mu,
    x_max=x_max,
    gain=gain,
    budget_max=budget_max,
    saturation=saturation,
    include_optional=include_optional,
    revenue_mode=revenue_mode,
)

result = simulate(model, plan, scenario, options)
summary = summarize(result)

# ---------------------------------------------------------------------------
# Заголовок и KPI
# ---------------------------------------------------------------------------

st.title("Динамика показателей креативности")
st.caption(
    "Графовая модель «меры → факторы креативности → среда». "
    f"Конфиг {model.params.version}; θ={theta:g}; "
    f"бюджет {budget_max:g} млн руб. "
    "Комплекс мер и вершины меняются на вкладке «Граф», "
    "параметры расчёта — слева."
)

target = model.node(summary["target_node"])
growth = summary["growth_pct_T"]
t_goal = summary["t_goal"]
k_value = summary["k_T"]

c1, c2, c3 = st.columns(3)
c1.metric(
    f"{target.id}(T), {target.unit}",
    f"{summary['target_phys_T']:.1f}",
    delta=f"{growth:+.1f} % к базе" if growth is not None else None,
)
c2.metric(
    "Цель",
    "да" if summary["reached"] else "нет",
    delta=f"{t_goal:g} мес."
    if t_goal is not None
    else "за пределами периода моделирования",
)
admissible = bool(summary["reached"] and summary["feasible"])
c3.metric(
    "k = R / C",
    fmt(k_value if admissible else None, 2),
    help=(
        "Эффект на единицу затрат."
        if admissible
        else "k не является критерием: план не достигает цели или превышает бюджет."
    ),
)

c4, c5, c6 = st.columns(3)
c4.metric("Затраты, млн руб.", f"{summary['cost_total']:.1f}")
c5.metric(
    "σ₁ (§5, статика)",
    f"{effect_strength(model, plan.measures):.2f}",
    help="Сумма путей длины ≤ 2 от мер к X1 — величина из первого столбца §5.",
)
c6.metric(
    "Меры в комплексе",
    f"{len(plan.measures)} / {len(model.measure_ids)}"
    if model.measure_ids
    else "—",
    help=", ".join(model.node(m).short or m for m in plan.measures)
    or "комплекс без мер",
)
if not summary["feasible"]:
    st.error(
        "План превышает бюджет: "
        f"{summary['cost_total']:.1f} > {options.budget_max:.1f} млн руб."
    )
for warning in summary.get("warnings", []):
    if "бюджет" not in warning.lower():
        st.warning(warning)

# ---------------------------------------------------------------------------
# Именованные планы
# ---------------------------------------------------------------------------

named_rows = compare_named_plans(model, scenario, options)

# ---------------------------------------------------------------------------
# Вкладки
# ---------------------------------------------------------------------------

tab_dynamics, tab_indicators, tab_compare, tab_graph, tab_data = st.tabs(
    [
        "Динамика",
        "Показатели",
        "Сравнение планов",
        "Граф",
        "Данные",
    ]
)

with tab_dynamics:
    show_fig(plot_dynamics(result, show_scenario=False))
    st.caption(
        "Слева сверху — целевой показатель X1 в процентах к базе. "
        "Справа сверху — все пять показателей в нормированном виде. "
        "Внизу — профили мер и экономика."
    )

with tab_indicators:
    show_fig(plot_indicators(result, show_scenario=False))
    st.caption(
        "Каждый показатель в своих единицах. Пунктир — база, красная линия — цель. "
        "Для X3 и X7 «лучше» означает ниже."
    )

with tab_compare:
    st.subheader("Именованные планы")
    st.caption(
        "Собственные старты планов сохранены: гибрид запускает m3 с "
        "3-го месяца. Для именованного активного комплекса его строка "
        "совпадает с KPI; «Свой комплекс мер» считается в KPI отдельно."
    )
    named_table = pd.DataFrame(
        [
            {
                "План": row["plan_label"],
                "Меры": ", ".join(row["measures"]) or "—",
                "C, млн": num(row["cost_total"], 1),
                "T_goal, мес.": num(row["t_goal"], 0),
                "x1(T)": num(row["x_target_T"], 2),
                "Рост, %": num(row["growth_pct_T"], 1),
                "k = R/C": num(
                    row["k_T"] if row["reached"] and row["feasible"] else None, 2
                ),
                "Цель": "да" if row["reached"] else "нет",
                "Бюджет": "ок" if row["feasible"] else "превышен",
            }
            for row in named_rows
        ]
    )
    st.dataframe(named_table, width="stretch", hide_index=True)
    st.caption(
        "k показан только для планов, достигающих цели в пределах бюджета; "
        "для остальных стоит «—», потому что они не являются допустимым решением."
    )

    st.divider()
    st.subheader("Все подмножества мер")
    st.caption(
        "Здесь все меры стартуют с 0-го месяца — это сетка для поиска "
        "минимальных наборов, а не сравнение с именованными планами."
    )
    rows = compare_plans(model, scenario, options, include_empty=True)

    def _subset_sort_key(row):
        admissible = row["reached"] and row["feasible"]
        return (
            not admissible,
            not row["reached"],
            -(row["k_T"] or 0.0) if admissible else 0.0,
            -row["x_target_T"],
            row["cost_total"],
        )

    rows.sort(key=_subset_sort_key)
    table = pd.DataFrame(
        [
            {
                "План": row["plan"],
                "Меры": ", ".join(row["measures"]) or "—",
                "C, млн": num(row["cost_total"], 1),
                "T_goal, мес.": num(row["t_goal"], 0),
                "x1(T)": num(row["x_target_T"], 2),
                "Рост, %": num(row["growth_pct_T"], 1),
                "k = R/C": num(
                    row["k_T"] if row["reached"] and row["feasible"] else None, 3
                ),
                "Цель": "да" if row["reached"] else "нет",
                "Бюджет": "ок" if row["feasible"] else "превышен",
            }
            for row in rows
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "k показан только для допустимых наборов (цель достигнута, бюджет "
        "соблюдён); остальные строки отсортированы по x1(T), а не по k."
    )

    mss = minimal_sufficient_sets(rows)
    critical = critical_removal_sets(rows, model.measure_ids)
    front = pareto_front(rows)
    a, b, c = st.columns(3)
    a.markdown("**Минимальные достаточные наборы**")
    a.write(
        ", ".join("{ }" if not s else "{" + ", ".join(s) + "}" for s in mss) or "—"
    )
    b.markdown("**Критические меры**")
    b.write(", ".join("{" + ", ".join(s) + "}" for s in critical) or "—")
    b.caption("Удаление этих мер ломает полный комплект.")
    c.markdown("**Парето-фронт**")
    c.write(", ".join(r["plan"] for r in front) or "—")

    show_fig(plot_compare(rows))

with tab_graph:
    flash = st.session_state.pop("graph_flash", None)
    if flash:
        if graph_error:
            st.toast(flash, icon="⚠️")
            st.warning(flash)
        else:
            st.toast(flash, icon="✅")
            st.success(flash)

    st.subheader("Активный комплекс мер")
    plan_options = _editable_plan_names(model) + [CUSTOM_PLAN]
    current_plan_name = st.session_state.get("active_plan_name")
    plan_edit_name = st.selectbox(
        "Активный комплекс мер",
        plan_options,
        index=(
            plan_options.index(current_plan_name)
            if current_plan_name in plan_options
            else 0
        ),
        format_func=lambda name: (
            "Свой комплекс мер"
            if name == CUSTOM_PLAN
            else model.plans[name].label
        ),
        key="active_plan_name",
        help="Этот комплекс используется в KPI и на всех вкладках. "
        "Состав именованных планов фиксирован; для выбора мер выберите "
        "«Свой комплекс мер».",
    )
    st.caption(
        "Именованные планы сохраняют свой состав: в них можно менять "
        "только параметры уже включённых мер. «Свой комплекс мер» позволяет "
        "включать и исключать меры; его состав настраивается в пространстве "
        "«Меры M»."
    )

    st.divider()
    st.subheader("Граф: просмотр и редактирование")
    if graph_error:
        st.error("Модель не применена к расчётам:\n\n" + graph_error)
        st.caption(
            "Показана и используется последняя валидная версия. "
            "Исправьте параметры в карточках ниже."
        )

    show_fig(
        plot_graph(
            model,
            active_measures=plan.measures,
            include_optional=include_optional,
        )
    )
    st.caption(
        "Толщина дуги — |w|, синий — положительное влияние, красный — отрицательное. "
        "Зелёная обводка — меры, включённые в активный комплекс."
    )

    st.divider()
    c_reset, c_download = st.columns(2)
    if c_reset.button(
        "Сбросить",
        help="Вернуть вершины, дуги и именованные планы исходного YAML. "
        "Свой комплекс мер остаётся в текущей сессии.",
    ):
        st.session_state.node_rows = node_rows(base_model)
        st.session_state.edge_rows = edge_rows(base_model)
        st.session_state.plan_rows = plan_rows(base_model)
        st.session_state.environment_values = _base_environment_values(
            base_model, active_scenario_name
        )
        st.session_state.graph_epoch += 1
        st.session_state.last_valid_model = base_model
        st.session_state.graph_flash = "Модель сброшена к исходному конфигу."
        st.rerun()

    c_download.download_button(
        "Скачать YAML",
        data=model_to_yaml(model).encode("utf-8"),
        file_name=f"{config_path.stem}_edited.yaml",
        mime="application/yaml",
        help="Сохраняется последняя валидная модель вместе с планами.",
    )
    st.caption(
        f"Сейчас в модели: **{len(model.nodes)}** вершин, **{len(model.edges)}** дуг. "
        "Параметры в карточках применяются сразу после нажатия «Сохранить» "
        "в диалоговом окне."
    )

    epoch = st.session_state.graph_epoch
    node_table = st.session_state.node_rows
    edge_table = st.session_state.edge_rows
    node_ids = [
        str(row.get("id", "")).strip()
        for row in node_table
        if str(row.get("id", "")).strip()
    ]
    indicator_ids = [
        str(row.get("id", "")).strip()
        for row in node_table
        if str(row.get("kind", "")) == INDICATOR and str(row.get("id", "")).strip()
    ]
    node_labels = {
        str(row.get("id", "")).strip(): (
            str(row.get("short") or row.get("label") or row.get("id", ""))
        )
        for row in node_table
        if str(row.get("id", "")).strip()
    }

    st.subheader("Вершины")
    st.caption(
        "Редактирование разнесено по трём пространствам. ID и тип вершины "
        "фиксированы; название и параметры меняются в карточке вершины."
    )
    space = st.radio(
        "Пространство вершин",
        ["Показатели X", "Меры M", "Среда Y"],
        horizontal=True,
        key="vertex_space",
        label_visibility="collapsed",
    )

    if space == "Показатели X":
        indicator_rows = [
            row
            for row in node_table
            if str(row.get("kind", "")) == INDICATOR
        ]
        if not indicator_rows:
            st.caption("В модели нет показателей.")
        for row in indicator_rows:
            node_id = str(row.get("id", "")).strip()
            if not node_id:
                continue
            short = str(row.get("short") or row.get("label") or node_id)
            with st.container(border=True):
                c_text, c_action = st.columns([4, 1])
                with c_text:
                    st.markdown(f"**{node_id} · {short}**")
                    st.write(str(row.get("label") or ""))
                    st.caption(
                        f"Направление: "
                        f"{'рост (up)' if row.get('direction') == 'up' else 'снижение (down)'} · "
                        f"база {fmt(row.get('base'))} · цель {fmt(row.get('target'))} · "
                        f"rise {fmt(row.get('rise'), 0)} мес."
                    )
                with c_action:
                    if st.button(
                        "Изменить",
                        key=f"edit_indicator_{node_id}_{epoch}",
                    ):
                        edit_indicator_dialog(node_id, epoch)

    elif space == "Меры M":
        is_custom = plan_edit_name == CUSTOM_PLAN
        if is_custom:
            plan_label = "Свой комплекс мер"
            plan_data = st.session_state.custom_plan_programs
            st.multiselect(
                "Меры в комплексе",
                model.measure_ids,
                key="custom_plan_measures",
                format_func=lambda m: f"{m} · {model.node(m).short or m}",
                help="Отметьте меры, которые входят в свой комплекс. "
                "Их расписание задаётся в карточках ниже.",
            )
            st.caption(
                "Состав своего комплекса задаётся здесь; параметры включённых "
                "мер меняются в карточках ниже."
            )
        else:
            plan_label = model.plans[plan_edit_name].label
            plan_data = st.session_state.plan_rows.get(plan_edit_name, {})
            st.caption(
                "Состав именованного плана фиксирован: добавлять или убирать "
                "меры нельзя, но параметры включённых мер можно менять."
            )

        measure_rows = [
            row
            for row in node_table
            if str(row.get("kind", "")) == MEASURE
        ]
        if not measure_rows:
            st.caption("В модели нет мер.")
        for row in measure_rows:
            node_id = str(row.get("id", "")).strip()
            if not node_id:
                continue
            short = str(row.get("short") or row.get("label") or node_id)
            program = plan_data.get(node_id)
            with st.container(border=True):
                c_text, c_action = st.columns([4, 1])
                with c_text:
                    st.markdown(f"**{node_id} · {short}**")
                    st.write(str(row.get("label") or ""))
                    st.caption(
                        f"Стоимость: {fmt(row.get('cost'), 1)} млн · "
                        f"ответственный: {row.get('owner') or '—'}"
                    )
                    if isinstance(program, dict):
                        duration = program.get("duration")
                        duration_text = (
                            "до конца периода"
                            if duration is None
                            else f"{fmt(duration, 0)} мес."
                        )
                        st.caption(
                            f"В комплексе «{plan_label}»: старт "
                            f"{fmt(program.get('start'), 0)} мес. · "
                            f"интенсивность {fmt(program.get('intensity'))} · "
                            f"{duration_text}"
                        )
                    elif is_custom:
                        st.caption(
                            "В свой комплекс не входит. Отметьте меру в списке "
                            "выше, чтобы задать расписание."
                        )
                    else:
                        st.caption(
                            f"В комплекс «{plan_label}» не входит. "
                            "Состав именованного плана фиксирован."
                        )
                with c_action:
                    if st.button(
                        "Изменить",
                        key=f"edit_measure_{plan_edit_name}_{node_id}_{epoch}",
                    ):
                        edit_measure_dialog(
                            node_id,
                            plan_edit_name,
                            plan_label,
                            epoch,
                            int(horizon),
                            custom=is_custom,
                        )

    else:
        environment_values = st.session_state.environment_values
        st.caption(
            "Отклонение от номинала используется в базовом расчёте и "
            "задаётся в карточке фактора."
        )
        environment_rows = [
            row
            for row in node_table
            if str(row.get("kind", "")) == ENVIRONMENT
        ]
        if not environment_rows:
            st.caption("В модели нет факторов среды.")
        for row in environment_rows:
            node_id = str(row.get("id", "")).strip()
            if not node_id:
                continue
            short = str(row.get("short") or row.get("label") or node_id)
            deviation = float(environment_values.get(node_id, 0.0))
            with st.container(border=True):
                c_text, c_action = st.columns([4, 1])
                with c_text:
                    st.markdown(f"**{node_id} · {short}**")
                    st.write(str(row.get("label") or ""))
                    st.caption(
                        f"Отклонение от номинала: {deviation:+.2f} · "
                        f"единица: {row.get('unit') or '—'}"
                    )
                with c_action:
                    if st.button(
                        "Изменить",
                        key=f"edit_environment_{node_id}_{epoch}",
                    ):
                        edit_environment_dialog(node_id, epoch)

    st.divider()
    st.subheader("Дуги")
    st.caption(
        "«В» — только показатели X: по модели дуга должна входить в показатель. "
        "Новую строку можно добавить внизу таблицы."
    )
    edge_frame = pd.DataFrame(edge_table, columns=EDGE_FIELDS)
    with st.form(f"graph_editor_edges_{epoch}"):
        edited_edges = st.data_editor(
            edge_frame,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            height=320,
            column_config={
                "from": st.column_config.SelectboxColumn(
                    "Из", options=node_ids, required=True, width="small"
                ),
                "to": st.column_config.SelectboxColumn(
                    "В", options=indicator_ids, required=True, width="small"
                ),
                "w": st.column_config.NumberColumn(
                    "Вес w", format="%.2f", step=0.05
                ),
                "delay": st.column_config.NumberColumn(
                    "Задержка τ, мес.", format="%.0f", min_value=0, step=1
                ),
                "optional": st.column_config.CheckboxColumn("Опц."),
                "source": st.column_config.TextColumn("Класс", width="small"),
                "mechanism": st.column_config.TextColumn(
                    "Механизм", width="large"
                ),
            },
            key=f"edge_editor_{epoch}",
        )
        submitted = st.form_submit_button(
            "Применить изменения дуг", type="primary"
        )

    if submitted:
        st.session_state.edge_rows = edited_edges.to_dict("records")
        try:
            build_model(
                base_model,
                st.session_state.node_rows,
                st.session_state.edge_rows,
                st.session_state.plan_rows,
                {active_scenario_name: st.session_state.environment_values},
            )
        except Exception:  # noqa: BLE001
            st.session_state.graph_flash = (
                "Правки внесены, но модель пока невалидна — "
                "смотрите сообщение об ошибке ниже."
            )
        else:
            st.session_state.graph_flash = "Дуги применены."
        # Новый epoch сбрасывает состояние data_editor: иначе после
        # добавления/удаления строки Streamlit может применить старые
        # added_rows/deleted_rows к уже обновлённой таблице.
        st.session_state.graph_epoch += 1
        st.rerun()

    st.divider()
    with st.expander("Добавить вершину", expanded=False):
        new_kind = st.selectbox(
            "Тип новой вершины",
            [INDICATOR, MEASURE, ENVIRONMENT],
            format_func=lambda kind: KIND_LABELS[kind],
            key=f"new_kind_{epoch}",
        )
        if new_kind == ENVIRONMENT:
            st.caption(
                "Новый фактор среды по умолчанию равен 0; отклонение "
                "задаётся в карточке фактора."
            )
        elif new_kind == MEASURE:
            st.caption(
                "Новая мера появится в пространстве «Меры M». Чтобы включить "
                "её в свой комплекс, отметьте её в списке мер и задайте "
                "расписание в карточке."
            )
        else:
            st.caption(
                "Новый показатель появится на графе; добавьте входящие дуги, "
                "чтобы он менялся."
            )
        suggested_id = suggest_node_id(model, new_kind)
        with st.form(f"add_node_{epoch}_{new_kind}"):
            new_id = st.text_input("ID", value=suggested_id)
            new_label = st.text_input(
                "Название", placeholder="Например, «Новый показатель»"
            )
            new_short = st.text_input(
                "Короткое имя",
                placeholder="Если пусто — возьмётся название",
            )
            new_unit = st.text_input(
                "Единица измерения", placeholder="например, % или мес."
            )

            if new_kind == INDICATOR:
                c1, c2, c3 = st.columns(3)
                new_base = c1.number_input(
                    "База", value=100.0, format="%.2f"
                )
                new_target = c2.number_input(
                    "Цель", value=125.0, format="%.2f"
                )
                new_rise = c3.number_input(
                    "rise, мес.", value=18.0, min_value=1.0, step=1.0
                )
                new_direction = st.selectbox(
                    "Направление",
                    ["up", "down"],
                    format_func=lambda d: "up — больше лучше"
                    if d == "up"
                    else "down — меньше лучше",
                )
                new_cost = 0.0
            elif new_kind == MEASURE:
                new_cost = st.number_input(
                    "Стоимость, млн руб.",
                    value=10.0,
                    min_value=0.0,
                    step=1.0,
                )
                new_base = new_target = new_rise = 0.0
                new_direction = "up"
            else:
                new_base = new_target = new_rise = new_cost = 0.0
                new_direction = "up"

            add_node = st.form_submit_button("Добавить вершину")

        if add_node:
            try:
                node_id = validate_new_node_id(
                    new_id,
                    [
                        str(row.get("id", "")).strip()
                        for row in st.session_state.node_rows
                    ],
                )
                row = new_node_row(
                    model,
                    node_id,
                    new_kind,
                    label=new_label,
                    short=new_short,
                    unit=new_unit,
                    direction=new_direction,
                    base=new_base,
                    target=new_target,
                    rise=new_rise,
                    cost=new_cost,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.node_rows.append(row)
                if new_kind == ENVIRONMENT:
                    st.session_state.environment_values.setdefault(node_id, 0.0)
                _finish_graph_edit(
                    f"Добавлена вершина {node_id}. Не забудьте добавить дуги."
                )

    with st.expander("Удалить вершину", expanded=False):
        deletable = [
            node_id for node_id in node_ids if node_id != model.params.target_node
        ]
        if not deletable:
            st.caption("Нет вершин, доступных для удаления.")
        else:
            victim = st.selectbox(
                "Вершина",
                deletable,
                format_func=lambda node_id: (
                    f"{node_id} — {node_labels.get(node_id, '')}"
                ),
                key=f"delete_node_{epoch}",
            )
            st.caption(
                "Связанные дуги будут удалены. Мера исчезнет из именованных "
                "планов и своего комплекса, фактор среды — из модели. "
                "Целевой показатель "
                f"{model.params.target_node} удалить нельзя."
            )
            if st.button("Удалить вершину", key=f"delete_button_{epoch}"):
                st.session_state.node_rows = [
                    row
                    for row in st.session_state.node_rows
                    if str(row.get("id", "")).strip() != victim
                ]
                st.session_state.edge_rows = [
                    row
                    for row in st.session_state.edge_rows
                    if str(row.get("from", "")).strip() != victim
                    and str(row.get("to", "")).strip() != victim
                ]
                for programs in st.session_state.plan_rows.values():
                    programs.pop(victim, None)
                st.session_state.environment_values.pop(victim, None)
                _finish_graph_edit(f"Вершина {victim} удалена.")

    with st.expander("Сохранить YAML в проект"):
        st.caption(
            "Кнопка «Скачать YAML» выше сохраняет последнюю валидную модель "
            "на компьютер и работает всегда. Запись в configs/ — для "
            "локального запуска."
        )
        yaml_name = st.text_input(
            "Имя файла",
            value=f"{config_path.stem}_edited.yaml",
            key=f"yaml_name_{epoch}",
        )
        overwrite = st.checkbox(
            "Перезаписать существующий файл",
            value=False,
            key=f"yaml_overwrite_{epoch}",
        )
        if st.button("Сохранить в configs/", key=f"yaml_save_{epoch}"):
            safe_name = Path(yaml_name).name
            if not safe_name.endswith((".yaml", ".yml")):
                safe_name += ".yaml"
            if safe_name in {"", ".yaml", ".yml"}:
                st.error("Введите имя файла.")
            elif safe_name.lower() in {"karkas.yaml", "karkas.yml"}:
                st.error(
                    "Базовый конфиг karkas.yaml защищён от перезаписи. "
                    "Выберите другое имя."
                )
            else:
                target_path = CONFIG_DIR / safe_name
                if target_path.exists() and not overwrite:
                    st.error(
                        f"Файл {safe_name} уже существует. Поставьте галочку "
                        "перезаписи или выберите другое имя."
                    )
                else:
                    try:
                        save_model_yaml(model, target_path)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Не удалось сохранить: {exc}")
                    else:
                        _finish_graph_edit(f"Сохранено: {target_path.name}.")

with tab_data:
    st.subheader("Ряды динамики")
    data = {
        "месяц": result.time,
    }
    for node in model.indicators:
        data[f"{node.id} норм."] = result.states[node.id]
        data[f"{node.id} физ."] = result.physical[node.id]
    for measure in model.measure_ids:
        data[f"u {measure}"] = result.measures[measure]
    data["R"] = result.revenue
    data["C накопл."] = result.cost_cum
    data["k"] = [
        (r / c) if c > 0 else None for r, c in zip(result.revenue, result.cost_cum)
    ]
    frame = pd.DataFrame(data)
    st.dataframe(frame, width="stretch", hide_index=True)

    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Скачать dynamics.csv",
        data=csv_bytes,
        file_name=f"{plan.name}_dynamics.csv",
        mime="text/csv",
    )
    export_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"scenario", "scenario_label"}
    }
    st.download_button(
        "Скачать summary.json",
        data=json.dumps(
            export_summary, ensure_ascii=False, indent=2, default=str
        ).encode("utf-8"),
        file_name=f"{plan.name}_summary.json",
        mime="application/json",
    )
