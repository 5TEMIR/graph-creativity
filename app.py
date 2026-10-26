"""Streamlit-приложение: песочница для графовой модели креативности.

Два режима: ЛПР (анкета, бюджет, рекомендация) и исследователь
(все подмножества, σ₁, редактирование графа).

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
    NODE_FIELDS,
    build_model,
    edge_rows,
    model_to_yaml,
    new_node_row,
    node_rows,
    save_model_yaml,
    suggest_node_id,
    validate_new_node_id,
)
from graphcreativity.engine import SimOptions, simulate
from graphcreativity.metrics import (
    compare_named_plans,
    compare_plans,
    critical_event_combinations,
    critical_removal_sets,
    effect_strength,
    minimal_sufficient_sets,
    pareto_front,
    scenario_robustness,
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
    st.title("🕸️ Песочница")
    st.caption("Графовая модель повышения креативности предприятия")
    mode = st.radio(
        "Режим",
        ["ЛПР", "Исследователь"],
        index=0,
        horizontal=True,
        help="ЛПР: анкета, бюджет и рекомендация. Исследователь: все подмножества, "
        "σ₁, редактирование графа и отладочные параметры.",
    )
    config_path = st.selectbox(
        "Конфиг",
        configs,
        index=default_config_index,
        format_func=lambda p: p.name,
        help="Модель, дуги, параметры, планы и сценарии — всё в YAML.",
    )
    try:
        base_model = load_model(config_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Ошибка загрузки модели:\n\n{exc}")
        st.stop()

    # Состояние редактора графа. При смене конфига сбрасываем его.
    if st.session_state.get("graph_config") != str(config_path):
        st.session_state.graph_config = str(config_path)
        st.session_state.graph_epoch = st.session_state.get("graph_epoch", 0) + 1
        st.session_state.node_rows = node_rows(base_model)
        st.session_state.edge_rows = edge_rows(base_model)
        st.session_state.last_valid_model = base_model

    try:
        model = build_model(
            base_model,
            st.session_state.node_rows,
            st.session_state.edge_rows,
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

    if mode == "ЛПР":
        with st.expander("Анкета предприятия", expanded=False):
            st.caption(
                "Замените демонстрационные числа на данные предприятия. "
                "База и цель — в физических единицах; для X3 и X7 «лучше» "
                "означает ниже, поэтому цель меньше базы. Цель X1 — цель "
                "предприятия; цели X2/X4/X7 — вспомогательные ориентиры "
                "нормировки, их можно оставить демонстрационными и проверить "
                "в чувствительности."
            )
            indicator_ids = ["X1", "X2", "X3", "X4", "X7"]
            sheet = pd.DataFrame(
                [
                    {
                        "Код": node_id,
                        "Показатель": model.node(node_id).label,
                        "База": float(model.node(node_id).base),
                        "Цель": float(model.node(node_id).target),
                        "Ед.": model.node(node_id).unit,
                    }
                    for node_id in indicator_ids
                ]
            )
            with st.form("enterprise_form"):
                edited = st.data_editor(
                    sheet,
                    hide_index=True,
                    disabled=["Код", "Показатель", "Ед."],
                    column_config={
                        "База": st.column_config.NumberColumn(format="%.2f"),
                        "Цель": st.column_config.NumberColumn(format="%.2f"),
                    },
                    key="enterprise_sheet",
                )
                st.markdown("**Сметы мер, млн руб.**")
                cost_cols = st.columns(3)
                m1_cost = cost_cols[0].number_input(
                    "m1 · идеи",
                    min_value=0.0,
                    value=float(model.node("m1").cost),
                    step=1.0,
                )
                m3_cost = cost_cols[1].number_input(
                    "m3 · акселератор",
                    min_value=0.0,
                    value=float(model.node("m3").cost),
                    step=1.0,
                )
                m4_cost = cost_cols[2].number_input(
                    "m4 · патенты",
                    min_value=0.0,
                    value=float(model.node("m4").cost),
                    step=1.0,
                )
                submitted = st.form_submit_button("Применить данные")

            if submitted:
                updates: dict[str, dict[str, float]] = {}
                for _, row in edited.iterrows():
                    updates[str(row["Код"])] = {
                        "base": float(row["База"]),
                        "target": float(row["Цель"]),
                    }
                updates["m1"] = {"cost": float(m1_cost)}
                updates["m3"] = {"cost": float(m3_cost)}
                updates["m4"] = {"cost": float(m4_cost)}
                rows = [dict(row) for row in st.session_state.node_rows]
                for row in rows:
                    if row["id"] in updates:
                        row.update(updates[row["id"]])
                st.session_state.node_rows = rows
                st.session_state.enterprise_flash = "Данные предприятия применены."
                st.rerun()

    st.divider()
    st.subheader("Сценарий")
    scenario_name = st.selectbox(
        "Внешняя среда",
        list(model.scenarios),
        format_func=lambda n: model.scenarios[n].label,
    )
    scenario = model.scenario(scenario_name)

    st.divider()
    st.subheader("План мероприятий")
    horizon = st.slider(
        "Горизонт, мес.",
        min_value=12,
        max_value=60,
        value=int(model.params.horizon),
        step=1,
    )
    plan_options = ["__custom__"] + list(model.plans)
    default_plan = "hybrid" if "hybrid" in model.plans else plan_options[0]
    plan_key = st.selectbox(
        "Комплекс мер",
        plan_options,
        index=plan_options.index(default_plan),
        format_func=lambda k: "Свой набор…"
        if k == "__custom__"
        else model.plans[k].label,
    )

    if plan_key == "__custom__":
        selected = st.multiselect(
            "Меры",
            model.measure_ids,
            default=model.measure_ids[:2],
            format_func=lambda m: model.node(m).short or m,
        )
        # После удаления меры из графа виджет может ещё хранить старый ID.
        selected = [measure for measure in selected if measure in model.measure_ids]
        starts: dict[str, float] = {}
        for measure in selected:
            starts[measure] = float(
                st.slider(
                    f"Старт {measure} — {model.node(measure).short or measure}, мес.",
                    min_value=0,
                    max_value=horizon,
                    value=0,
                    step=1,
                )
            )
        plan = Plan(
            name="custom",
            label="Свой набор",
            programs={
                measure: MeasureProgram(start=starts[measure]) for measure in selected
            },
        )
    else:
        plan = model.plan(plan_key)
        st.caption(
            "Меры: "
            + (", ".join(model.node(m).short or m for m in plan.measures) or "нет")
        )

    st.divider()
    st.subheader("Параметры расчёта")
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

    if mode == "Исследователь":
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
    else:
        mu = float(model.params.mu)
        x_max = float(model.params.x_max)
        gain = float(model.params.gain)
        saturation = str(model.params.saturation)
        include_optional = False
        revenue_mode = str(model.params.revenue_mode)

    with st.expander("Как читать модель"):
        st.markdown(
            r"""
Нормировка: $x=0$ — база, $x=1$ — цель, $x>1$ — перевыполнение.

$$
\sigma_i(t)=\sum_{p} w_{pi}\,a_p(t-\tau_{pi}),\qquad
x_i(t+1)=x_i(t)+\lambda_i(x_{\max}-x_i(t))[\sigma_i-\theta]_+-\mu_i x_i(t)
$$

$\lambda_i=\ln 10/\tau_i^{rise}$. Меры — ступеньки $u_j(t)$,
среда — отклонение сценария от номинала.

$R$ в режиме ЛПР считается в процентах прироста $X_1$; рублёвый режим
монетизирует только $X_7$ и поэтому не используется для сравнения планов.
"""
        )

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

enterprise_flash = st.session_state.pop("enterprise_flash", None)
if enterprise_flash:
    st.success(enterprise_flash)

# ---------------------------------------------------------------------------
# Заголовок и KPI
# ---------------------------------------------------------------------------

st.title("Динамика показателей креативности")
st.caption(
    "Графовая модель «меры → факторы креативности → среда». "
    f"Конфиг {model.params.version}; θ={theta:g}; "
    f"бюджет {budget_max:g} млн руб. "
    "Меняйте план, сценарий и параметры слева — графики пересчитываются."
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
    delta=f"{t_goal:g} мес." if t_goal is not None else "за горизонтом",
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
if mode == "ЛПР":
    c5.metric(
        "Бюджет, млн руб.",
        f"{options.budget_max:.0f}",
        delta="ок" if summary["feasible"] else "превышен",
        delta_color="normal" if summary["feasible"] else "inverse",
    )
else:
    c5.metric(
        "σ₁ (§5, статика)",
        f"{effect_strength(model, plan.measures):.2f}",
        help="Сумма путей длины ≤ 2 от мер к X1 — величина из первого столбца §5.",
    )
c6.metric(
    "Меры в плане",
    f"{len(plan.measures)} / {len(model.measure_ids)}"
    if model.measure_ids
    else "—",
    help=", ".join(model.node(m).short or m for m in plan.measures) or "план без мер",
)
if mode == "ЛПР":
    st.caption(
        "KPI относятся к выбранному плану; блок «Рекомендация» ниже "
        "сравнивает все именованные планы с их собственными стартами."
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
# Режим ЛПР: рекомендация, а не набор отладочных метрик
# ---------------------------------------------------------------------------

named_rows = compare_named_plans(model, scenario, options)

if mode == "ЛПР":
    st.divider()
    st.subheader("Рекомендация")
    st.caption(
        f"Сценарий: {scenario.label}. Сравниваются именованные планы с их "
        "собственными стартами; устойчивость проверяется по всем сценариям."
    )
    candidates = [row for row in named_rows if row["reached"] and row["feasible"]]
    priority = st.radio(
        "Приоритет выбора",
        ["эффект/затраты", "время", "стоимость"],
        horizontal=True,
        help="По умолчанию максимизируется k = R/C; время и стоимость — "
        "альтернативные правила для ЛПР.",
    )
    if not candidates:
        st.warning(
            "Ни один именованный план не достигает цели в рамках бюджета. "
            "Увеличьте бюджет, горизонт или переключитесь в режим «Исследователь»."
        )
    else:
        if priority == "эффект/затраты":
            best = max(
                candidates,
                key=lambda row: (
                    row["k_T"] or 0.0,
                    -row["t_goal"],
                    -row["cost_total"],
                ),
            )
        elif priority == "время":
            best = min(
                candidates,
                key=lambda row: (row["t_goal"], row["cost_total"]),
            )
        else:
            best = min(
                candidates,
                key=lambda row: (row["cost_total"], row["t_goal"]),
            )
        robustness = scenario_robustness(model, best["plan"], options=options)
        best_plan = model.plan(best["plan"])
        owners = [
            f"{model.node(measure).short or measure} — "
            f"{model.node(measure).owner or 'ответственный не задан'}"
            for measure in best_plan.measures
        ]
        starts = [
            f"{model.node(measure).short or measure}: старт {program.start:g} мес."
            for measure, program in best_plan.programs.items()
        ]

        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"### {best['plan_label']}")
            k_text = (
                f"$k = R/C$ = **{best['k_T']:.2f}**"
                if best["k_T"] is not None
                else "$k$ не определён (C = 0)"
            )
            st.write(
                f"Цель достигается к **{best['t_goal']:g} мес.**, "
                f"стоимость **{best['cost_total']:.1f} млн руб.**, {k_text}."
            )
            st.write("**Кому:** " + "; ".join(owners))
            st.caption("**Когда:** " + "; ".join(starts))
            st.caption(
                "Правило выбора: цель достигнута, бюджет соблюдён, затем — "
                f"приоритет «{priority}»."
            )
        with right:
            st.metric(
                "Устойчивость",
                f"{robustness['radius']} / {len(model.atomic_scenarios)}",
                help="Сколько одновременных неблагоприятных событий выдерживает план.",
            )
            st.metric(
                "Сценарии с целью",
                f"{robustness['success_share'] * 100:.0f} %",
                help="Доля сценариев, где цель достигнута и бюджет соблюдён.",
            )
            st.metric(
                "Худший запас x1(T) − 1",
                f"{robustness['min_margin']:+.2f}",
            )

        alt = pd.DataFrame(
            [
                {
                    "План": row["plan_label"],
                    "T_goal, мес.": num(row["t_goal"], 0),
                    "C, млн": num(row["cost_total"], 1),
                    "k": num(row["k_T"], 2),
                    "Цель": "да" if row["reached"] else "нет",
                    "Бюджет": "ок" if row["feasible"] else "превышен",
                }
                for row in sorted(
                    named_rows,
                    key=lambda row: (
                        not (row["reached"] and row["feasible"]),
                        row["t_goal"] if row["t_goal"] is not None else 10**9,
                        row["cost_total"],
                    ),
                )
            ]
        )
        st.caption("Именованные планы с их собственными стартами:")
        st.dataframe(alt, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Вкладки
# ---------------------------------------------------------------------------

tab_dynamics, tab_indicators, tab_compare, tab_scenarios, tab_graph, tab_data = st.tabs(
    [
        "Динамика",
        "Показатели" if mode == "Исследователь" else "Показатели (иссл.)",
        "Сравнение планов",
        "Сценарии",
        "Граф" if mode == "Исследователь" else "Граф (иссл.)",
        "Данные",
    ]
)

with tab_dynamics:
    show_fig(plot_dynamics(result))
    st.caption(
        "Слева сверху — целевой показатель X1 в процентах к базе. "
        "Справа сверху — все пять показателей в нормированном виде. "
        "Внизу — профили мер и экономика."
    )

with tab_indicators:
    show_fig(plot_indicators(result))
    st.caption(
        "Каждый показатель в своих единицах. Пунктир — база, красная линия — цель. "
        "Для X3 и X7 «лучше» означает ниже."
    )

with tab_compare:
    st.subheader("Именованные планы")
    st.caption(
        f"Сценарий: {scenario.label}. Собственные старты планов сохранены: "
        "гибрид запускает m3 с 3-го месяца. Строка выбранного плана "
        "совпадает с KPI наверху."
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
        f"Сценарий: {scenario.label}. Здесь все меры стартуют с 0-го месяца — "
        "это сетка для поиска минимальных наборов, а не сравнение с "
        "именованными планами."
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

    st.divider()
    st.subheader("Критические комбинации событий")
    st.caption(
        "Минимальные наборы неблагоприятных событий, при которых план "
        "перестаёт достигать цели в номинале. Это отдельный объект от "
        "«критических мер»."
    )
    event_rows = []
    named_nominal = {
        row["plan"]: row
        for row in compare_named_plans(model, "nominal", options)
    }
    for plan_name in model.plans:
        combos = critical_event_combinations(model, plan_name, options=options)
        baseline = named_nominal.get(plan_name, {})
        baseline_ok = bool(baseline.get("reached") and baseline.get("feasible"))
        if not baseline_ok:
            text = "план недопустим уже без событий (цель/бюджет)"
        elif combos:
            text = ", ".join(
                "{"
                + ", ".join(model.scenario(event).label or event for event in combo)
                + "}"
                for combo in combos
            )
        else:
            text = "нет (план устойчив ко всем комбинациям)"
        event_rows.append(
            {
                "План": model.plan(plan_name).label,
                "Критические комбинации": text,
            }
        )
    st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Матрица план × сценарий")
    st.caption(
        "T_goal в месяцах; «—» — цель не достигнута; «бюджет» — план "
        "превышает бюджет. «Радиус» — сколько одновременных событий "
        "план выдерживает."
    )
    matrix_rows = []
    for plan_name, plan_obj in model.plans.items():
        robustness = scenario_robustness(model, plan_obj, options=options)
        row = {"План": plan_obj.label}
        for scenario_row in robustness["rows"]:
            if not scenario_row["feasible"]:
                value = "бюджет"
            elif scenario_row["reached"]:
                value = f"{scenario_row['t_goal']:g} мес."
            else:
                value = "—"
            row[scenario_row["label"]] = value
        row["Радиус"] = (
            robustness["radius"] if robustness["radius"] >= 0 else "—"
        )
        matrix_rows.append(row)
    st.dataframe(pd.DataFrame(matrix_rows), width="stretch", hide_index=True)

    show_fig(plot_compare(rows))

    st.divider()
    st.subheader("Устойчивость к сценариям")
    if st.checkbox("Посчитать устойчивость всех планов", value=False):
        with st.spinner("Прогоняю сценарии…"):
            robust_rows = []
            for row in rows:
                plan_obj = Plan(
                    name=row["plan"],
                    label=row["plan_label"],
                    programs={
                        m: MeasureProgram(start=0.0) for m in row["measures"]
                    },
                )
                robustness = scenario_robustness(model, plan_obj, options=options)
                robust_rows.append(
                    {
                        "План": row["plan"],
                        "Доля сценариев с целью": robustness["success_share"] * 100,
                        "Радиус, событий": robustness["radius"]
                        if robustness["radius"] >= 0
                        else "—",
                        "Худший запас": robustness["min_margin"],
                    }
                )
        st.dataframe(
            pd.DataFrame(robust_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Доля сценариев с целью": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.0f%%"
                ),
                "Радиус, событий": st.column_config.TextColumn(),
                "Худший запас": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        st.caption("Включите галочку, чтобы прогнать каждый план по всем сценариям.")

with tab_scenarios:
    st.subheader("Веер сценариев для выбранного плана")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for sc in model.scenarios.values():
        res = simulate(model, plan, sc, options)
        ax.plot(
            res.time,
            res.states[res.target_id],
            lw=1.8,
            label=f"{sc.label} (T={res.first_goal_time() or '—'})",
        )
    ax.axhline(1.0, color="#C00000", ls="--", lw=1.1, label="цель x=1")
    ax.set_xlabel("месяц")
    ax.set_ylabel(f"{target.id}, норм.")
    ax.set_title("X1 по сценариям внешней среды")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    show_fig(fig)

    scenario_rows = []
    for sc in model.scenarios.values():
        res = simulate(model, plan, sc, options)
        s = summarize(res)
        scenario_rows.append(
            {
                "Сценарий": sc.label,
                "Достигнута": "да" if s["reached"] else "нет",
                "T_goal": num(s["t_goal"], 0),
                "x1(T)": num(s["x_target_T"], 2),
                "Запас": num(s["margin"], 2),
                "Рост, %": num(s["growth_pct_T"], 1),
                "k": num(s["k_T"], 3),
                "Бюджет": "ок" if s["feasible"] else "превышен",
            }
        )
    st.dataframe(
        pd.DataFrame(scenario_rows),
        width="stretch",
        hide_index=True,
    )

    selected_robustness = scenario_robustness(model, plan, options=options)
    st.caption(
        f"Радиус устойчивости: **{selected_robustness['radius']}** из "
        f"{len(model.atomic_scenarios)} одновременных событий; "
        f"доля сценариев с целью — {selected_robustness['success_share'] * 100:.0f} %."
    )

with tab_graph:
    flash = st.session_state.pop("graph_flash", None)
    if flash:
        if graph_error:
            st.toast(flash, icon="⚠️")
            st.warning(flash)
        else:
            st.toast(flash, icon="✅")
            st.success(flash)

    st.subheader("Граф: просмотр и редактирование")
    if mode == "ЛПР":
        st.info(
            "Это исследовательский раздел. Для решения используйте верхний блок "
            "«Рекомендация»; редактирование графа меняет расчётную модель."
        )
    if graph_error:
        st.error("Граф не применён к расчётам:\n\n" + graph_error)
        st.caption(
            "Показана и используется последняя валидная версия. "
            "Исправьте таблицы ниже и нажмите «Применить изменения»."
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
        "Зелёная обводка — меры, включённые в текущий план."
    )

    st.divider()
    c_reset, c_download = st.columns(2)
    if c_reset.button(
        "Сбросить",
        help="Вернуть вершины и дуги исходного YAML. Параметры и планы не меняются.",
    ):
        st.session_state.node_rows = node_rows(base_model)
        st.session_state.edge_rows = edge_rows(base_model)
        st.session_state.graph_epoch += 1
        st.session_state.last_valid_model = base_model
        st.session_state.graph_flash = "Граф сброшен к исходному конфигу."
        st.rerun()

    c_download.download_button(
        "Скачать YAML",
        data=model_to_yaml(model).encode("utf-8"),
        file_name=f"{config_path.stem}_edited.yaml",
        mime="application/yaml",
        help="Сохраняется последняя валидная модель вместе с планами и сценариями.",
    )
    st.caption(
        f"Сейчас в модели: **{len(model.nodes)}** вершин, **{len(model.edges)}** дуг. "
        "Изменения из таблиц применяются к расчётам только после нажатия "
        "«Применить изменения»."
    )

    st.subheader("Вершины и дуги")
    st.caption(
        "ID и тип существующей вершины фиксированы. Меняйте параметры, дуги "
        "и координаты; новую вершину можно добавить ниже. Для факторов среды Y "
        "поля «база» и «цель» не используются."
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

    node_frame = pd.DataFrame(node_table, columns=NODE_FIELDS)
    edge_frame = pd.DataFrame(edge_table, columns=EDGE_FIELDS)

    with st.form(f"graph_editor_{epoch}"):
        st.markdown("**Вершины**")
        edited_nodes = st.data_editor(
            node_frame,
            num_rows="fixed",
            hide_index=True,
            width="stretch",
            height=330,
            disabled=["id", "kind"],
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "kind": st.column_config.SelectboxColumn(
                    "Тип",
                    options=[INDICATOR, MEASURE, ENVIRONMENT],
                    format_func=lambda kind: KIND_LABELS.get(kind, kind),
                    width="medium",
                ),
                "label": st.column_config.TextColumn("Название", width="large"),
                "short": st.column_config.TextColumn("Коротко", width="medium"),
                "unit": st.column_config.TextColumn("Ед.", width="small"),
                "direction": st.column_config.SelectboxColumn(
                    "Направление",
                    options=["up", "down"],
                    format_func=lambda d: "рост (up)"
                    if d == "up"
                    else "снижение (down)",
                    width="small",
                    help="up: больше — лучше; down: меньше — лучше (X растёт при снижении).",
                ),
                "base": st.column_config.NumberColumn(
                    "База", format="%.2f", help="Физическое базовое значение — для показателей."
                ),
                "target": st.column_config.NumberColumn(
                    "Цель", format="%.2f", help="Физическая цель — для показателей."
                ),
                "rise": st.column_config.NumberColumn(
                    "rise, мес.",
                    format="%.0f",
                    min_value=0,
                    help="Время до плато — только для показателей.",
                ),
                "cost": st.column_config.NumberColumn(
                    "Стоимость, млн",
                    format="%.1f",
                    min_value=0,
                    help="Разовая стоимость меры, млн руб.",
                ),
                "owner": st.column_config.TextColumn(
                    "Ответственный",
                    width="medium",
                    help="Подразделение, которому передаётся мера.",
                ),
                "x": st.column_config.NumberColumn("x", format="%.1f"),
                "y": st.column_config.NumberColumn("y", format="%.1f"),
            },
            key=f"node_editor_{epoch}",
        )

        st.markdown("**Дуги**")
        st.caption(
            "«В» — только показатели X: по модели дуга должна входить в показатель. "
            "Новую строку можно добавить внизу таблицы."
        )
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
            "Применить изменения", type="primary"
        )

    if submitted:
        st.session_state.node_rows = edited_nodes.to_dict("records")
        st.session_state.edge_rows = edited_edges.to_dict("records")
        try:
            build_model(
                base_model,
                st.session_state.node_rows,
                st.session_state.edge_rows,
            )
        except Exception:  # noqa: BLE001
            st.session_state.graph_flash = (
                "Правки внесены в таблицы, но граф пока невалиден — "
                "смотрите сообщение об ошибке ниже."
            )
        else:
            st.session_state.graph_flash = "Изменения графа применены."
        # Новый epoch сбрасывает состояние data_editor: иначе после
        # добавления/удаления строки Streamlit может применить старые
        # added_rows/deleted_rows к уже обновлённой таблице.
        st.session_state.graph_epoch += 1
        st.rerun()

    st.divider()
    st.caption(
        "Действия ниже работают с уже применённой моделью: если вы правили "
        "таблицы, сначала нажмите «Применить изменения», иначе неприменённые "
        "правки в них не попадут."
    )
    with st.expander("Добавить вершину", expanded=False):
        new_kind = st.selectbox(
            "Тип новой вершины",
            [INDICATOR, MEASURE, ENVIRONMENT],
            format_func=lambda kind: KIND_LABELS[kind],
            key=f"new_kind_{epoch}",
        )
        if new_kind == ENVIRONMENT:
            st.caption(
                "Новый фактор среды по умолчанию равен 0 во всех сценариях; "
                "его значения задаются в YAML после экспорта."
            )
        elif new_kind == MEASURE:
            st.caption(
                "Новая мера попадёт в перебор планов; в текущий план её можно "
                "добавить через «Свой набор…» в боковой панели."
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
                st.session_state.graph_epoch += 1
                st.session_state.graph_flash = (
                    f"Добавлена вершина {node_id}. Не забудьте добавить дуги."
                )
                st.rerun()

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
                "Связанные дуги будут удалены. Мера исчезнет из планов, "
                "фактор среды — из сценариев. Целевой показатель "
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
                st.session_state.graph_epoch += 1
                st.session_state.graph_flash = f"Вершина {victim} удалена."
                st.rerun()

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
                        st.session_state.graph_flash = (
                            f"Сохранено: {target_path.name}."
                        )
                        st.rerun()

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
        file_name=f"{plan.name}_{scenario.name}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Скачать summary.json",
        data=json.dumps(summary, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name=f"{plan.name}_{scenario.name}_summary.json",
        mime="application/json",
    )
