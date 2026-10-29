from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_streamlit_app_runs_without_exceptions():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception, [str(exc) for exc in at.exception]
    titles = [title.value for title in at.title]
    assert "🕸️ Графовая модель" in titles
    assert not any(radio.label == "Режим" for radio in at.radio)
    labels = [metric.label for metric in at.metric]
    assert "k = R / C" in labels
    assert "Меры в комплексе" in labels
    tab_labels = [tab.label for tab in at.tabs]
    assert tab_labels == [
        "Динамика",
        "Показатели",
        "Сравнение планов",
        "Граф",
        "Данные",
    ]
    subheaders = [block.value for block in at.subheader]
    assert "Матрица план × сценарий" not in subheaders
    assert "Устойчивость к сценариям" not in subheaders
    assert "Критические комбинации событий" not in subheaders
    assert not any(
        "сценарий" in caption.value.lower() for caption in at.caption
    )


def test_sidebar_has_no_scenario_or_plan_selectors():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception

    sidebar_labels = [box.label for box in at.sidebar.selectbox]
    assert "Внешняя среда" not in sidebar_labels
    assert "Комплекс мер" not in sidebar_labels

    plan = next(
        box for box in at.selectbox if box.label == "Активный комплекс мер"
    )
    assert "Все меры" not in plan.options
    assert "Свой комплекс мер" in plan.options


def test_vertex_spaces_open_measure_and_environment_dialogs():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception

    space = next(
        radio for radio in at.radio if radio.label == "Пространство вершин"
    )
    assert space.options == ["Показатели X", "Меры M", "Среда Y"]

    space.set_value("Меры M").run()
    next(button for button in at.button if button.label == "Изменить").click().run()

    assert not at.exception
    assert any(field.label == "Название" for field in at.text_input)
    assert any(slider.label == "Начало, мес." for slider in at.slider)
    assert any(slider.label == "Интенсивность" for slider in at.slider)
    assert any(
        field.label == "Продолжительность, мес."
        for field in at.number_input
    )
    assert any(
        field.label == "Периодичность, мес."
        for field in at.number_input
    )
    assert any(slider.label == "Конец, мес." for slider in at.slider)
    assert not any(
        checkbox.label.startswith("Входит в план")
        for checkbox in at.checkbox
    )

    at = AppTest.from_file(str(APP), default_timeout=90).run()
    space = next(
        radio for radio in at.radio if radio.label == "Пространство вершин"
    )
    space.set_value("Среда Y").run()
    assert not any(
        box.label == "Сценарий для настройки отклонений"
        for box in at.selectbox
    )
    assert not any(
        "сценарий" in caption.value.lower() for caption in at.caption
    )
    next(button for button in at.button if button.label == "Изменить").click().run()

    assert not at.exception
    assert any(
        slider.label == "Отклонение от номинала" for slider in at.slider
    )
    assert not any(
        "сценарий" in caption.value.lower() for caption in at.caption
    )


def test_named_plan_does_not_allow_adding_measure():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception

    space = next(
        radio for radio in at.radio if radio.label == "Пространство вершин"
    )
    space.set_value("Меры M").run()

    m4_button = next(
        button
        for button in at.button
        if button.key and "edit_measure_hybrid_m4_" in button.key
    )
    m4_button.click().run()

    assert not at.exception
    assert not any(
        slider.label == "Начало, мес." for slider in at.slider
    )
    assert not any(
        checkbox.label.startswith("Входит в план")
        for checkbox in at.checkbox
    )

    next(button for button in at.button if button.label == "Сохранить").click().run()
    assert not at.exception
    assert "m4" not in at.session_state["plan_rows"]["hybrid"]


def test_custom_plan_exposes_measure_membership_and_schedule_controls():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception

    plan = next(
        box for box in at.selectbox if box.label == "Активный комплекс мер"
    )
    plan.set_value("__custom__").run()

    assert not at.exception
    assert at.session_state["active_plan_name"] == "__custom__"
    space = next(
        radio for radio in at.radio if radio.label == "Пространство вершин"
    )
    space.set_value("Меры M").run()
    measures = next(
        box for box in at.multiselect if box.label == "Меры в комплексе"
    )
    assert set(measures.value) == {"m1", "m3"}

    measures.set_value(["m1", "m4"]).run()
    assert not at.exception
    assert set(at.session_state["custom_plan_programs"]) == {"m1", "m4"}
    metric = next(m for m in at.metric if m.label == "Меры в комплексе")
    assert metric.value == "2 / 3"

    m4_button = next(
        button
        for button in at.button
        if button.key and "__custom__" in button.key and "_m4_" in button.key
    )
    m4_button.click().run()

    assert not at.exception
    assert any(slider.label == "Начало, мес." for slider in at.slider)
    assert any(slider.label == "Интенсивность" for slider in at.slider)
    assert any(slider.label == "Конец, мес." for slider in at.slider)
    assert any(
        field.label == "Продолжительность, мес."
        for field in at.number_input
    )
    assert any(
        field.label == "Периодичность, мес."
        for field in at.number_input
    )

    at = AppTest.from_file(str(APP), default_timeout=90).run()
    plan = next(
        box for box in at.selectbox if box.label == "Активный комплекс мер"
    )
    plan.set_value("__custom__").run()
    space = next(
        radio for radio in at.radio if radio.label == "Пространство вершин"
    )
    space.set_value("Меры M").run()
    measures = next(
        box for box in at.multiselect if box.label == "Меры в комплексе"
    )
    measures.set_value(["m1"]).run()
    assert not at.exception
    assert set(at.session_state["custom_plan_programs"]) == {"m1"}
    metric = next(m for m in at.metric if m.label == "Меры в комплексе")
    assert metric.value == "1 / 3"


def test_graph_editor_applies_and_resets_edge_weight():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception

    editor_key = next(
        key for key in at.session_state if key.startswith("edge_editor_")
    )
    original_weight = at.session_state["edge_rows"][0]["w"]

    at.session_state[editor_key] = {
        "edited_rows": {0: {"w": 0.10}},
        "added_rows": [],
        "deleted_rows": [],
    }
    next(
        button
        for button in at.button
        if button.key and button.key.startswith("FormSubmitter:graph_editor_")
    ).click()
    at.run()

    assert not at.exception
    assert at.session_state["edge_rows"][0]["w"] == 0.10
    assert at.session_state["last_valid_model"].edges[0].w == 0.10

    edge_count = len(at.session_state["edge_rows"])
    editor_key = next(
        key for key in at.session_state if key.startswith("edge_editor_")
    )
    at.session_state[editor_key] = {
        "edited_rows": {},
        "added_rows": [
            {
                "from": "m1",
                "to": "X4",
                "w": 0.20,
                "delay": 3,
                "optional": False,
                "source": "Э",
                "mechanism": "проверка редактора",
            }
        ],
        "deleted_rows": [],
    }
    next(
        button
        for button in at.button
        if button.key and button.key.startswith("FormSubmitter:graph_editor_")
    ).click()
    at.run()

    assert not at.exception
    assert len(at.session_state["edge_rows"]) == edge_count + 1
    new_edges = [
        row
        for row in at.session_state["edge_rows"]
        if row["from"] == "m1" and row["to"] == "X4"
    ]
    assert len(new_edges) == 1

    # Повторный прогон не должен заново применять added_rows из старого
    # состояния data_editor.
    at.run()
    assert len(at.session_state["edge_rows"]) == edge_count + 1
    assert (
        sum(
            row["from"] == "m1" and row["to"] == "X4"
            for row in at.session_state["edge_rows"]
        )
        == 1
    )

    next(button for button in at.button if button.label == "Сбросить").click()
    at.run()

    assert not at.exception
    assert len(at.session_state["edge_rows"]) == edge_count
    assert at.session_state["edge_rows"][0]["w"] == original_weight
    assert at.session_state["last_valid_model"].edges[0].w == original_weight
