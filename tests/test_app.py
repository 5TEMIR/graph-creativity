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

    at = AppTest.from_file(str(APP), default_timeout=90).run()
    space = next(
        radio for radio in at.radio if radio.label == "Пространство вершин"
    )
    space.set_value("Среда Y").run()
    next(button for button in at.button if button.label == "Изменить").click().run()

    assert not at.exception
    assert any(
        slider.label == "Отклонение от номинала" for slider in at.slider
    )


def test_custom_plan_exposes_measure_schedule_controls():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception

    plan = next(field for field in at.selectbox if field.label == "Комплекс мер")
    plan.set_value("__custom__").run()

    assert not at.exception
    assert any(slider.label == "Старт m1, мес." for slider in at.slider)
    assert any(slider.label == "Интенсивность m1" for slider in at.slider)
    assert any(
        field.label == "Продолжительность m1, мес."
        for field in at.number_input
    )
    assert any(
        checkbox.label == "m1: до конца периода"
        for checkbox in at.checkbox
    )


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
