from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_streamlit_app_runs_without_exceptions():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception, [str(exc) for exc in at.exception]
    assert at.title
    labels = [metric.label for metric in at.metric]
    assert "k = R / C" in labels


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
