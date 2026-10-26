from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_reproduce_thesis_writes_consistent_tables(tmp_path):
    out = tmp_path / "thesis"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reproduce_thesis.py"),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr

    section5 = (out / "section5.tex").read_text(encoding="utf-8")
    theta = (out / "theta_sensitivity.tex").read_text(encoding="utf-8")
    recommendation = (out / "recommendation.md").read_text(encoding="utf-8")

    assert r"\{m_{1},m_{3}\}" in section5
    assert r"\{m_{3},m_{4}\}" in section5
    assert "0,300" in theta
    assert "0,425" in theta
    # при 0,425 P1 уже не проходит — в таблице должен быть прочерк
    assert r"0,425 & $\{m_{1},m_{3}\}$ & ---" in theta
    assert "Гибрид" in recommendation
    assert (out / "plan_scenario_matrix.csv").exists()
    assert (out / "theta_sensitivity.csv").exists()
