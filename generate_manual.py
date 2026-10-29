#!/usr/bin/env python3
"""Сборка руководства пользователя: Markdown -> PDF.

Источник текста: docs/Руководство_пользователя.md
Результат:      Руководство_пользователя.pdf

Скрипт также сверяет ключевые числа демо-конфига с теми, что приведены
в руководстве. Если конфиг изменился, сборка остановится с сообщением,
чтобы в документе не остались устаревшие числа.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from fpdf import FPDF, FontFace
from fpdf.enums import TableBordersLayout, TableCellFillMode
from PIL import Image

from graphcreativity.engine import SimOptions
from graphcreativity.metrics import (
    compare_named_plans,
    compare_plans,
    critical_event_combinations,
    critical_removal_sets,
    minimal_sufficient_sets,
    pareto_front,
    scenario_robustness,
)
from graphcreativity.model import load_model

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
SOURCE = DOCS / "Руководство_пользователя.md"
OUTPUT = ROOT / "Руководство_пользователя.pdf"
CONFIG = ROOT / "configs" / "karkas.yaml"

PAGE_W = 210.0
PAGE_H = 297.0
MARGIN = 18.0
CONTENT_W = PAGE_W - 2 * MARGIN

FONT_DIR = Path("/usr/share/fonts/TTF")
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
FONT_ITALIC = FONT_DIR / "DejaVuSans-Oblique.ttf"
FONT_BOLD_ITALIC = FONT_DIR / "DejaVuSans-BoldOblique.ttf"
FONT_MONO = FONT_DIR / "DejaVuSansMono.ttf"

BLUE = (31, 78, 121)
DARK = (34, 34, 34)
GREY = (110, 110, 110)
LIGHT_BLUE = (238, 244, 250)
LIGHT_GREY = (246, 248, 250)
NOTE_BG = (255, 249, 230)
NOTE_LINE = (214, 158, 46)
TABLE_ALT = (246, 248, 250)


# ---------------------------------------------------------------------------
# Проверка чисел демо-конфига
# ---------------------------------------------------------------------------

def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.{digits}f}".replace(".", ",")


def check_demo_numbers() -> dict:
    model = load_model(CONFIG)
    options = SimOptions()
    named = {row["plan"]: row for row in compare_named_plans(model, "nominal", options)}
    rows = compare_plans(model, "nominal", options, include_empty=True)
    mss = minimal_sufficient_sets(rows)
    critical = critical_removal_sets(rows, model.measure_ids)
    front = pareto_front(rows)
    robust_hybrid = scenario_robustness(model, "hybrid", options=options)
    robust_p1 = scenario_robustness(model, "P1", options=options)
    events_p1 = critical_event_combinations(model, "P1", options=options)
    events_hybrid = critical_event_combinations(model, "hybrid", options=options)

    expected = {
        "hybrid_t": 30.0,
        "hybrid_cost": 43.0,
        "hybrid_k": 0.74,
        "full_t": 23.0,
        "full_cost": 53.0,
        "full_k": 0.68,
        "p1_t": 36.0,
        "p1_cost": 50.0,
        "p1_k": 0.50,
    }
    actual = {
        "hybrid_t": named["hybrid"]["t_goal"],
        "hybrid_cost": named["hybrid"]["cost_total"],
        "hybrid_k": named["hybrid"]["k_T"],
        "full_t": named["full"]["t_goal"],
        "full_cost": named["full"]["cost_total"],
        "full_k": named["full"]["k_T"],
        "p1_t": named["P1"]["t_goal"],
        "p1_cost": named["P1"]["cost_total"],
        "p1_k": named["P1"]["k_T"],
    }
    problems = []
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if actual_value is None or abs(actual_value - expected_value) > 0.011:
            problems.append(f"{key}: ожидалось {expected_value}, получено {actual_value}")

    if set(mss) != {("m1", "m3"), ("m3", "m4")}:
        problems.append(f"МДН: {mss}")
    if set(critical) != {("m3",), ("m1", "m4")}:
        problems.append(f"Критические меры: {critical}")
    if {row["plan"] for row in front} != {"m1+m3+m4", "m1+m3"}:
        problems.append(f"Парето: {[row['plan'] for row in front]}")
    if robust_hybrid["radius"] != 3 or abs(robust_hybrid["success_share"] - 1.0) > 1e-9:
        problems.append(
            f"Устойчивость гибрида: радиус {robust_hybrid['radius']}, "
            f"доля {robust_hybrid['success_share']}"
        )
    if robust_p1["radius"] != 0 or abs(robust_p1["success_share"] - 0.2) > 1e-9:
        problems.append(
            f"Устойчивость P1: радиус {robust_p1['radius']}, "
            f"доля {robust_p1['success_share']}"
        )
    if len(events_p1) != 3:
        problems.append(f"Критические события P1: {events_p1}")
    if events_hybrid:
        problems.append(f"Критические события гибрида: {events_hybrid}")

    theta30 = SimOptions(theta=0.30)
    mss30 = minimal_sufficient_sets(
        compare_plans(model, "nominal", theta30, include_empty=True)
    )
    if ("m1",) not in mss30:
        problems.append(f"При θ=0,30 нет {{m1}}: {mss30}")

    budget45 = SimOptions(theta=0.42, budget_max=45.0)
    mss45 = minimal_sufficient_sets(
        compare_plans(model, "nominal", budget45, include_empty=True)
    )
    if ("m3", "m4") in mss45:
        problems.append(f"При бюджете 45 остался {{m3,m4}}: {mss45}")

    if problems:
        print("Числа в руководстве не совпадают с текущим конфигом:", file=sys.stderr)
        for problem in problems:
            print("  -", problem, file=sys.stderr)
        raise SystemExit(1)

    return {
        "model": model,
        "named": named,
        "rows": rows,
        "mss": mss,
        "critical": critical,
        "front": front,
        "robust_hybrid": robust_hybrid,
        "robust_p1": robust_p1,
        "events_p1": events_p1,
    }


# ---------------------------------------------------------------------------
# Разбор Markdown
# ---------------------------------------------------------------------------

@dataclass
class Block:
    kind: str
    payload: dict


def _clean_inline(text: str) -> str:
    """Убрать backticks, оставив текст; fpdf сам разберёт **жирный**."""
    return text.replace("`", "")


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_markdown(path: Path) -> list[Block]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    i = 0
    in_cover = False
    pending_widths: list[float] | None = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "<!-- cover:start -->":
            in_cover = True
            i += 1
            continue
        if stripped == "<!-- cover:end -->":
            in_cover = False
            i += 1
            continue
        if in_cover:
            i += 1
            continue
        if stripped.startswith("<!-- widths:"):
            raw = stripped[len("<!-- widths:") :].rstrip("->").strip()
            pending_widths = [float(part.strip()) for part in raw.split(",")]
            i += 1
            continue
        if stripped.startswith("<!--"):
            i += 1
            continue
        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            i += 1
            blocks.append(Block("formula", {"text": "\n".join(code_lines)}))
            continue

        if stripped.startswith("#### "):
            blocks.append(Block("h4", {"text": stripped[5:].strip()}))
            i += 1
            continue
        if stripped.startswith("### "):
            blocks.append(Block("h3", {"text": stripped[4:].strip()}))
            i += 1
            continue
        if stripped.startswith("## "):
            blocks.append(Block("h2", {"text": stripped[3:].strip()}))
            i += 1
            continue
        if stripped.startswith("# "):
            # Заголовок первого уровня встречается только на обложке.
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            if len(table_lines) >= 2:
                header = _table_cells(table_lines[0])
                body = [
                    _table_cells(row)
                    for row in table_lines[2:]
                    if not re.fullmatch(r"\|[\s:|-]+\|", row)
                ]
                blocks.append(
                    Block(
                        "table",
                        {
                            "header": header,
                            "rows": body,
                            "widths": pending_widths,
                        },
                    )
                )
                pending_widths = None
            continue

        if stripped.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if match:
                blocks.append(
                    Block("image", {"caption": match.group(1), "path": match.group(2)})
                )
                i += 1
                continue

        if stripped.startswith(">"):
            note_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                note_lines.append(lines[i].strip()[1:].strip())
                i += 1
            blocks.append(Block("note", {"text": " ".join(note_lines)}))
            continue

        if stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append(Block("bullets", {"items": items}))
            continue

        if re.match(r"^\d+\.\s", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            blocks.append(Block("numbered", {"items": items}))
            continue

        if stripped == "---":
            blocks.append(Block("rule", {}))
            i += 1
            continue

        paragraph = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("|")
                or nxt.startswith("```")
                or nxt.startswith("![")
                or nxt.startswith(">")
                or nxt.startswith("- ")
                or re.match(r"^\d+\.\s", nxt)
                or nxt.startswith("<!--")
                or nxt == "---"
            ):
                break
            paragraph.append(nxt)
            i += 1
        blocks.append(Block("p", {"text": " ".join(paragraph)}))

    return blocks


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class ManualPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=MARGIN)
        self.set_title("Руководство пользователя: графовая модель креативности")
        self.set_author("graph-creativity")
        self.set_subject("Параметры, показатели и формулы")
        self.add_font("DejaVu", "", str(FONT_REGULAR))
        self.add_font("DejaVu", "B", str(FONT_BOLD))
        self.add_font("DejaVu", "I", str(FONT_ITALIC))
        self.add_font("DejaVu", "BI", str(FONT_BOLD_ITALIC))
        self.add_font("DejaVuMono", "", str(FONT_MONO))
        self.alias_nb_pages()
        self.outline_entries: list[tuple[int, str, int]] = []

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "I", 7.8)
        self.set_text_color(*GREY)
        self.set_y(10)
        self.cell(
            0,
            5,
            "Руководство пользователя · графовая модель креативности",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_draw_color(220, 225, 232)
        self.line(MARGIN, 16, PAGE_W - MARGIN, 16)
        self.set_y(MARGIN)

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 5, f"Стр. {self.page_no()} / {{nb}}", align="C")


def ensure_space(pdf: ManualPDF, height: float) -> None:
    if pdf.get_y() + height > PAGE_H - MARGIN:
        pdf.add_page()


def render_toc_pages(pdf: ManualPDF, entries: list[tuple[int, str, int]], offset: int) -> None:
    """Нарисовать содержание. offset — номер страницы перед первой страницей тела."""
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 10, "Содержание", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.5)
    pdf.line(MARGIN, pdf.get_y() + 1, PAGE_W - MARGIN, pdf.get_y() + 1)
    pdf.ln(6)
    for level, name, body_page in entries:
        page_number = body_page + offset
        indent = 5.0 * level
        pdf.set_font("DejaVu", "B" if level == 0 else "", 10.2 if level == 0 else 8.8)
        pdf.set_text_color(*(BLUE if level == 0 else DARK))
        if pdf.get_y() > PAGE_H - MARGIN - 8:
            pdf.add_page()
        link = pdf.add_link(page=page_number)
        pdf.set_x(MARGIN + indent)
        pdf.cell(
            CONTENT_W - indent - 16,
            5.6 if level == 0 else 4.8,
            name,
            link=link,
            new_x="RIGHT",
            new_y="TOP",
        )
        pdf.set_font("DejaVu", "", 8.2)
        pdf.set_text_color(*GREY)
        pdf.cell(16, 5.6 if level == 0 else 4.8, str(page_number), align="R", link=link)
        pdf.ln(5.6 if level == 0 else 4.8)


def render_cover(pdf: ManualPDF) -> None:
    pdf.add_page()
    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 0, PAGE_W, 72, style="F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("DejaVu", "B", 24)
    pdf.set_xy(MARGIN, 25)
    pdf.multi_cell(CONTENT_W, 11, "Руководство пользователя", align="L")

    pdf.set_font("DejaVu", "", 13)
    pdf.set_xy(MARGIN, 44)
    pdf.multi_cell(
        CONTENT_W,
        6.5,
        "Графовая модель\nповышения креативности предприятия",
        align="L",
    )

    pdf.set_text_color(70, 70, 70)
    pdf.set_font("DejaVu", "", 10.5)
    pdf.set_xy(MARGIN, 86)
    pdf.cell(0, 6, "Версия модели 1.2-demo  ·  октябрь 2026")

    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.8)
    pdf.line(MARGIN, 96, PAGE_W - MARGIN, 96)

    pdf.set_text_color(*DARK)
    pdf.set_font("DejaVu", "", 11)
    pdf.set_xy(MARGIN, 106)
    pdf.multi_cell(
        CONTENT_W,
        6.2,
        "Настоящее руководство объясняет назначение каждого ползунка и поля, "
        "смысл показателей и формулы, по которым считаются эффект, затраты "
        "и устойчивость.",
        align="J",
    )

    pdf.ln(5)
    pdf.set_font("DejaVu", "B", 10.5)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 7, "Что внутри", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(*DARK)
    for item in [
        "единый исследовательский интерфейс;",
        "все параметры боковой панели и их диапазоны;",
        "KPI, таблицы, вкладки и редактор графа;",
        "математическая модель и аналитические объекты;",
        "справочник демо-конфига и ответы на типовые вопросы.",
    ]:
        pdf.set_x(MARGIN + 4)
        pdf.multi_cell(CONTENT_W - 4, 5.6, "•  " + item)

    pdf.set_y(PAGE_H - 38)
    pdf.set_draw_color(220, 225, 232)
    pdf.line(MARGIN, PAGE_H - 32, PAGE_W - MARGIN, PAGE_H - 32)
    pdf.set_font("DejaVu", "I", 8.5)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(
        CONTENT_W,
        4.8,
        "Документ описывает экранные элементы, параметры и расчётную логику "
        "приложения.",
        align="C",
    )


def render_h2(pdf: ManualPDF, text: str) -> None:
    text = _clean_inline(text)
    ensure_space(pdf, 72)
    pdf.start_section(text, level=0)
    pdf.outline_entries.append((0, text, pdf.page_no()))
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(*BLUE)
    pdf.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.45)
    pdf.line(MARGIN, pdf.get_y() + 1, PAGE_W - MARGIN, pdf.get_y() + 1)
    pdf.ln(4)


def render_h3(pdf: ManualPDF, text: str, reserve: float = 46) -> None:
    text = _clean_inline(text)
    ensure_space(pdf, reserve)
    pdf.ln(1.5)
    pdf.start_section(text, level=1)
    pdf.outline_entries.append((1, text, pdf.page_no()))
    pdf.set_font("DejaVu", "B", 12.2)
    pdf.set_text_color(*BLUE)
    pdf.multi_cell(0, 6.4, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.2)


def render_h4(pdf: ManualPDF, text: str) -> None:
    text = _clean_inline(text)
    ensure_space(pdf, 18)
    pdf.ln(1)
    pdf.set_font("DejaVu", "B", 10.6)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(0, 5.6, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.8)


def render_paragraph(pdf: ManualPDF, text: str) -> None:
    ensure_space(pdf, 14)
    pdf.set_font("DejaVu", "", 10.3)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(
        0,
        5.35,
        _clean_inline(text),
        markdown=True,
        align="L",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(1.4)


def render_bullets(pdf: ManualPDF, items: list[str]) -> None:
    pdf.set_text_color(*DARK)
    for item in items:
        ensure_space(pdf, 10)
        pdf.set_font("DejaVu", "", 10.1)
        pdf.set_x(MARGIN + 3)
        pdf.multi_cell(
            CONTENT_W - 3,
            5.15,
            "•  " + _clean_inline(item),
            markdown=True,
            align="L",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(0.6)
    pdf.ln(1.2)


def render_numbered(pdf: ManualPDF, items: list[str]) -> None:
    pdf.set_text_color(*DARK)
    for number, item in enumerate(items, start=1):
        ensure_space(pdf, 10)
        pdf.set_font("DejaVu", "", 10.1)
        pdf.set_x(MARGIN + 3)
        pdf.multi_cell(
            CONTENT_W - 3,
            5.15,
            f"{number}.  " + _clean_inline(item),
            markdown=True,
            align="L",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(0.6)
    pdf.ln(1.2)


def render_note(pdf: ManualPDF, text: str) -> None:
    ensure_space(pdf, 24)
    pdf.set_fill_color(*NOTE_BG)
    pdf.set_draw_color(*NOTE_LINE)
    pdf.set_text_color(80, 55, 10)
    pdf.set_font("DejaVu", "", 9.8)
    pdf.multi_cell(
        0,
        5.2,
        _clean_inline(text),
        border="L",
        fill=True,
        markdown=True,
        align="J",
        padding=2.6,
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2.2)


def render_formula(pdf: ManualPDF, text: str) -> None:
    ensure_space(pdf, 18)
    pdf.set_fill_color(*LIGHT_GREY)
    pdf.set_draw_color(210, 216, 224)
    pdf.set_text_color(28, 52, 80)
    pdf.set_font("DejaVuMono", "", 8.3)
    pdf.multi_cell(
        0,
        4.35,
        text,
        border=1,
        fill=True,
        align="L",
        padding=2.4,
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2.2)


def estimated_table_height(rows: list[list[str]]) -> float:
    return 10.0 + len(rows) * 7.6


def render_table(pdf: ManualPDF, header: list[str], rows: list[list[str]], widths: list[float] | None) -> None:
    ncols = max(1, len(header))
    if not widths or len(widths) != ncols:
        widths = [100.0 / ncols] * ncols
    total = sum(widths)
    if total <= 0:
        widths = [100.0 / ncols] * ncols
    else:
        widths = [w * 100.0 / total for w in widths]
    col_widths = [w * CONTENT_W / 100.0 for w in widths]

    # Небольшие таблицы лучше не разрывать между страницами: перед ними
    # резервируем примерную высоту. Длинные таблицы могут разрываться.
    if len(rows) <= 12:
        estimated = estimated_table_height(rows)
        ensure_space(pdf, estimated)
    else:
        ensure_space(pdf, 32)

    if ncols >= 6:
        font_size = 7.3
    elif ncols == 5:
        font_size = 7.8
    elif ncols == 4:
        font_size = 8.2
    elif ncols == 3:
        font_size = 8.7
    else:
        font_size = 9.2

    pdf.set_font("DejaVu", "", font_size)
    pdf.set_text_color(*DARK)
    with pdf.table(
        col_widths=col_widths,
        width=CONTENT_W,
        first_row_as_headings=True,
        text_align="LEFT",
        line_height=4.25,
        padding=1.25,
        borders_layout=TableBordersLayout.HORIZONTAL_LINES,
        cell_fill_mode=TableCellFillMode.ROWS,
        cell_fill_color=TABLE_ALT,
        headings_style=FontFace(
            emphasis="BOLD",
            color=(255, 255, 255),
            fill_color=BLUE,
            size_pt=font_size,
        ),
        markdown=True,
    ) as table:
        row = table.row()
        for cell in header:
            row.cell(_clean_inline(cell))
        for body_row in rows:
            row = table.row()
            for idx in range(ncols):
                cell = body_row[idx] if idx < len(body_row) else ""
                row.cell(_clean_inline(cell))
    pdf.ln(3.0)


def render_image(pdf: ManualPDF, path: Path, caption: str, number: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Нет изображения: {path}")
    with Image.open(path) as img:
        iw, ih = img.size
    if path.name.startswith("02_"):
        width = 56.0
    else:
        width = CONTENT_W
    height = width * ih / iw
    ensure_space(pdf, height + 14)
    x = (PAGE_W - width) / 2
    pdf.image(str(path), x=x, y=pdf.get_y(), w=width)
    pdf.set_y(pdf.get_y() + height + 2)
    pdf.set_font("DejaVu", "I", 8.4)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(
        0,
        4.6,
        f"Рис. {number}. {caption}",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2.5)


def render_rule(pdf: ManualPDF) -> None:
    ensure_space(pdf, 8)
    pdf.set_draw_color(220, 225, 232)
    pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
    pdf.ln(3)


def render_blocks(pdf: ManualPDF, blocks: list[Block]) -> None:
    image_number = 1
    for index, block in enumerate(blocks):
        kind = block.kind
        payload = block.payload
        if kind == "h2":
            render_h2(pdf, payload["text"])
        elif kind == "h3":
            reserve = 46.0
            if index + 1 < len(blocks) and blocks[index + 1].kind == "table":
                following_rows = blocks[index + 1].payload["rows"]
                if len(following_rows) <= 12:
                    reserve = max(reserve, estimated_table_height(following_rows) + 10.0)
            render_h3(pdf, payload["text"], reserve=reserve)
        elif kind == "h4":
            render_h4(pdf, payload["text"])
        elif kind == "p":
            paragraph_text = payload["text"]
            if (
                paragraph_text.strip() in {"**Формулы:**", "**Формула:**"}
                and index + 1 < len(blocks)
                and blocks[index + 1].kind == "formula"
            ):
                ensure_space(pdf, 34)
            render_paragraph(pdf, paragraph_text)
        elif kind == "bullets":
            render_bullets(pdf, payload["items"])
        elif kind == "numbered":
            render_numbered(pdf, payload["items"])
        elif kind == "note":
            render_note(pdf, payload["text"])
        elif kind == "formula":
            render_formula(pdf, payload["text"])
        elif kind == "table":
            render_table(
                pdf,
                payload["header"],
                payload["rows"],
                payload["widths"],
            )
        elif kind == "image":
            render_image(
                pdf,
                DOCS / payload["path"],
                payload["caption"],
                image_number,
            )
            image_number += 1
        elif kind == "rule":
            render_rule(pdf)
        else:
            raise ValueError(f"Неизвестный блок: {kind}")


def build_pdf(blocks: list[Block]) -> None:
    # Первый проход: тело без обложки и содержания. Нужен только для того,
    # чтобы получить номера страниц разделов и количество страниц тела.
    body = ManualPDF()
    body.add_page()
    render_blocks(body, blocks)
    entries = list(body.outline_entries)
    body_pages = body.pages_count

    # Второй проход: пробуем содержание, чтобы узнать его длину. Номера
    # страниц при этом ещё относительные, но на количество страниц это не влияет.
    toc_probe = ManualPDF()
    render_toc_pages(toc_probe, entries, offset=0)
    toc_pages = toc_probe.pages_count

    offset = 1 + toc_pages  # обложка + страницы содержания
    pdf = ManualPDF()
    render_cover(pdf)
    render_toc_pages(pdf, entries, offset=offset)
    if pdf.pages_count != toc_pages + 1:
        raise RuntimeError(
            f"Содержание заняло {pdf.pages_count - 1} стр. вместо {toc_pages}"
        )
    pdf.add_page()
    render_blocks(pdf, blocks)
    pdf.output(str(OUTPUT))
    print(f"PDF: {OUTPUT} ({pdf.pages_count} стр., тело {body_pages} стр.)")


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Нет исходника: {SOURCE}")
    demo = check_demo_numbers()
    print(
        "Проверка чисел: OK · "
        f"гибрид T={_fmt(demo['named']['hybrid']['t_goal'])}, "
        f"C={_fmt(demo['named']['hybrid']['cost_total'])}, "
        f"k={_fmt(demo['named']['hybrid']['k_T'])}"
    )
    blocks = parse_markdown(SOURCE)
    print(f"Блоков Markdown: {len(blocks)}")
    build_pdf(blocks)


if __name__ == "__main__":
    main()
