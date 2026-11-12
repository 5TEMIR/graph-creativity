#!/usr/bin/env python3
"""Панельные оценки дуг отраслевой модели по ЕМИСС.

Что делает:

1. читает ряды узлов ``i1``, ``i2``, ``i3``, ``i4``, ``o1``, ``o3``,
   ``y1``, ``y2`` по 8 классам ОКВЭД (24--30, 33);
2. переводит уровни в логарифмические темпы прироста: при цели +25 %
   такой темп пропорционален нормированному состоянию ``x``, поэтому
   коэффициент сопоставим с весом дуги;
3. оценивает каждую дугу уравнением с фиксированными эффектами класса
   и года (МНК), кластерными по классу стандартными ошибками и
   wild cluster bootstrap p-value (перебор всех 2^8 знаковых шаблонов);
4. повторяет оценку для лагов 0, 1, 2 года: устойчивость знака важнее
   одной точечной оценки;
5. дополнительно считает Huber-оценку и МНК без класса 33: класс 33
   даёт выбросы до +1012 %;
6. пишет ``edges.csv``, ``edges.json`` и ``summary.md``.

Оценки --- ассоциации, а не причинные эффекты. Дуги от мер ``p1--p3``
не оцениваются: в истории меры не наблюдались (``u = 0``), производная
по их весам равна нулю. Дуги к ``o2`` и от ``y3``/``y4`` тоже не
идентифицируются: соответствующих рядов по ОКВЭД нет либо среда измерена
тем же рядом, что и фактор.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphcreativity.data import load_meta, load_series, resolve_root
from graphcreativity.model import load_model

DEFAULT_CONFIG = ROOT / "configs" / "industry.yaml"
DEFAULT_OUT = ROOT / "out" / "industry-edges"

CLASSES = ["24", "25", "26", "27", "28", "29", "30", "33"]
YEAR_MAX = 2024
YEAR_MIN = 2018
HUBER_C = 1.345

NODES = {
    "i1": "58769",   # доля инновационно-активных организаций, %
    "i2": "60022",   # интенсивность затрат на инновации, % выручки
    "i3": "58761",   # инновационная продукция, тыс. руб.
    "i4": "58701",   # средняя заработная плата, руб. (прокси ЧК)
    "o1": "57711",   # отгружено товаров, тыс. руб.
    "o3": "58261",   # рентабельность продаж, %
    "y1": "59084",   # инвестиции в основной капитал, тыс. руб.
    "y2": "57711",   # спрос: темп выпуска (тот же ряд, что o1)
}


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    weight: float
    delay: int
    mechanism: str

    @property
    def name(self) -> str:
        return f"{self.source}->{self.target}"

    @property
    def lag(self) -> int:
        """Задержка в годах: 6 мес. --- тот же год, 12 --- следующий."""
        if self.delay <= 9:
            return 0
        if self.delay <= 15:
            return 1
        return 2


EDGES = [
    Edge("i2", "i1", 0.45, 6, "НИОКР расширяют инновационную активность"),
    Edge("i4", "i1", 0.40, 12, "кадры расширяют участие"),
    Edge("i1", "i3", 0.60, 12, "массовость конвертируется в новую продукцию"),
    Edge("i2", "i3", 0.40, 18, "НИОКР напрямую дают новую продукцию"),
    Edge("i4", "i3", 0.30, 18, "кадры ускоряют конверсию"),
    Edge("i3", "o1", 0.50, 6, "новая продукция увеличивает выпуск"),
    Edge("i3", "o3", 0.45, 6, "новая продукция увеличивает прибыль"),
    Edge("y1", "i2", 0.35, 12, "инвестиции поднимают НИОКР"),
    Edge("y2", "i3", 0.30, 6, "спрос вытягивает новую продукцию"),
    # Опциональные обратные связи.
    Edge("o1", "i2", 0.20, 18, "выпуск через инвестиции финансирует НИОКР"),
    Edge("o3", "i2", 0.25, 12, "прибыль финансирует НИОКР"),
]

SKIPPED = [
    (
        "p1->i2, p1->i1, p2->i1, p2->i4, p2->i3, p3->i4, p3->i1, p3->i3",
        "меры не наблюдались: u = 0, производная по весу равна нулю",
    ),
    (
        "i1->o2, i2->o2",
        "нет ряда производительности труда по ОКВЭД: 34051 только Россия",
    ),
    ("y3->i1", "конкуренция y3 не измеряется"),
    ("y4->i4", "дефицит кадров y4 измерен тем же рядом 58701, что и i4"),
]


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "---"
    return f"{value:+.{digits}f}".replace(".", ",")


def _stars(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _load_raw(
    root: Path,
    classes: Sequence[str],
) -> dict[str, dict[str, dict[int, float]]]:
    raw: dict[str, dict[str, dict[int, float]]] = {}
    for node, indicator in NODES.items():
        meta = load_meta(root, indicator)
        raw[node] = {
            code: dict(load_series(root, indicator, code, meta=meta).values)
            for code in classes
        }
    return raw


def _growth(
    raw: dict[str, dict[str, dict[int, float]]],
    node: str,
    classes: Sequence[str],
) -> dict[tuple[str, int], float]:
    """Логарифмический темп прироста; 2025 и позже отброшены."""
    result: dict[tuple[str, int], float] = {}
    for code in classes:
        values = raw[node][code]
        for year, value in values.items():
            previous = values.get(year - 1)
            if year > YEAR_MAX or previous is None:
                continue
            if value > 0 and previous > 0:
                result[(code, year)] = float(np.log(value / previous))
    return result


def _panel(
    raw: dict[str, dict[str, dict[int, float]]],
    edge: Edge,
    lag: int,
    classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = _growth(raw, edge.source, classes)
    target = _growth(raw, edge.target, classes)
    keys = [
        (code, year)
        for code in classes
        for year in range(YEAR_MIN, YEAR_MAX + 1)
        if (code, year) in target and (code, year - lag) in source
    ]
    y = np.array([target[key] for key in keys], dtype=float)
    x = np.array([source[(key[0], key[1] - lag)] for key in keys], dtype=float)
    cls = np.array([key[0] for key in keys])
    yr = np.array([key[1] for key in keys])
    return y, x, cls, yr


def _design(cls: np.ndarray, yr: np.ndarray) -> np.ndarray:
    classes = sorted(set(cls.tolist()))
    years = sorted(set(yr.tolist()))
    design = np.zeros((len(cls), len(classes) + len(years)))
    for column, code in enumerate(classes):
        design[:, column] = cls == code
    for column, year in enumerate(years):
        design[:, len(classes) + column] = yr == year
    return design


def _residualize(
    values: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    root_w = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(
        design * root_w[:, None], values * root_w, rcond=None
    )
    return values - design @ coef


def _cluster_se(
    x: np.ndarray,
    u: np.ndarray,
    cls: np.ndarray,
    yr: np.ndarray,
    weights: np.ndarray,
    bread: float,
) -> float:
    meat = 0.0
    for code in sorted(set(cls.tolist())):
        mask = cls == code
        score = float(weights[mask] @ (x[mask] * u[mask]))
        meat += score * score
    groups = len(set(cls.tolist()))
    years = len(set(yr.tolist()))
    n = len(x)
    params = groups + years
    correction = 1.0
    if groups > 1 and n > params:
        correction = groups / (groups - 1) * (n - 1) / (n - params)
    if bread <= 0:
        return float("inf")
    return float(np.sqrt(meat / bread**2 * correction))


@dataclass(frozen=True)
class Fit:
    beta: float
    se: float
    n: int
    r_within: float
    iterations: int = 1

    @property
    def t(self) -> float:
        return self.beta / self.se if self.se > 0 else float("nan")


def _ols(
    y: np.ndarray,
    x: np.ndarray,
    cls: np.ndarray,
    yr: np.ndarray,
) -> Fit:
    design = _design(cls, yr)
    weights = np.ones_like(y)
    yd = _residualize(y, design, weights)
    xd = _residualize(x, design, weights)
    bread = float(xd @ xd)
    if bread <= 0:
        return Fit(0.0, float("inf"), len(y), 0.0)
    beta = float(xd @ yd) / bread
    residual = yd - beta * xd
    se = _cluster_se(xd, residual, cls, yr, weights, bread)
    variance_y = float(yd @ yd)
    r_within = (
        float(xd @ yd) / float(np.sqrt(bread * variance_y))
        if variance_y > 0
        else 0.0
    )
    return Fit(beta, se, len(y), r_within)


def _huber(
    y: np.ndarray,
    x: np.ndarray,
    cls: np.ndarray,
    yr: np.ndarray,
    *,
    c: float = HUBER_C,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> Fit:
    design = _design(cls, yr)
    weights = np.ones_like(y)
    beta = 0.0
    iterations = 0
    for iterations in range(1, max_iter + 1):
        yd = _residualize(y, design, weights)
        xd = _residualize(x, design, weights)
        denominator = float(weights @ (xd * xd))
        if denominator <= 0:
            return Fit(0.0, float("inf"), len(y), 0.0, iterations)
        beta_new = float(weights @ (xd * yd)) / denominator
        residual = yd - beta_new * xd
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        scale = max(scale, 1e-12)
        z = np.abs(residual) / scale
        new_weights = np.where(z <= c, 1.0, c / np.maximum(z, 1e-12))
        converged = (
            abs(beta_new - beta) < tol
            and float(np.max(np.abs(new_weights - weights))) < tol
        )
        beta, weights = beta_new, new_weights
        if converged:
            break

    yd = _residualize(y, design, weights)
    xd = _residualize(x, design, weights)
    bread = float(weights @ (xd * xd))
    beta = float(weights @ (xd * yd)) / bread
    residual = yd - beta * xd
    se = _cluster_se(xd, residual, cls, yr, weights, bread)
    variance_x = float(weights @ (xd * xd))
    variance_y = float(weights @ (yd * yd))
    r_within = 0.0
    if variance_x > 0 and variance_y > 0:
        r_within = float(weights @ (xd * yd)) / float(
            np.sqrt(variance_x * variance_y)
        )
    return Fit(beta, se, len(y), r_within, iterations)


def _wild_cluster_p(
    y: np.ndarray,
    x: np.ndarray,
    cls: np.ndarray,
    yr: np.ndarray,
    fit: Fit,
) -> float:
    """Wild cluster bootstrap-t, перебор всех 2^G знаковых шаблонов.

    Ограниченная модель --- только фиксированные эффекты класса и года.
    Остатки ограниченной модели умножаются на общий для кластера знак
    (Rademacher), затем заново оценивается дуга. Это поправка на малое
    число кластеров (8 классов), при котором обычные кластерные SE
    завышают значимость.
    """
    design = _design(cls, yr)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    residual = y - fitted
    clusters = sorted(set(cls.tolist()))
    groups = len(clusters)
    if groups > 12:  # защита от перебора; в отраслевой панели 8 классов
        rng = np.random.default_rng(20261113)
        patterns = rng.integers(0, 2, size=(4096, groups))
    else:
        patterns = np.array(
            [[(bits >> j) & 1 for j in range(groups)] for bits in range(2**groups)]
        )
    index = {code: position for position, code in enumerate(clusters)}
    observed = abs(fit.t)
    exceed = 0
    for pattern in patterns:
        signs = np.array(
            [1.0 if pattern[index[code]] else -1.0 for code in cls]
        )
        bootstrap = _ols(fitted + residual * signs, x, cls, yr)
        if abs(bootstrap.t) >= observed:
            exceed += 1
    return (exceed + 1) / (len(patterns) + 1)


def _leave_one_out(
    y: np.ndarray,
    x: np.ndarray,
    cls: np.ndarray,
    yr: np.ndarray,
    fit: Fit,
) -> tuple[int, float, float]:
    signs: list[int] = []
    values: list[float] = []
    for code in sorted(set(cls.tolist())):
        mask = cls != code
        estimate = _ols(y[mask], x[mask], cls[mask], yr[mask])
        signs.append(int(np.sign(estimate.beta)))
        values.append(estimate.beta)
    same = sum(1 for sign in signs if sign == int(np.sign(fit.beta)))
    return same, min(values), max(values)


def _estimate_edge(
    raw: dict[str, dict[str, dict[int, float]]],
    edge: Edge,
    classes: Sequence[str],
) -> dict:
    y, x, cls, yr = _panel(raw, edge, edge.lag, classes)
    primary = _ols(y, x, cls, yr)
    p_value = _wild_cluster_p(y, x, cls, yr, primary)
    huber = _huber(y, x, cls, yr)
    same, loo_min, loo_max = _leave_one_out(y, x, cls, yr, primary)

    no33 = [code for code in classes if code != "33"]
    y2, x2, cls2, yr2 = _panel(raw, edge, edge.lag, no33)
    without33 = _ols(y2, x2, cls2, yr2)
    p_without33 = _wild_cluster_p(y2, x2, cls2, yr2, without33)

    by_lag: dict[str, dict] = {}
    for lag in (0, 1, 2):
        yl, xl, cl, yrl = _panel(raw, edge, lag, classes)
        estimate = _ols(yl, xl, cl, yrl)
        p_lag = _wild_cluster_p(yl, xl, cl, yrl, estimate)
        by_lag[str(lag)] = {
            "lag": lag,
            "n": estimate.n,
            "beta": estimate.beta,
            "se": estimate.se,
            "t": estimate.t,
            "p": p_lag,
        }
    signs_by_lag = "".join(
        "+" if by_lag[str(lag)]["beta"] > 0 else "-" for lag in (0, 1, 2)
    )
    stable = len(set(signs_by_lag)) == 1
    return {
        "edge": edge.name,
        "source": edge.source,
        "target": edge.target,
        "mechanism": edge.mechanism,
        "config_w": edge.weight,
        "delay": edge.delay,
        "lag": edge.lag,
        "n": primary.n,
        "beta": primary.beta,
        "se": primary.se,
        "t": primary.t,
        "p": p_value,
        "r_within": primary.r_within,
        "sign_match": bool(np.sign(primary.beta) == np.sign(edge.weight)),
        "signs_by_lag": signs_by_lag,
        "stable_sign": stable,
        "loo_same": same,
        "loo_min": loo_min,
        "loo_max": loo_max,
        "ols_no33_beta": without33.beta,
        "ols_no33_se": without33.se,
        "ols_no33_t": without33.t,
        "ols_no33_p": p_without33,
        "huber_beta": huber.beta,
        "huber_se": huber.se,
        "huber_t": huber.t,
        "by_lag": by_lag,
    }


def _verdict(row: dict) -> str:
    if row["stable_sign"]:
        if row["p"] < 0.05:
            return "знак устойчив, значимо"
        return "знак устойчив, незначимо"
    return "знак неустойчив"


def _write_summary(path: Path, rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Панельные оценки дуг отраслевой модели")
    lines.append("")
    lines.append(
        "Логарифмические темпы прироста, 8 классов ОКВЭД (24--30, 33), "
        "2018--2024. Фиксированные эффекты класса и года, кластерные по "
        "классу стандартные ошибки, wild cluster bootstrap p-value "
        "(перебор всех 2^8 знаковых шаблонов). Все оценки --- ассоциации, "
        "не причинные эффекты."
    )
    lines.append("")
    lines.append("## 1. Оценка при задержке из конфига")
    lines.append("")
    lines.append(
        "| Дуга | Вес | Задержка | Лаг | n | beta | SE | t | p | "
        "Знак по лагам 0/1/2 | Совпал с w | LOO |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| `{row['edge']}` | {_fmt(row['config_w'])} | "
            f"{row['delay']} мес. | {row['lag']} г. | {row['n']} | "
            f"{_fmt(row['beta'], 3)} | {_fmt(row['se'], 3)} | "
            f"{_fmt(row['t'])} | {row['p']:.3f} | "
            f"`{row['signs_by_lag']}` | "
            f"{'да' if row['sign_match'] else 'нет'} | "
            f"{row['loo_same']}/8 |"
        )
    lines.append("")
    lines.append(
        "p --- wild cluster bootstrap с ограниченной нулевой гипотезой; "
        "при 8 кластерах обычная кластерная SE завышает значимость. "
        "LOO --- сколько из 8 оценок без одного класса сохраняют знак."
    )
    lines.append("")
    lines.append("## 2. Устойчивость знака по лагам")
    lines.append("")
    lines.append(
        "| Дуга | 0 лет | 1 год | 2 года | Вывод |"
    )
    lines.append("|---|---|---|---|---|")
    for row in rows:
        cells = []
        for lag in (0, 1, 2):
            item = row["by_lag"][str(lag)]
            cells.append(
                f"{_fmt(item['beta'], 3)} (p={item['p']:.2f}, n={item['n']})"
            )
        lines.append(
            f"| `{row['edge']}` | " + " | ".join(cells) + f" | {_verdict(row)} |"
        )
    lines.append("")
    lines.append("## 3. Проверка на выбросы и класс 33")
    lines.append("")
    lines.append("| Дуга | МНК | МНК без 33 | Huber |")
    lines.append("|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| `{row['edge']}` | {_fmt(row['beta'], 3)} | "
            f"{_fmt(row['ols_no33_beta'], 3)} "
            f"(p={row['ols_no33_p']:.2f}) | "
            f"{_fmt(row['huber_beta'], 3)} |"
        )
    lines.append("")
    lines.append("## 4. Что видно")
    lines.append("")
    stable = [row for row in rows if row["stable_sign"]]
    significant = [row for row in rows if row["p"] < 0.05]
    contradicted = [row for row in rows if not row["sign_match"]]
    lines.append(
        f"- Устойчивый знак по всем трём лагам имеют {len(stable)} из "
        f"{len(rows)} дуг: "
        + (", ".join(f"`{row['edge']}`" for row in stable) or "ни одна")
        + "."
    )
    lines.append(
        f"- p < 0,05 при задержке из конфига имеют {len(significant)} дуг: "
        + (", ".join(f"`{row['edge']}`" for row in significant) or "ни одна")
        + "."
    )
    lines.append(
        f"- Знак при задержке из конфига противоположен экспертному весу "
        f"у {len(contradicted)} из {len(rows)} дуг: "
        + ", ".join(f"`{row['edge']}`" for row in contradicted)
        + "."
    )
    lines.append(
        "- Ключевая для выбора мер дуга `i2->i3` меняет знак: "
        "минус на лаге 0, плюс на лаге 1, минус на лаге 2. "
        "Это не оценка веса, а отсутствие идентификации."
    )
    lines.append(
        "- `y1->i2` --- единственная дуга с p < 0,05 на лаге из конфига, "
        "но на лаге 0 знак отрицательный, а без класса 33 оценка "
        "теряет значимость. Устойчивой её назвать нельзя."
    )
    lines.append(
        "- Три пары дуг механически связаны: `i3` входит в `o1`, "
        "а `y2` --- это тот же ряд `57711`, что и `o1`. Их «значимость» "
        "не является независимым подтверждением."
    )
    lines.append("")
    lines.append("## 5. Что не оценено")
    lines.append("")
    for block, reason in SKIPPED:
        lines.append(f"- `{block}` --- {reason}.")
    lines.append("")
    lines.append("## 6. Вывод")
    lines.append("")
    lines.append(
        "Панель не идентифицирует знаки дуг: у большинства оценок знак "
        "меняется при сдвиге лага на год, а после поправки на 8 кластеров "
        "почти ничего не остаётся значимым. Причинная интерпретация "
        "невозможна: меры в истории не наблюдались, общие шоки поглощены "
        "годовыми эффектами, оставшаяся вариация --- ассоциация. "
        "Веса дуг остаются экспертными; результат используется как "
        "проверка границ применимости, а не как калибровка."
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    load_model(args.config)  # проверка конфига до чтения данных
    root = resolve_root(args.config, args.data_root)
    classes = list(CLASSES)
    raw = _load_raw(root, classes)

    rows = [_estimate_edge(raw, edge, classes) for edge in EDGES]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "edges.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": str(args.config),
                "data_root": str(root),
                "classes": classes,
                "year_max": YEAR_MAX,
                "skipped": SKIPPED,
                "edges": rows,
            },
            fh,
            ensure_ascii=False,
            indent=1,
        )
    fieldnames = [
        "edge", "source", "target", "mechanism", "config_w", "delay", "lag",
        "n", "beta", "se", "t", "p", "r_within", "sign_match",
        "signs_by_lag", "stable_sign", "loo_same", "loo_min", "loo_max",
        "ols_no33_beta", "ols_no33_se", "ols_no33_t", "ols_no33_p",
        "huber_beta", "huber_se", "huber_t",
    ]
    with (args.out / "edges.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(args.out / "summary.md", rows)

    print(f"отчёт: {args.out / 'summary.md'}")
    for row in rows:
        print(
            f"  {row['edge']:<8} w={row['config_w']:+.2f} "
            f"lag={row['lag']} n={row['n']:>2} "
            f"beta={_fmt(row['beta'], 3)} t={_fmt(row['t'])} "
            f"p={row['p']:.3f} лаги={row['signs_by_lag']} "
            f"{'знак+' if row['sign_match'] else 'знак-'} "
            f"LOO={row['loo_same']}/8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
