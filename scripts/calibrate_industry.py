#!/usr/bin/env python3
"""Первичная калибровка отраслевой модели по панелям ЕМИСС.

Что делает:

1. читает секцию ``data`` из конфига отраслевой модели;
2. собирает ряды «класс ОКВЭД x год» по страновому уровню;
3. считает базу и цель для узлов, у которых есть естественная единица
   (``i1`` в процентах, ``i2`` в процентах выручки);
4. считает наблюдаемую отдачу ``k_obs = 58761 / 60020``;
5. проверяет согласованность основных и альтернативных рядов;
6. сверяет масштаб сценарных шоков с историей инвестиций и выпуска;
7. пишет ``panel.csv``, ``calibration.json`` и ``summary.md``.

Веса дуг здесь не оцениваются: панель 8 классов x 7 лет не
идентифицирует причинный эффект мер. Скрипт калибрует уровни,
критерий и масштаб сценариев, а не причинные коэффициенты.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

# Прямой запуск (./scripts/calibrate_industry.py) не кладёт корень
# репозитория в sys.path: добавляем его до импорта пакета.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphcreativity.data import (
    Series,
    load_series_set,
    pearson,
    resolve_root,
    yoy_growth,
)
from graphcreativity.model import load_model

DEFAULT_CONFIG = ROOT / "configs" / "industry.yaml"
DEFAULT_OUT = ROOT / "out" / "industry-calibration"


@dataclass
class Loaded:
    alias: str
    indicator: str
    role: str
    series: dict[str, Series]


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "---"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:.{digits}f}".replace(".", ",")


_UNITS = {
    "процент": "%",
    "% выручки": "%",
    "тысяча рублей": "тыс. руб.",
    "миллион рублей": "млн руб.",
    "рубль": "руб.",
    "единица": "ед.",
    "человек": "чел.",
    "тысяча человек": "тыс. чел.",
}


def _unit(value: str) -> str:
    return _UNITS.get(value, value)


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "---"
    return f"{value * 100:+.{digits}f} %".replace(".", ",")


def _load_all(root: Path, spec: dict, okveds: Sequence[str]) -> dict[str, Loaded]:
    loaded: dict[str, Loaded] = {}
    for alias, item in (spec.get("nodes") or {}).items():
        options = {
            key: item[key]
            for key in ("level", "kind", "period")
            if key in item
        }
        load_kwargs = {"level": spec.get("level", "country")}
        load_kwargs.update(options)
        loaded[alias] = Loaded(
            alias=alias,
            indicator=str(item["id"]),
            role=str(item.get("role", "primary")),
            series=load_series_set(root, str(item["id"]), okveds, **load_kwargs),
        )

    # Производный ряд: интенсивность затрат как 60020 / 57711, %.
    # Нужен, чтобы сверить 60022 с тем отношением, которое стояло
    # в модели до подключения данных.
    if "i2_abs" in loaded and "o1" in loaded:
        derived: dict[str, Series] = {}
        for code in okveds:
            cost = loaded["i2_abs"].series[code]
            output = loaded["o1"].series[code]
            values = {
                year: 100.0 * cost.values[year] / output.values[year]
                for year in cost.values
                if output.values.get(year)
            }
            derived[code] = Series(
                indicator="60020/57711",
                okved=code,
                level=cost.level,
                values=values,
                unit="% выручки",
            )
        loaded["i2_implied"] = Loaded(
            alias="i2_implied",
            indicator="60020/57711",
            role="derived",
            series=derived,
        )
    return loaded


def _write_panel(path: Path, loaded: dict[str, Loaded]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["alias", "id", "role", "okved", "year", "value", "unit"])
        for item in loaded.values():
            for code, series in sorted(item.series.items()):
                for year in series.years():
                    writer.writerow(
                        [
                            item.alias,
                            item.indicator,
                            item.role,
                            code,
                            year,
                            series.values[year],
                            series.unit,
                        ]
                    )


def _base_target(
    loaded: dict[str, Loaded],
    model,
    okveds: Sequence[str],
    pilot: str,
    base_years: Sequence[int],
    growth: float,
) -> dict:
    """База и цель для узлов, у которых единица совпадает с моделью."""
    result: dict = {}
    for alias in ("i1", "i2", "i3"):
        item = loaded.get(alias)
        if item is None:
            continue
        node = model.node(alias)
        pilot_series = item.series[pilot]
        base_pilot = pilot_series.mean(base_years)
        if base_pilot is None:
            continue
        bases = {
            code: series.mean(base_years)
            for code, series in item.series.items()
            if series.mean(base_years) is not None
        }
        entry = {
            "id": item.indicator,
            "unit": pilot_series.unit,
            "base_pilot": base_pilot,
            "target_pilot": base_pilot * (1.0 + growth),
            "base_min": min(bases.values()),
            "base_max": max(bases.values()),
            "base_by_class": bases,
            "model_base": node.base,
            "model_target": node.target,
            "applied": alias in ("i1", "i2"),
        }
        if alias == "i3":
            # тыс. руб. -> млн руб. для читаемости отчёта
            base_pilot_mln = base_pilot / 1000.0
            entry["base_pilot_mln"] = base_pilot_mln
            entry["target_pilot_mln"] = base_pilot * (1.0 + growth) / 1000.0
            entry["base_min_mln"] = min(bases.values()) / 1000.0
            entry["base_max_mln"] = max(bases.values()) / 1000.0
            entry["scale_to_model_base"] = (
                base_pilot_mln / node.base if node.base else None
            )
        result[alias] = entry
    return result


def _k_obs(
    loaded: dict[str, Loaded],
    okveds: Sequence[str],
    pilot: str,
) -> dict:
    """k_obs = 58761 / 60020 по годам, тыс. руб. / тыс. руб."""
    output = loaded["i3"].series
    cost = loaded["i2_abs"].series
    by_class: dict[str, dict[int, float]] = {}
    for code in okveds:
        out = output[code].values
        cst = cost[code].values
        by_class[code] = {
            year: out[year] / cst[year]
            for year in out
            if cst.get(year)
        }
    pilot_values = by_class[pilot]
    years = sorted(pilot_values)
    stable = {y: v for y, v in pilot_values.items() if y <= 2024}
    return {
        "by_class": by_class,
        "pilot": pilot,
        "pilot_values": pilot_values,
        "years": years,
        "mean": mean(pilot_values.values()) if pilot_values else None,
        "mean_to_2024": mean(stable.values()) if stable else None,
        "min": min(pilot_values.values()) if pilot_values else None,
        "max": max(pilot_values.values()) if pilot_values else None,
    }


def _growth_pairs(
    loaded: dict[str, Loaded],
    okveds: Sequence[str],
    left: str,
    right: str,
) -> dict:
    """Сравнить два ряда: уровни (если единицы совпадают) и темпы."""
    left_item = loaded[left]
    right_item = loaded[right]
    left_unit = _unit(left_item.series[okveds[0]].unit)
    right_unit = _unit(right_item.series[okveds[0]].unit)
    same_unit = left_unit == right_unit

    level_pairs: list[tuple[float, float]] = []
    growth_pairs: list[tuple[float, float]] = []
    ratios: list[float] = []
    pilot_level_pairs: list[tuple[float, float]] = []
    pilot_growth_pairs: list[tuple[float, float]] = []

    for code in okveds:
        a = left_item.series[code]
        b = right_item.series[code]
        common = sorted(set(a.values) & set(b.values))
        if same_unit and len(common) >= 3:
            xs = [a.values[y] for y in common]
            ys = [b.values[y] for y in common]
            level_pairs.extend(zip(xs, ys))
            if code == "28":
                pilot_level_pairs.extend(zip(xs, ys))
            ratios.extend(x / y for x, y in zip(xs, ys) if y)
        ga = yoy_growth(a, common)
        gb = yoy_growth(b, common)
        years = sorted(set(ga) & set(gb))
        pairs = [(ga[y], gb[y]) for y in years]
        growth_pairs.extend(pairs)
        if code == "28":
            pilot_growth_pairs.extend(pairs)

    return {
        "left": left,
        "right": right,
        "same_unit": same_unit,
        "n_levels": len(level_pairs),
        "n_growth": len(growth_pairs),
        "r_levels": pearson(*zip(*level_pairs)) if level_pairs else None,
        "r_growth": pearson(*zip(*growth_pairs)) if growth_pairs else None,
        "r_levels_pilot": pearson(*zip(*pilot_level_pairs)) if pilot_level_pairs else None,
        "r_growth_pilot": pearson(*zip(*pilot_growth_pairs)) if pilot_growth_pairs else None,
        "ratio_min": min(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
        "ratio_mean": mean(ratios) if ratios else None,
        "left_unit": left_unit,
        "right_unit": right_unit,
    }


def _verdict(pair: dict) -> str:
    if pair["n_growth"] < 4:
        return "мало наблюдений"
    r = pair["r_growth"]
    if r is None:
        return "нет вариации"
    ratio = pair["ratio_mean"]
    spread = (
        pair["ratio_max"] / pair["ratio_min"]
        if pair["ratio_min"] not in (None, 0) and pair["ratio_max"] is not None
        else None
    )
    levels_differ = pair["same_unit"] and ratio is not None and (
        not 0.9 <= ratio <= 1.1 or (spread is not None and spread > 1.3)
    )
    if levels_differ:
        if r >= 0.7:
            return "уровни расходятся, темпы согласованы"
        if r >= 0.4:
            return "уровни расходятся, темпы слабо согласованы"
        return "не сопоставим"
    if r >= 0.7:
        return "согласован"
    if r >= 0.4:
        return "слабо согласован"
    return "расходится"


def _scenario_scale(loaded: dict[str, Loaded], okveds: Sequence[str], pilot: str) -> dict:
    """Исторические падения/росты, с которыми сравниваются шоки."""
    result: dict = {}
    for alias, label in (("y1_long", "инвестиции"), ("o1", "выпуск")):
        item = loaded.get(alias)
        if item is None:
            continue
        by_class = {}
        for code in okveds:
            growth = yoy_growth(item.series[code])
            if not growth:
                continue
            by_class[code] = {
                "min": min(growth.values()),
                "min_year": min(growth, key=growth.get),
                "max": max(growth.values()),
                "max_year": max(growth, key=growth.get),
            }
        result[alias] = {
            "label": label,
            "id": item.indicator,
            "by_class": by_class,
            "pilot": by_class.get(pilot),
        }
    return result


def _scenario_check(scale: dict, scenarios: dict) -> dict:
    """Сопоставить худшее историческое падение с шоком в конфиге."""
    checks: dict = {}
    mapping = {"y1": "y1_long", "y2": "o1"}
    for env, alias in mapping.items():
        item = scale.get(alias)
        if item is None or item.get("pilot") is None:
            continue
        shock = None
        for scenario in scenarios.values():
            values = getattr(scenario, "values", {}) or {}
            if env in values:
                value = float(values[env])
                shock = value if shock is None else min(shock, value)
        checks[env] = {
            "alias": alias,
            "history_min": item["pilot"]["min"],
            "history_min_year": item["pilot"]["min_year"],
            "history_max": item["pilot"]["max"],
            "history_max_year": item["pilot"]["max_year"],
            "scenario_shock": shock,
        }
    return checks


def _write_summary(path: Path, report: dict) -> None:
    base = report["base_target"]
    kobs = report["k_obs"]
    pairs = report["pairs"]
    checks = report["scenario_check"]

    lines: list[str] = []
    lines.append("# Первичная калибровка отраслевой модели по ЕМИСС")
    lines.append("")
    lines.append(
        f"Конфиг: `{report['config']}`. Корень данных: `{report['data_root']}`. "
        f"Пилот: ОКВЭД {report['pilot']}. Классы: {', '.join(report['okveds'])}."
    )
    lines.append("")
    lines.append(
        "Скрипт калибрует уровни, критерий и масштаб сценариев. "
        "Веса дуг остаются экспертными: панель не идентифицирует "
        "причинный эффект мер."
    )
    lines.append("")

    lines.append("## 1. База и цель")
    lines.append("")
    lines.append(
        "| Узел | Показатель | База (среднее 2019--2022) | Диапазон по классам | "
        "В модели | Предложение |"
    )
    lines.append("|---|---|---|---|---|---|")
    for alias, entry in base.items():
        if alias == "i3":
            base_text = f"{_fmt(entry['base_pilot_mln'])} млн руб."
            range_text = f"{_fmt(entry['base_min_mln'])}--{_fmt(entry['base_max_mln'])}"
        else:
            base_text = f"{_fmt(entry['base_pilot'])} {_unit(entry['unit'])}"
            range_text = f"{_fmt(entry['base_min'])}--{_fmt(entry['base_max'])}"
        proposal = (
            f"base {_fmt(entry['base_pilot'])}, target {_fmt(entry['target_pilot'])}"
            if entry["applied"]
            else "не менять (условный масштаб)"
        )
        lines.append(
            f"| `{alias}` | `{entry['id']}` | {base_text} | "
            f"{range_text} | "
            f"{_fmt(entry['model_base'], 0)} -> {_fmt(entry['model_target'], 0)} | "
            f"{proposal} |"
        )
    if "i3" in base:
        entry = base["i3"]
        lines.append("")
        lines.append(
            f"Для `i3` масштаб модели условный: фактическая база "
            f"{_fmt(entry['base_pilot_mln'])} млн руб. при базе модели "
            f"{_fmt(entry['model_base'], 0)} млн руб. --- масштаб "
            f"{_fmt(entry['scale_to_model_base'], 0)}x. "
            "Поэтому `k` сопоставим между планами, но не является рублёвой "
            "отдачей по отрасли."
        )
    lines.append("")

    lines.append("## 2. Наблюдаемая отдача `k_obs = 58761 / 60020`")
    lines.append("")
    lines.append("| Год | 58761, млн руб. | 60020, млн руб. | k_obs |")
    lines.append("|---|---|---|---|")
    output = report["raw"]["i3"][report["pilot"]]
    cost = report["raw"]["i2_abs"][report["pilot"]]
    for year in kobs["years"]:
        lines.append(
            f"| {year} | {_fmt(output.get(year, 0) / 1000.0, 0)} | "
            f"{_fmt(cost.get(year, 0) / 1000.0, 0)} | "
            f"{_fmt(kobs['pilot_values'][year])} |"
        )
    lines.append("")
    lines.append(
        f"Среднее за 2019--2024: {_fmt(kobs['mean_to_2024'])}; "
        f"за весь ряд 2019--2025: {_fmt(kobs['mean'])}, "
        f"диапазон {_fmt(kobs['min'])}--{_fmt(kobs['max'])}. "
        f"По всем классам среднее (до 2023): "
        f"{_fmt(report['k_obs_all_classes_mean'])}."
    )
    if report.get("k_obs_break"):
        lines.append("")
        lines.append(report["k_obs_break"])
    lines.append("")

    lines.append("## 3. Согласованность рядов")
    lines.append("")
    lines.append(
        "| Пара | Единицы | n темпов | r (уровни) | r (темпы) | "
        "Отношение уровней | Вывод |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for pair in pairs:
        ratio = (
            f"{_fmt(pair['ratio_min'])}--{_fmt(pair['ratio_max'])}"
            if pair["ratio_min"] is not None
            else "---"
        )
        lines.append(
            f"| `{pair['left']}` / `{pair['right']}` | "
            f"{pair['left_unit']} / {pair['right_unit']} | {pair['n_growth']} | "
            f"{_fmt(pair['r_levels'])} | {_fmt(pair['r_growth'])} | {ratio} | "
            f"{pair['verdict']} |"
        )
    lines.append("")
    lines.append(
        "Корреляция темпов на 6--7 точках --- ориентир, а не оценка. "
        "Она отвечает на вопрос «двигаются ли ряды вместе», не «какой ряд верен»."
    )
    for pair in pairs:
        if pair["left"] == "i2" and pair["right"] == "i2_implied":
            lines.append("")
            lines.append(
                f"`60022` в среднем на {_fmt((pair['ratio_mean'] - 1) * 100, 0)} % "
                f"выше отношения `60020/57711`, но разброс большой "
                f"({_fmt(pair['ratio_min'])}--{_fmt(pair['ratio_max'])} по классам "
                "и годам), а темпы совпадают. Для уровня нужен `60022`; "
                "отношение годится только как проверка динамики."
            )
        if pair["left"] == "o1" and pair["right"] == "o1_alt":
            lines.append("")
            lines.append(
                f"`57715` отличается от `57711` по уровню в "
                f"{_fmt(pair['ratio_min'])}--{_fmt(pair['ratio_max'])} раза "
                f"и по темпам (r = {_fmt(pair['r_growth'])}). Это другой "
                "контур, а не продолжение: только проверка устойчивости."
            )
    lines.append("")

    lines.append("## 4. Масштаб сценариев")
    lines.append("")
    lines.append("| Канал | Исторический минимум | Год | Максимум | Год | Шок в конфиге |")
    lines.append("|---|---|---|---|---|---|")
    for env, check in checks.items():
        lines.append(
            f"| `{env}` | {_fmt_pct(check['history_min'])} | {check['history_min_year']} | "
            f"{_fmt_pct(check['history_max'])} | {check['history_max_year']} | "
            f"{_fmt_pct(check['scenario_shock'])} |"
        )
    lines.append("")
    lines.append(
        "Шоки в конфиге --- стресс-сценарии, а не повторение истории. "
        "Исторический минимум по номинальному ряду задаёт нижнюю границу: "
        "всё, что глубже, --- проверка на прочность, а не прогноз."
    )
    lines.append("")

    lines.append("## 5. Что меняется в модели")
    lines.append("")
    lines.append("1. `i1` и `i2` получают базу и цель из ЕМИСС (см. таблицу 1).")
    lines.append(
        "2. `i3` остаётся в условном масштабе; фактический ряд используется "
        "для `k_obs` и проверки знаков, а не для пересчёта `R`."
    )
    lines.append(
        "3. Альтернативные ряды не подключаются как основные: они дают "
        "проверку устойчивости, а не замену."
    )
    lines.append(
        "4. Веса, задержки и пороги остаются класса Э. Для их оценки нужна "
        "вариация политики, которой в панели нет."
    )
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    config: Path,
    data_root: Path | None,
    out: Path,
    pilot: str | None,
) -> dict:
    model = load_model(config)
    spec = model.raw.get("data") or {}
    if not spec:
        raise ValueError(f"в конфиге {config} нет секции data")

    root = resolve_root(config, data_root)
    okveds = [str(code) for code in spec.get("okved") or []]
    if not okveds:
        raise ValueError("в секции data не задан список okved")
    pilot = str(pilot or spec.get("pilot_okved") or okveds[0])
    bounds = [int(y) for y in spec.get("base_years") or [2019, 2022]]
    base_years = list(range(min(bounds), max(bounds) + 1))
    growth = float(spec.get("target_growth", 0.25))

    loaded = _load_all(root, spec, okveds)

    out.mkdir(parents=True, exist_ok=True)
    _write_panel(out / "panel.csv", loaded)

    base_target = _base_target(loaded, model, okveds, pilot, base_years, growth)
    kobs = _k_obs(loaded, okveds, pilot)
    kobs_all = mean(
        value
        for values in kobs["by_class"].values()
        for year, value in values.items()
        if year <= 2023
    )

    pairs = []
    for left, right in spec.get("pairs") or []:
        pair = _growth_pairs(loaded, okveds, str(left), str(right))
        pair["verdict"] = _verdict(pair)
        pairs.append(pair)

    scale = _scenario_scale(loaded, okveds, pilot)
    checks = _scenario_check(scale, model.scenarios)

    # Разрыв 2025 года в доле инновационной продукции: если доля
    # удваивается, ряд 2025 нельзя использовать в калибровке.
    k_obs_break = None
    share = loaded.get("i3_share")
    if share is not None:
        values = share.series[pilot].values
        if 2024 in values and 2025 in values and values[2024]:
            jump = values[2025] / values[2024] - 1.0
            if abs(jump) > 0.5:
                k_obs_break = (
                    f"Внимание: доля инновационной продукции (`58762`) в 2025 г. "
                    f"составляет {_fmt(values[2025])} % против {_fmt(values[2024])} % "
                    f"в 2024 г. ({_fmt_pct(jump)}). Похоже на смену методологии "
                    "или неполный круг; 2025 г. в калибровку не включать."
                )

    report = {
        "config": str(config),
        "data_root": str(root),
        "pilot": pilot,
        "okveds": okveds,
        "base_years": base_years,
        "target_growth": growth,
        "base_target": base_target,
        "k_obs": kobs,
        "k_obs_all_classes_mean": kobs_all,
        "k_obs_break": k_obs_break,
        "pairs": pairs,
        "scenario_scale": scale,
        "scenario_check": checks,
        "raw": {
            alias: {
                code: {year: series.values[year] for year in series.years()}
                for code, series in item.series.items()
            }
            for alias, item in loaded.items()
        },
    }

    with (out / "calibration.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=str)

    _write_summary(out / "summary.md", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot", default=None)
    args = parser.parse_args(argv)

    report = build_report(args.config, args.data_root, args.out, args.pilot)
    print(f"панель:   {args.out / 'panel.csv'}")
    print(f"отчёт:    {args.out / 'summary.md'}")
    print(f"пилот {report['pilot']}: k_obs 2019--2024 "
          f"{_fmt(report['k_obs']['mean_to_2024'])}, "
          f"диапазон {_fmt(report['k_obs']['min'])}--{_fmt(report['k_obs']['max'])}")
    for pair in report["pairs"]:
        print(
            f"  {pair['left']}/{pair['right']}: r(темпы)={_fmt(pair['r_growth'])}, "
            f"n={pair['n_growth']}, {pair['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
