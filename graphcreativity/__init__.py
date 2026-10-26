"""Симулятор графовой модели повышения креативности предприятия."""

from .engine import SimOptions, SimulationResult, simulate
from .metrics import (
    compare_named_plans,
    compare_plans,
    critical_combinations,
    critical_event_combinations,
    critical_removal_sets,
    effect_strength,
    minimal_sufficient_sets,
    pareto_front,
    robustness_radius,
    scenario_robustness,
    summarize,
)
from .model import Model, load_model

__all__ = [
    "Model",
    "SimOptions",
    "SimulationResult",
    "compare_named_plans",
    "compare_plans",
    "critical_combinations",
    "critical_event_combinations",
    "critical_removal_sets",
    "effect_strength",
    "load_model",
    "minimal_sufficient_sets",
    "pareto_front",
    "robustness_radius",
    "scenario_robustness",
    "simulate",
    "summarize",
]

__version__ = "0.2.0"
