"""Reproduce the numerical results for the BIOMATH major revision.

Run from this directory with::

    python reproduce_figures.py

The script writes publication-resolution figures and machine-readable CSV
summaries to ``figures/``.  Parameters are illustrative, not field estimates.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp


OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

PARAMETER_ORDER = (
    "Lambda", "beta", "phi", "theta", "eta", "rho",
    "k", "mu", "d", "xi", "delta",
)

BASELINE = {
    "Lambda": 10.0,
    "beta": 0.02,
    "phi": 0.01,
    "theta": 0.2,
    "eta": 0.3,
    "rho": 0.5,
    "k": 0.4,
    "mu": 0.3,
    "d": 0.1,
    "xi": 0.6,
    "delta": 0.3,
}

DFE_SCENARIO = {
    **BASELINE,
    "beta": 0.005,
    "phi": 0.001,
    "mu": 0.9,
    "delta": 1.0,
}

INITIAL_EE = np.array([30.0, 10.0, 5.0, 3.0, 2.0])
# N(0)=90 <= Lambda/d=100 and P(0)=50 <= xi Lambda/(d delta)=60.
INITIAL_DFE = np.array([40.0, 20.0, 20.0, 10.0, 50.0])


def gamma(p: Mapping[str, float]) -> float:
    return p["beta"] + p["phi"] * p["xi"] / p["delta"]


def r0(p: Mapping[str, float]) -> float:
    numerator = p["k"] * p["Lambda"] * (
        p["theta"] + p["d"] + p["rho"] * p["eta"]
    )
    denominator = (
        p["d"]
        * (p["k"] + p["d"])
        * (p["mu"] + p["d"])
        * (p["eta"] + p["theta"] + p["d"])
    )
    return numerator / denominator * gamma(p)


def rhs(_time: float, state: Sequence[float], p: Mapping[str, float]) -> list[float]:
    s, st, exposed, infectious, pathogen = state
    incidence_s = p["beta"] * s * infectious + p["phi"] * s * pathogen
    incidence_st = p["rho"] * (
        p["beta"] * st * infectious + p["phi"] * st * pathogen
    )
    return [
        p["Lambda"] + p["theta"] * st - incidence_s - (p["eta"] + p["d"]) * s,
        p["eta"] * s - incidence_st - (p["theta"] + p["d"]) * st,
        incidence_s + incidence_st - (p["k"] + p["d"]) * exposed,
        p["k"] * exposed - (p["mu"] + p["d"]) * infectious,
        p["xi"] * infectious - p["delta"] * pathogen,
    ]


def integrate(
    p: Mapping[str, float], initial: np.ndarray, horizon: float, points: int
) -> object:
    time = np.linspace(0.0, horizon, points)
    solution = solve_ivp(
        rhs,
        (0.0, horizon),
        initial,
        args=(p,),
        method="Radau",
        t_eval=time,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return solution


def disease_free_equilibrium(p: Mapping[str, float]) -> np.ndarray:
    denominator = p["d"] * (p["eta"] + p["theta"] + p["d"])
    return np.array([
        p["Lambda"] * (p["theta"] + p["d"]) / denominator,
        p["Lambda"] * p["eta"] / denominator,
        0.0,
        0.0,
        0.0,
    ])


def equilibrium_coefficients(p: Mapping[str, float]) -> tuple[float, float, float]:
    route = gamma(p)
    scale = (p["k"] + p["d"]) * (p["mu"] + p["d"]) / p["k"]
    a2 = scale * p["rho"] * route
    a1 = scale * (
        p["theta"] + p["d"] + p["rho"] * p["eta"] + p["rho"] * p["d"]
    ) - p["Lambda"] * p["rho"] * route
    a0 = (
        p["d"]
        * scale
        * (p["eta"] + p["theta"] + p["d"])
        / route
        * (1.0 - r0(p))
    )
    return a2, a1, a0


def endemic_infectious(p: Mapping[str, float], zero_below_threshold: bool = False) -> float:
    if r0(p) <= 1.0:
        if zero_below_threshold:
            return 0.0
        raise ValueError("The positive endemic equilibrium requires R0 > 1.")
    a2, a1, a0 = equilibrium_coefficients(p)
    discriminant = a1 * a1 - 4.0 * a2 * a0
    if discriminant <= 0.0:
        raise ArithmeticError("Expected a strictly positive quadratic discriminant.")
    return (-a1 + np.sqrt(discriminant)) / (2.0 * a2)


def endemic_equilibrium(p: Mapping[str, float]) -> np.ndarray:
    infectious = endemic_infectious(p)
    route = gamma(p)
    pathogen = p["xi"] * infectious / p["delta"]
    exposed = (p["mu"] + p["d"]) * infectious / p["k"]
    denominator = (
        (route * infectious + p["eta"] + p["d"])
        * (p["rho"] * route * infectious + p["theta"] + p["d"])
        / p["eta"]
        - p["theta"]
    )
    protected = p["Lambda"] / denominator
    susceptible = (
        p["rho"] * route * infectious + p["theta"] + p["d"]
    ) * protected / p["eta"]
    return np.array([susceptible, protected, exposed, infectious, pathogen])


def save(figure: plt.Figure, filename: str) -> None:
    figure.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_rows(filename: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with (OUTPUT_DIR / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def plot_stability_scenarios() -> None:
    scenarios = (
        (DFE_SCENARIO, INITIAL_DFE, 200.0, "Disease-free regime", disease_free_equilibrium(DFE_SCENARIO)),
        (BASELINE, INITIAL_EE, 300.0, "Endemic regime", endemic_equilibrium(BASELINE)),
    )
    colours = ("teal", "darkorange", "purple", "crimson")
    labels = (r"$S(t)$", r"$S_T(t)$", r"$E(t)$", r"$I(t)$")
    styles = ("-", "--", "-.", "-")
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.1), sharex="col")
    output_rows = []

    for column, (p, initial, horizon, title, equilibrium) in enumerate(scenarios):
        solution = integrate(p, initial, horizon, 1600)
        plant_axis = axes[0, column]
        pathogen_axis = axes[1, column]
        for index in range(4):
            plant_axis.plot(
                solution.t, solution.y[index], label=labels[index],
                color=colours[index], linestyle=styles[index], linewidth=1.5,
            )
            plant_axis.axhline(equilibrium[index], color=colours[index], linestyle=":", linewidth=0.8)
        pathogen_axis.plot(solution.t, solution.y[4], color="saddlebrown", label=r"$P(t)$", linewidth=1.6)
        pathogen_axis.axhline(equilibrium[4], color="saddlebrown", linestyle=":", linewidth=0.9)
        plant_axis.set_title(fr"{title} ($\mathcal{{R}}_0={r0(p):.3f}$)")
        plant_axis.set_ylabel("Plant density")
        pathogen_axis.set_ylabel("Pathogen concentration")
        pathogen_axis.set_xlabel("Time (days)")
        for axis in (plant_axis, pathogen_axis):
            axis.grid(True, linestyle=":", alpha=0.5)
            axis.legend(fontsize=8, loc="best")
        output_rows.append((title, r0(p), *equilibrium))

    figure.tight_layout()
    save(figure, "stability_scenarios.png")
    write_rows(
        "stability_equilibria.csv",
        ("scenario", "R0", "S", "ST", "E", "I", "P"),
        output_rows,
    )


def normalised_sensitivities(
    outcome: Callable[[Mapping[str, float]], float], p: Mapping[str, float]
) -> dict[str, float]:
    base = outcome(p)
    indices = {}
    for name in PARAMETER_ORDER:
        step = 1e-5 * p[name]
        lower = {**p, name: p[name] - step}
        upper = {**p, name: p[name] + step}
        derivative = (outcome(upper) - outcome(lower)) / (2.0 * step)
        indices[name] = derivative * p[name] / base
    return indices


def plot_sensitivities() -> None:
    threshold = normalised_sensitivities(r0, BASELINE)
    burden = normalised_sensitivities(endemic_infectious, BASELINE)
    display = {
        "Lambda": r"$\Lambda$", "beta": r"$\beta$", "phi": r"$\phi$",
        "theta": r"$\theta$", "eta": r"$\eta$", "rho": r"$\rho$",
        "k": r"$k$", "mu": r"$\mu$", "d": r"$d$", "xi": r"$\xi$",
        "delta": r"$\delta$",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 5.3))
    for axis, values, title in (
        (axes[0], threshold, r"(a) Sensitivity of $\mathcal{R}_0$"),
        (axes[1], burden, r"(b) Sensitivity of $I^*$"),
    ):
        ordered = sorted(values, key=lambda name: abs(values[name]))
        numbers = [values[name] for name in ordered]
        colours = ["#b2182b" if value < 0 else "#1b7837" for value in numbers]
        axis.barh([display[name] for name in ordered], numbers, color=colours)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Normalised local sensitivity")
        axis.grid(True, axis="x", linestyle=":", alpha=0.45)
    figure.tight_layout()
    save(figure, "sensitivity_indices.png")
    write_rows(
        "sensitivity_indices.csv",
        ("parameter", "R0_sensitivity", "Istar_sensitivity"),
        [(name, threshold[name], burden[name]) for name in PARAMETER_ORDER],
    )


def plot_identifiability_and_bifurcation() -> None:
    r0_values = np.linspace(0.2, 10.0, 250)
    multiplier = r0(BASELINE) / gamma(BASELINE)
    infectious_values = []
    for value in r0_values:
        route = value / multiplier
        p = {**BASELINE, "beta": 0.5 * route, "phi": 0.5 * route * BASELINE["delta"] / BASELINE["xi"]}
        infectious_values.append(endemic_infectious(p, zero_below_threshold=True))

    route_scenarios = (
        ("mostly local", {**BASELINE, "beta": 0.038, "phi": 0.001}),
        ("mixed", BASELINE),
        ("mostly reservoir", {**BASELINE, "beta": 0.002, "phi": 0.019}),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    axes[0].plot(r0_values, infectious_values, color="navy", linewidth=2.0)
    axes[0].axvline(1.0, color="crimson", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel(r"$\mathcal{R}_0$")
    axes[0].set_ylabel(r"Endemic infectious density $I^*$")
    axes[0].set_title("(c) Forward endemic branch")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    colours = ("#1f77b4", "#2ca02c", "#d62728")
    scenario_rows = []
    for colour, (label, p) in zip(colours, route_scenarios):
        solution = integrate(p, INITIAL_EE, 120.0, 1200)
        axes[1].plot(solution.t, solution.y[3], label=label, color=colour, linewidth=1.6)
        axes[2].plot(solution.t, solution.y[4], label=label, color=colour, linewidth=1.6)
        scenario_rows.append((label, p["beta"], p["phi"], gamma(p), r0(p), endemic_infectious(p)))
    for axis, title, ylabel in (
        (axes[1], "(d) Infectious transient", r"$I(t)$"),
        (axes[2], "(e) Reservoir transient", r"$P(t)$"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Time (days)")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.legend(fontsize=8)
    figure.tight_layout()
    save(figure, "identifiability_bifurcation.png")
    write_rows(
        "same_gamma_scenarios.csv",
        ("scenario", "beta", "phi", "Gamma", "R0", "Istar"),
        scenario_rows,
    )
    write_rows(
        "bifurcation_branch.csv",
        ("R0", "Istar"),
        list(zip(r0_values, infectious_values)),
    )


def combine_diagnostic_figures() -> None:
    """Stack the two diagnostic panels into one full-width manuscript figure."""
    with Image.open(OUTPUT_DIR / "sensitivity_indices.png") as top_source:
        top = top_source.convert("RGB")
    with Image.open(OUTPUT_DIR / "identifiability_bifurcation.png") as bottom_source:
        bottom = bottom_source.convert("RGB")

    target_width = max(top.width, bottom.width)
    if top.width != target_width:
        target_height = round(top.height * target_width / top.width)
        top = top.resize((target_width, target_height), Image.Resampling.LANCZOS)
    if bottom.width != target_width:
        target_height = round(bottom.height * target_width / bottom.width)
        bottom = bottom.resize((target_width, target_height), Image.Resampling.LANCZOS)

    gap = 24
    combined = Image.new("RGB", (target_width, top.height + gap + bottom.height), "white")
    combined.paste(top, (0, 0))
    combined.paste(bottom, (0, top.height + gap))
    combined.save(OUTPUT_DIR / "numerical_diagnostics.png", dpi=(300, 300))


def write_control_sweeps() -> None:
    rows = []
    for name, values in (
        ("eta", (0.05, 0.2, 0.5, 0.9)),
        ("mu", (0.1, 0.3, 0.6, 0.9)),
        ("delta", (0.1, 0.4, 0.8, 1.5)),
    ):
        for value in values:
            p = {**BASELINE, name: value}
            rows.append((name, value, r0(p), endemic_infectious(p), *endemic_equilibrium(p)))
    write_rows(
        "control_sweeps.csv",
        ("parameter", "value", "R0", "Istar", "S", "ST", "E", "I", "P"),
        rows,
    )


def write_baseline() -> None:
    write_rows(
        "illustrative_baseline_parameters.csv",
        ("parameter", "value", "status"),
        [(name, BASELINE[name], "illustrative") for name in PARAMETER_ORDER],
    )


def main() -> None:
    plt.style.use("seaborn-v0_8-paper")
    print(f"R0 (baseline) = {r0(BASELINE):.6f}")
    print(f"R0 (DFE)      = {r0(DFE_SCENARIO):.6f}")
    print("EE =", np.array2string(endemic_equilibrium(BASELINE), precision=6))
    print("DFE =", np.array2string(disease_free_equilibrium(DFE_SCENARIO), precision=6))
    plot_stability_scenarios()
    plot_sensitivities()
    plot_identifiability_and_bifurcation()
    combine_diagnostic_figures()
    write_control_sweeps()
    write_baseline()


if __name__ == "__main__":
    main()
