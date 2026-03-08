from datetime import datetime
from pathlib import Path
from time import perf_counter

import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "bioma"))
from problems import ackley4, griewank, schwefel2_26

from lonpy import LON, BasinHoppingSampler, BasinHoppingSamplerConfig, LONVisualizer

OUTPUT_DIR = Path(__file__).parent / "benchmark_output"

_COL_WIDTHS = {
    "Function": 11,
    "Config": 40,
    "Nodes": 6,
    "Edges": 6,
    "Accept %": 9,
    "Best Fit": 12,
    "Time (s)": 9,
}
_COLS = list(_COL_WIDTHS.keys())


def _hr(char: str = "─") -> str:
    return "┼".join(char * (_COL_WIDTHS[c] + 2) for c in _COLS)


def _row(values: dict) -> str:
    cells = []
    for col in _COLS:
        w = _COL_WIDTHS[col]
        v = str(values.get(col, ""))
        if col in ("Nodes", "Edges"):
            cells.append(v.rjust(w))
        else:
            cells.append(v.ljust(w))
    return "│ " + " │ ".join(cells) + " │"


def _header() -> str:
    return _row({c: c for c in _COLS})


def _print_section_table(df: pd.DataFrame, title: str, log_lines: list[str]) -> None:
    border_top = "┌" + "┬".join("─" * (_COL_WIDTHS[c] + 2) for c in _COLS) + "┐"
    border_head = "├" + _hr() + "┤"
    border_sep = "├" + _hr("─") + "┤"
    border_bottom = "└" + "┴".join("─" * (_COL_WIDTHS[c] + 2) for c in _COLS) + "┘"

    lines = [
        "",
        f"  {title}",
        border_top,
        _header(),
        border_head,
    ]
    prev_func = None
    for _, row in df.iterrows():
        if prev_func is not None and row["Function"] != prev_func:
            lines.append(border_sep)
        lines.append(_row(row.to_dict()))
        prev_func = row["Function"]
    lines.append(border_bottom)

    output = "\n".join(lines)
    print(output)
    log_lines.append(output)


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)


def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def run_benchmark():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines: list[str] = [f"lonpy benchmark  —  {run_ts}\n"]

    dim = 5

    # Each function gets its own domain; sampler picks per-function domain
    functions = [
        ("Sphere", sphere, [(-5.0, 5.0)] * dim),
        ("Rosenbrock", rosenbrock, [(-5.0, 5.0)] * dim),
        ("Rastrigin", rastrigin, [(-5.12, 5.12)] * dim),
        ("Griewank", griewank, [(-600.0, 600.0)] * dim),
        ("Schwefel", schwefel2_26, [(-500.0, 500.0)] * dim),
        ("Ackley4", ackley4, [(-35.0, 35.0)] * dim),
    ]

    # --- Section 1: Step-mode / step-size comparison ---
    step_configs = {
        "1. Fixed (Defaults)": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=500,
            n_iter_no_change=100,
            step_mode="fixed",
            step_size=0.01,
            seed=42,
        ),
        "2. Percentage Steps (Small)": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=500,
            n_iter_no_change=100,
            step_mode="percentage",
            step_size=0.05,
            seed=42,
        ),
        "3. Percentage Steps (Medium)": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=500,
            n_iter_no_change=100,
            step_mode="percentage",
            step_size=0.10,
            seed=42,
        ),
        "4. Percentage Steps (Large)": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=500,
            n_iter_no_change=100,
            step_mode="percentage",
            step_size=0.20,
            seed=42,
        ),
    }

    # --- Section 2: n_iter_no_change comparison (is smaller better?) ---
    early_stop_configs = {
        "iter-10": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=None,
            n_iter_no_change=10,
            step_mode="percentage",
            step_size=0.10,
            seed=42,
        ),
        "iter-25": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=None,
            n_iter_no_change=25,
            step_mode="percentage",
            step_size=0.10,
            seed=42,
        ),
        "iter-50": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=None,
            n_iter_no_change=50,
            step_mode="percentage",
            step_size=0.10,
            seed=42,
        ),
        "iter-100": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=None,
            n_iter_no_change=100,
            step_mode="percentage",
            step_size=0.10,
            seed=42,
        ),
        "iter-250": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=None,
            n_iter_no_change=250,
            step_mode="percentage",
            step_size=0.10,
            seed=42,
        ),
        "iter-500": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=None,
            n_iter_no_change=500,
            step_mode="percentage",
            step_size=0.10,
            seed=42,
        ),
        "iter-1000": BasinHoppingSamplerConfig(
            n_runs=20,
            max_iter=None,
            n_iter_no_change=1000,
            step_mode="percentage",
            step_size=0.10,
            seed=42,
        ),
    }

    viz = LONVisualizer()
    results = []

    for section_name, configs in [
        ("step_mode", step_configs),
        ("n_iter_no_change", early_stop_configs),
    ]:
        section_header = f"\n{'=' * 60}\n  Section: {section_name}\n{'=' * 60}"
        print(section_header)
        log_lines.append(section_header)

        for func_name, func, domain in functions:
            func_header = f"\n  [{func_name}]"
            print(func_header)
            log_lines.append(func_header)

            for config_name, config in configs.items():
                sampler = BasinHoppingSampler(config)

                start_time = perf_counter()
                trace_df, raw_records = sampler.sample(func, domain)
                elapsed = perf_counter() - start_time

                # Acceptance metrics
                total_perturbations = len(raw_records)
                accepted_perturbations = sum(1 for r in raw_records if r["accepted"])
                acceptance_rate = (
                    accepted_perturbations / total_perturbations if total_perturbations else 0.0
                )

                if not trace_df.empty:
                    unique_nodes = pd.concat([trace_df["node1"], trace_df["node2"]]).nunique()
                    best_fitness = min(trace_df["fit2"].min(), trace_df["fit1"].min())
                    n_edges = len(trace_df)

                    # Build LON and save 2D visualisation
                    lon = LON.from_trace_data(trace_df)
                    safe_config = config_name.replace(" ", "_").replace(".", "").replace("/", "-")
                    img_path = OUTPUT_DIR / section_name / func_name / f"{safe_config}.png"
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    fig = viz.plot_2d(lon, output_path=img_path, seed=42)
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                    img_msg = f"    [{config_name}] LON saved → {img_path.relative_to(Path(__file__).parent)}"
                    print(img_msg)
                    log_lines.append(img_msg)
                else:
                    unique_nodes = 0
                    best_fitness = float("inf")
                    n_edges = 0

                results.append(
                    {
                        "Section": section_name,
                        "Function": func_name,
                        "Config": config_name,
                        "Nodes": unique_nodes,
                        "Edges": n_edges,
                        "Accept %": f"{acceptance_rate:.1%}",
                        "Best Fit": f"{best_fitness:.4f}",
                        "Time (s)": f"{elapsed:.2f}",
                    }
                )

    df_results = pd.DataFrame(results)

    step_df = (
        df_results[df_results["Section"] == "step_mode"]
        .drop(columns="Section")
        .reset_index(drop=True)
    )
    es_df = (
        df_results[df_results["Section"] == "n_iter_no_change"]
        .drop(columns="Section")
        .reset_index(drop=True)
    )

    _print_section_table(step_df, "Step-mode / step-size comparison", log_lines)
    _print_section_table(es_df, "n_iter_no_change sweep  (smaller → faster early-stop)", log_lines)

    csv_path = OUTPUT_DIR / "results.csv"
    df_results.to_csv(csv_path, index=False)

    log_path = OUTPUT_DIR / "results.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"\nResults saved → {csv_path}")
    print(f"Log saved     → {log_path}")


if __name__ == "__main__":
    run_benchmark()
