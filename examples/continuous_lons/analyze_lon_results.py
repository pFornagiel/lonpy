"""Analyze and print LON statistics from trace data files.

This script reads LON trace data from CSV files and computes/prints the
same statistics as shown in figure_4.py without needing to re-run the
expensive sampling experiment.
"""

from random import seed
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from lonpy.visualization import LONVisualizer

# Add src to path to import local lonpy
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lonpy import LON

# Global optimum values for benchmark functions
GLOBAL_OPTIMA = {
    "Ackley": 0.0,
    "Rastrigin": 0.0,
    "Birastrigin": 0.0,  # Approximate global optimum
}


def analyze_trace_file(filepath: str, global_optimum: float, tolerance: float = 1e-4) -> dict:
    """Load trace data and compute LON metrics.
    
    Args:
        filepath: Path to CSV file with trace data.
        global_optimum: Known global optimum value.
        tolerance: Tolerance for considering a solution as global optimum.
        
    Returns:
        Dictionary with LON, CMLON, and their metrics.
    """
    trace_df = pd.read_csv(filepath)
    
    # Create LON from trace data
    lon = LON.from_trace_data(trace_df)
    cmlon = lon.to_cmlon()

    visualizer = LONVisualizer()

    # visualizer.plot_3d(lon, output_path=f"{Path(filepath).stem}_lon_3d.png")
    # visualizer.plot_2d(lon, output_path=f"{Path(filepath).stem}_lon_2d.png")
    visualizer.visualize_all(lon, output_folder=f"{Path(filepath).stem}_lon_all")

    
    lon_metrics = lon.compute_metrics()
    cmlon_metrics = cmlon.compute_metrics()
    
    # Compute success and deviation metrics
    # Get the final fitness value for each run (last fit2 in each run)
    final_fitness_per_run = trace_df.groupby('run')['fit2'].last()
    
    # Convert scaled integer fitness back to float
    # Based on hash_digits=5, the scale is 10^5
    final_fitness_values = final_fitness_per_run / 100000.0
    
    # Success: proportion of runs that reached global optimum (within tolerance)
    success = np.sum(np.abs(final_fitness_values - global_optimum) <= tolerance) / len(final_fitness_values)
    
    # Deviation: mean absolute difference from global optimum
    deviation = np.mean(np.abs(final_fitness_values - global_optimum))
    
    return {
        "lon": lon,
        "cmlon": cmlon,
        "lon_metrics": lon_metrics,
        "cmlon_metrics": cmlon_metrics,
        "success": success,
        "deviation": deviation,
    }


def print_summary(results: dict) -> None:
    """Print formatted summary of LON metrics.
    
    Args:
        results: Dictionary with function names as keys and result dicts as values.
    """
    print("=" * 90)
    print("Summary of LON Metrics")
    print("=" * 90)
    print(f"{'Function':<15} {'n_optima':>10} {'n_funnels':>10} {'n_global':>10} {'strength':>10} {'success':>10} {'deviation':>12}")
    print("-" * 90)

    for name in sorted(results.keys()):
        m = results[name]["cmlon_metrics"]
        success = results[name]["success"]
        deviation = results[name]["deviation"]
        print(
            f"{name:<15} {m['n_optima']:>10} {m['n_funnels']:>10} "
            f"{m['n_global_funnels']:>10} {m['strength']:>10.4f} {success:>10.4f} {deviation:>12.6f}"
        )

    print("=" * 90)


def print_detailed_summary(results: dict) -> None:
    """Print detailed summary with both LON and CMLON metrics.
    
    Args:
        results: Dictionary with function names as keys and result dicts as values.
    """
    print("=" * 80)
    print("Detailed LON and CMLON Metrics")
    print("=" * 80)
    
    for name in sorted(results.keys()):
        lon_m = results[name]["lon_metrics"]
        cmlon_m = results[name]["cmlon_metrics"]
        
        print(f"\n{name}:")
        print(f"  LON:")
        print(f"    n_optima:         {lon_m['n_optima']}")
        print(f"    n_funnels:        {lon_m['n_funnels']}")
        print(f"    n_global_funnels: {lon_m['n_global_funnels']}")
        print(f"    neutral:          {lon_m['neutral']:.4f}")
        print(f"    strength:         {lon_m['strength']:.4f}")
        
        print(f"  CMLON:")
        print(f"    n_optima:         {cmlon_m['n_optima']}")
        print(f"    n_funnels:        {cmlon_m['n_funnels']}")
        print(f"    n_global_funnels: {cmlon_m['n_global_funnels']}")
        print(f"    neutral:          {cmlon_m['neutral']:.4f}")
        print(f"    strength:         {cmlon_m['strength']:.4f}")
        
        # Compression ratio
        compression = 1.0 - (cmlon_m['n_optima'] / lon_m['n_optima']) if lon_m['n_optima'] > 0 else 0
        print(f"    compression:      {compression:.4f} ({100*compression:.1f}%)")
        
        # Success and deviation
        print(f"  Performance:")
        print(f"    success:          {results[name]['success']:.4f} ({100*results[name]['success']:.1f}%)")
        print(f"    deviation:        {results[name]['deviation']:.6f}")


if __name__ == "__main__":
    # Define expected trace data files
    trace_files = {
        "Ackley": "ackley_trace_data.csv",
        "Rastrigin": "rastrigin_trace_data.csv",
        "Birastrigin": "birastrigin_trace_data.csv",
    }
    
    results = {}
    
    print("Loading trace data and computing metrics...")
    print()
    
    for name, filename in trace_files.items():
        filepath = Path(__file__).parent / filename
        
        if not filepath.exists():
            print(f"Warning: {filename} not found at {filepath}")
            continue
        
        print(f"Processing {name}...", end=" ")
        global_optimum = GLOBAL_OPTIMA.get(name, 0.0)
        results[name] = analyze_trace_file(str(filepath), global_optimum=global_optimum)
        print("Done")
    
    print()
    
    if results:
        print_summary(results)
        print()
        print_detailed_summary(results)
    else:
        print("No trace data files found. Please run figure_4.py first to generate them.")
