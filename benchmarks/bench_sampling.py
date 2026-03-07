"""
Benchmarks for BasinHoppingSampler.sample().

Quick smoke suite (fast, use to check whether an optimisation helped):
    pytest benchmarks/ -k quick --benchmark-only --benchmark-sort=mean

Full suite:
    pytest benchmarks/ --benchmark-only --benchmark-sort=mean
    pytest benchmarks/ --benchmark-only --benchmark-histogram
    pytest benchmarks/ --benchmark-only --benchmark-save=baseline

Compare against a saved baseline:
    pytest benchmarks/ --benchmark-only --benchmark-compare=baseline

Profile with cProfile:
    python -m cProfile -o benchmarks/profile.out -m pytest benchmarks/ --benchmark-disable
    python -m pstats benchmarks/profile.out

Or using the built-in profiler helper at the bottom of this file:
    python benchmarks/bench_sampling.py
"""

import cProfile
import io
import pstats

import numpy as np

from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray) -> float:
    A = 10
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ackley(x: np.ndarray) -> float:
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + 20.0
        + np.e
    )


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------

DOMAIN_2D = [(-5.0, 5.0), (-5.0, 5.0)]
DOMAIN_5D = [(-5.0, 5.0)] * 5
DOMAIN_10D = [(-5.0, 5.0)] * 10

# ---------------------------------------------------------------------------
# Sampler configs
# ---------------------------------------------------------------------------

_BASE = dict(n_iter_no_change=100, seed=42)

CFG_SMALL = BasinHoppingSamplerConfig(n_runs=10, **_BASE)
CFG_MEDIUM = BasinHoppingSamplerConfig(n_runs=50, **_BASE)
CFG_LARGE = BasinHoppingSamplerConfig(n_runs=200, **_BASE)

CFG_STEP_PCT = BasinHoppingSamplerConfig(n_runs=50, step_mode="percentage", step_size=0.05, **_BASE)
CFG_BOUNDED_OFF = BasinHoppingSamplerConfig(n_runs=50, bounded=False, **_BASE)
CFG_FIT_PREC = BasinHoppingSamplerConfig(n_runs=50, fitness_precision=4, coordinate_precision=3, **_BASE)

# Quick config: minimal runs / iterations — finishes in <1 s total, good for
# a fast sanity-check that a change improved (or didn't regress) performance.
CFG_QUICK = BasinHoppingSamplerConfig(n_runs=5, n_iter_no_change=50, seed=42)


# ---------------------------------------------------------------------------
# pytest-benchmark fixtures
# ---------------------------------------------------------------------------

def _run_sample(func, domain, config):
    sampler = BasinHoppingSampler(config)
    return sampler.sample(func, domain)


def _run_sample_concurrent(func, domain, config, *, n_jobs: int = -1, prefer: str = "threads"):
    """Equivalent to _run_sample but executes runs concurrently via joblib.

    Mirrors what sample() does: resolve initial points → run BH chains →
    construct trace data; the only difference is that the BH chains are
    dispatched in parallel.
    """
    sampler = BasinHoppingSampler(config)
    pts = sampler._resolve_initial_points(None, domain)
    raw = sampler._basin_hopping_sampling_concurrent(
        func, domain, pts, n_jobs=n_jobs, prefer=prefer
    )
    return sampler._construct_trace_data(raw), raw


# ---------------------------------------------------------------------------
# QUICK suite  (pytest -k quick --benchmark-only)
# One benchmark per objective function, 5 runs each.
# Run this after every change to see the effect at a glance.
# ---------------------------------------------------------------------------

def test_bench_quick_sphere(benchmark):
    """Quick: sphere 2-D, 5 runs."""
    benchmark(_run_sample, sphere, DOMAIN_2D, CFG_QUICK)


def test_bench_quick_rastrigin(benchmark):
    """Quick: rastrigin 2-D, 5 runs."""
    benchmark(_run_sample, rastrigin, DOMAIN_2D, CFG_QUICK)


def test_bench_quick_rosenbrock(benchmark):
    """Quick: rosenbrock 2-D, 5 runs."""
    benchmark(_run_sample, rosenbrock, DOMAIN_2D, CFG_QUICK)


def test_bench_quick_ackley(benchmark):
    """Quick: ackley 2-D, 5 runs."""
    benchmark(_run_sample, ackley, DOMAIN_2D, CFG_QUICK)


# --- n_runs scaling ---------------------------------------------------------

def test_bench_sample_small_runs(benchmark):
    """10 runs, sphere 2-D — baseline cost."""
    benchmark(_run_sample, sphere, DOMAIN_2D, CFG_SMALL)


def test_bench_sample_medium_runs(benchmark):
    """50 runs, sphere 2-D."""
    benchmark(_run_sample, sphere, DOMAIN_2D, CFG_MEDIUM)


def test_bench_sample_large_runs(benchmark):
    """200 runs, sphere 2-D — scales with n_runs."""
    benchmark(_run_sample, sphere, DOMAIN_2D, CFG_LARGE)


# --- dimensionality scaling -------------------------------------------------

def test_bench_sample_2d(benchmark):
    """50 runs, sphere 2-D."""
    benchmark(_run_sample, sphere, DOMAIN_2D, CFG_MEDIUM)


def test_bench_sample_5d(benchmark):
    """50 runs, sphere 5-D."""
    benchmark(_run_sample, sphere, DOMAIN_5D, CFG_MEDIUM)


def test_bench_sample_10d(benchmark):
    """50 runs, sphere 10-D — scales with dimensionality."""
    benchmark(_run_sample, sphere, DOMAIN_10D, CFG_MEDIUM)


# --- objective function complexity ------------------------------------------

def test_bench_sample_sphere(benchmark):
    """50 runs, sphere (cheap quadratic)."""
    benchmark(_run_sample, sphere, DOMAIN_2D, CFG_MEDIUM)


def test_bench_sample_rastrigin(benchmark):
    """50 runs, Rastrigin (many local optima, trig-heavy)."""
    benchmark(_run_sample, rastrigin, DOMAIN_2D, CFG_MEDIUM)


def test_bench_sample_rosenbrock(benchmark):
    """50 runs, Rosenbrock (narrow valley, hard for L-BFGS-B)."""
    benchmark(_run_sample, rosenbrock, DOMAIN_2D, CFG_MEDIUM)


def test_bench_sample_ackley(benchmark):
    """50 runs, Ackley (exp/trig overhead)."""
    benchmark(_run_sample, ackley, DOMAIN_2D, CFG_MEDIUM)


# --- configuration variants -------------------------------------------------

def test_bench_sample_step_percentage(benchmark):
    """50 runs, percentage-based step size."""
    benchmark(_run_sample, sphere, DOMAIN_2D, CFG_STEP_PCT)


def test_bench_sample_unbounded(benchmark):
    """50 runs, no bound enforcement during perturbation."""
    benchmark(_run_sample, sphere, DOMAIN_2D, CFG_BOUNDED_OFF)


def test_bench_sample_with_precision(benchmark):
    """50 runs, fitness and coordinate precision rounding enabled."""
    benchmark(_run_sample, sphere, DOMAIN_2D, CFG_FIT_PREC)


# ---------------------------------------------------------------------------
# CONCURRENT suite  — parallel counterparts to key sequential benchmarks
#
# Naming convention: replace "sample" with "concurrent" so that
# --benchmark-compare shows sequential vs concurrent side-by-side.
# ---------------------------------------------------------------------------

# --- Quick concurrent (mirrors QUICK suite) ---------------------------------

def test_bench_quick_concurrent_sphere(benchmark):
    """Quick concurrent: sphere 2-D, 5 runs."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_2D, CFG_QUICK)


def test_bench_quick_concurrent_rastrigin(benchmark):
    """Quick concurrent: rastrigin 2-D, 5 runs."""
    benchmark(_run_sample_concurrent, rastrigin, DOMAIN_2D, CFG_QUICK)


def test_bench_quick_concurrent_rosenbrock(benchmark):
    """Quick concurrent: rosenbrock 2-D, 5 runs."""
    benchmark(_run_sample_concurrent, rosenbrock, DOMAIN_2D, CFG_QUICK)


def test_bench_quick_concurrent_ackley(benchmark):
    """Quick concurrent: ackley 2-D, 5 runs."""
    benchmark(_run_sample_concurrent, ackley, DOMAIN_2D, CFG_QUICK)


# --- n_jobs scaling ---------------------------------------------------------
# Same workload (sphere 2-D, 50 runs); vary the number of worker threads.
# Shows the overhead of parallelism at low job counts and the gain at higher
# counts.  Compare against test_bench_sample_medium_runs (sequential).

def test_bench_concurrent_n_jobs_1(benchmark):
    """Concurrent, n_jobs=1 — joblib overhead with no actual parallelism."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_2D, CFG_MEDIUM, n_jobs=1)


def test_bench_concurrent_n_jobs_2(benchmark):
    """Concurrent, n_jobs=2 — two threads."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_2D, CFG_MEDIUM, n_jobs=2)


def test_bench_concurrent_n_jobs_4(benchmark):
    """Concurrent, n_jobs=4 — four threads."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_2D, CFG_MEDIUM, n_jobs=4)


def test_bench_concurrent_n_jobs_all(benchmark):
    """Concurrent, n_jobs=-1 — all available CPUs."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_2D, CFG_MEDIUM, n_jobs=-1)


# --- threads vs processes ---------------------------------------------------
# Compare joblib backends on a medium workload.  Both use n_jobs=-1.
# "threads" avoids pickling but shares the GIL for pure-Python code;
# "processes" fully bypasses the GIL at the cost of serialisation overhead.

def test_bench_concurrent_prefer_threads(benchmark):
    """Concurrent threads backend, sphere 2-D, 50 runs."""
    benchmark(
        _run_sample_concurrent, sphere, DOMAIN_2D, CFG_MEDIUM,
        n_jobs=-1, prefer="threads",
    )


def test_bench_concurrent_prefer_processes(benchmark):
    """Concurrent processes backend, sphere 2-D, 50 runs."""
    benchmark(
        _run_sample_concurrent, sphere, DOMAIN_2D, CFG_MEDIUM,
        n_jobs=-1, prefer="processes",
    )


# --- n_runs scaling (concurrent) -------------------------------------------

def test_bench_concurrent_small_runs(benchmark):
    """Concurrent, 10 runs, sphere 2-D."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_2D, CFG_SMALL)


def test_bench_concurrent_medium_runs(benchmark):
    """Concurrent, 50 runs, sphere 2-D."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_2D, CFG_MEDIUM)


def test_bench_concurrent_large_runs(benchmark):
    """Concurrent, 200 runs, sphere 2-D."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_2D, CFG_LARGE)


# --- dimensionality scaling (concurrent) ------------------------------------

def test_bench_concurrent_2d(benchmark):
    """Concurrent, 50 runs, sphere 2-D."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_2D, CFG_MEDIUM)


def test_bench_concurrent_5d(benchmark):
    """Concurrent, 50 runs, sphere 5-D."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_5D, CFG_MEDIUM)


def test_bench_concurrent_10d(benchmark):
    """Concurrent, 50 runs, sphere 10-D."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_10D, CFG_MEDIUM)


# --- objective function complexity (concurrent) -----------------------------

def test_bench_concurrent_sphere(benchmark):
    """Concurrent, 50 runs, sphere (cheap quadratic)."""
    benchmark(_run_sample_concurrent, sphere, DOMAIN_2D, CFG_MEDIUM)


def test_bench_concurrent_rastrigin(benchmark):
    """Concurrent, 50 runs, Rastrigin (many local optima, trig-heavy)."""
    benchmark(_run_sample_concurrent, rastrigin, DOMAIN_2D, CFG_MEDIUM)


def test_bench_concurrent_rosenbrock(benchmark):
    """Concurrent, 50 runs, Rosenbrock (narrow valley)."""
    benchmark(_run_sample_concurrent, rosenbrock, DOMAIN_2D, CFG_MEDIUM)


def test_bench_concurrent_ackley(benchmark):
    """Concurrent, 50 runs, Ackley (exp/trig overhead)."""
    benchmark(_run_sample_concurrent, ackley, DOMAIN_2D, CFG_MEDIUM)


# ---------------------------------------------------------------------------
# Standalone cProfile helper
# ---------------------------------------------------------------------------

def _profile(func, domain, config, *, top_n: int = 20) -> None:
    """Run sample() under cProfile and print the top-N hotspots."""
    sampler = BasinHoppingSampler(config)
    pr = cProfile.Profile()
    pr.enable()
    sampler.sample(func, domain)
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(top_n)
    print(buf.getvalue())


def _compare_sequential_vs_concurrent(
    func,
    domain,
    config,
    n_repeats: int = 3,
    n_jobs: int = -1,
) -> None:
    """Print mean wall-clock times for sequential vs concurrent variants."""
    import time

    def _timeit(fn, *args, **kwargs) -> float:
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)
        return sum(times) / len(times)

    t_seq = _timeit(_run_sample, func, domain, config)
    t_thr = _timeit(_run_sample_concurrent, func, domain, config, n_jobs=n_jobs, prefer="threads")
    t_proc = _timeit(_run_sample_concurrent, func, domain, config, n_jobs=n_jobs, prefer="processes")

    speedup_thr = t_seq / t_thr if t_thr > 0 else float("inf")
    speedup_proc = t_seq / t_proc if t_proc > 0 else float("inf")

    print(f"  sequential          : {t_seq:.3f} s")
    print(f"  concurrent (threads): {t_thr:.3f} s  [{speedup_thr:+.2f}x]")
    print(f"  concurrent (procs)  : {t_proc:.3f} s  [{speedup_proc:+.2f}x]")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("cProfile — sphere 2-D, 50 runs")
    print("=" * 60)
    _profile(sphere, DOMAIN_2D, CFG_MEDIUM)

    print("=" * 60)
    print("cProfile — rastrigin 2-D, 50 runs")
    print("=" * 60)
    _profile(rastrigin, DOMAIN_2D, CFG_MEDIUM)

    print("=" * 60)
    print("cProfile — sphere 10-D, 50 runs")
    print("=" * 60)
    _profile(sphere, DOMAIN_10D, CFG_MEDIUM)

    # ------------------------------------------------------------------
    # Wall-clock comparison: sequential vs concurrent
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Sequential vs concurrent — wall-clock comparison (mean of 3)")
    print("=" * 60)

    comparisons = [
        ("sphere 2-D, 50 runs",    sphere,     DOMAIN_2D,  CFG_MEDIUM),
        ("rastrigin 2-D, 50 runs", rastrigin,  DOMAIN_2D,  CFG_MEDIUM),
        ("rosenbrock 2-D, 50 runs",rosenbrock, DOMAIN_2D,  CFG_MEDIUM),
        ("ackley 2-D, 50 runs",    ackley,     DOMAIN_2D,  CFG_MEDIUM),
        ("sphere 5-D, 50 runs",    sphere,     DOMAIN_5D,  CFG_MEDIUM),
        ("sphere 10-D, 50 runs",   sphere,     DOMAIN_10D, CFG_MEDIUM),
        ("sphere 2-D, 200 runs",   sphere,     DOMAIN_2D,  CFG_LARGE),
    ]

    for label, func_, domain_, config_ in comparisons:
        print(f"[ {label} ]")
        _compare_sequential_vs_concurrent(func_, domain_, config_)
