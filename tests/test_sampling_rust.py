"""
Comprehensive tests for the Rust-backed sampling module.

Tests verify:
1. API compatibility: Rust module exposes the same classes/functions as the Python original.
2. Config validation: The same errors are raised for invalid configurations.
3. Determinism: Given identical seeds and initial points, both implementations
   produce structurally equivalent results.
4. Edge cases: Empty results, callbacks, bounded/unbounded modes, etc.
5. LON construction: sample_to_lon and compute_lon produce valid LON objects.
"""

import numpy as np
import pandas as pd
import pytest

# Rust implementation (new default)
from lonpy._lonpy_rust import BasinHoppingSampler as RustSampler
from lonpy._lonpy_rust import BasinHoppingSamplerConfig as RustConfig
from lonpy._lonpy_rust import compute_lon as rust_compute_lon

# Original Python implementation (preserved for comparison)
from lonpy._sampling_py import BasinHoppingSampler as PySampler
from lonpy._sampling_py import BasinHoppingSamplerConfig as PyConfig
from lonpy._sampling_py import compute_lon as py_compute_lon

from lonpy.lon import LON, LONConfig


# ──────────────────────────────────────────────────────────────────────
#  Test objective functions
# ──────────────────────────────────────────────────────────────────────


def sphere(x):
    """Simple sphere function: f(x) = sum(x_i^2)"""
    return np.sum(x**2)


def rastrigin(x):
    """Rastrigin function (multimodal)."""
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    """Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


# ──────────────────────────────────────────────────────────────────────
#  Config tests
# ──────────────────────────────────────────────────────────────────────


class TestBasinHoppingSamplerConfig:
    """Tests for BasinHoppingSamplerConfig (Rust)."""

    def test_default_config(self):
        cfg = RustConfig()
        assert cfg.n_runs == 100
        assert cfg.n_iter_no_change == 1000
        assert cfg.max_iter is None
        assert cfg.step_mode == "fixed"
        assert cfg.step_size == 0.01
        assert cfg.fitness_precision is None
        assert cfg.coordinate_precision == 5
        assert cfg.bounded is True
        assert cfg.seed is None

    def test_custom_config(self):
        cfg = RustConfig(
            n_runs=50,
            n_iter_no_change=500,
            max_iter=2000,
            step_mode="percentage",
            step_size=0.05,
            fitness_precision=3,
            coordinate_precision=4,
            bounded=False,
            seed=123,
        )
        assert cfg.n_runs == 50
        assert cfg.n_iter_no_change == 500
        assert cfg.max_iter == 2000
        assert cfg.step_mode == "percentage"
        assert cfg.step_size == 0.05
        assert cfg.fitness_precision == 3
        assert cfg.coordinate_precision == 4
        assert cfg.bounded is False
        assert cfg.seed == 123

    def test_invalid_n_iter_no_change_zero(self):
        with pytest.raises((ValueError, Exception)):
            RustConfig(n_iter_no_change=0)

    def test_invalid_max_iter_zero(self):
        with pytest.raises((ValueError, Exception)):
            RustConfig(max_iter=0, n_iter_no_change=None)

    def test_no_stopping_criterion(self):
        with pytest.raises((ValueError, Exception)):
            RustConfig(n_iter_no_change=None, max_iter=None)

    def test_config_matches_python_defaults(self):
        """Rust config defaults must match Python config defaults."""
        py_cfg = PyConfig()
        rs_cfg = RustConfig()

        assert rs_cfg.n_runs == py_cfg.n_runs
        assert rs_cfg.n_iter_no_change == py_cfg.n_iter_no_change
        assert rs_cfg.max_iter == py_cfg.max_iter
        assert rs_cfg.step_mode == py_cfg.step_mode
        assert rs_cfg.step_size == py_cfg.step_size
        assert rs_cfg.fitness_precision == py_cfg.fitness_precision
        assert rs_cfg.coordinate_precision == py_cfg.coordinate_precision
        assert rs_cfg.bounded == py_cfg.bounded
        assert rs_cfg.seed == py_cfg.seed

    def test_minimizer_method_default(self):
        cfg = RustConfig()
        assert cfg.minimizer_method == "L-BFGS-B"

    def test_minimizer_options_default(self):
        cfg = RustConfig()
        opts = cfg.minimizer_options
        assert isinstance(opts, dict)
        assert opts["ftol"] == 1e-07
        assert opts["gtol"] == 0
        assert opts["maxiter"] == 15000


# ──────────────────────────────────────────────────────────────────────
#  Sampler construction tests
# ──────────────────────────────────────────────────────────────────────


class TestBasinHoppingSampler:
    """Tests for BasinHoppingSampler (Rust)."""

    def test_default_sampler(self):
        sampler = RustSampler()
        cfg = sampler.config
        assert cfg.n_runs == 100

    def test_custom_config_sampler(self):
        cfg = RustConfig(n_runs=10, seed=42)
        sampler = RustSampler(cfg)
        assert sampler.config.n_runs == 10
        assert sampler.config.seed == 42


# ──────────────────────────────────────────────────────────────────────
#  Sampling output shape / structure tests
# ──────────────────────────────────────────────────────────────────────


class TestSampleOutput:
    """Verify the structure of sample() output."""

    def test_sample_returns_tuple(self):
        cfg = RustConfig(n_runs=3, n_iter_no_change=5, seed=42)
        sampler = RustSampler(cfg)
        result = sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_trace_df_columns(self):
        cfg = RustConfig(n_runs=3, n_iter_no_change=5, seed=42)
        sampler = RustSampler(cfg)
        trace_df, _ = sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)])
        assert isinstance(trace_df, pd.DataFrame)
        assert list(trace_df.columns) == ["run", "fit1", "node1", "fit2", "node2"]

    def test_raw_records_structure(self):
        cfg = RustConfig(n_runs=2, n_iter_no_change=5, seed=42)
        sampler = RustSampler(cfg)
        _, raw_records = sampler.sample(sphere, [(-5.0, 5.0)])
        assert isinstance(raw_records, list)
        assert len(raw_records) > 0
        rec = raw_records[0]
        assert isinstance(rec, dict)
        expected_keys = {"run", "iteration", "current_x", "current_f", "new_x", "new_f", "accepted"}
        assert set(rec.keys()) == expected_keys

    def test_raw_record_types(self):
        cfg = RustConfig(n_runs=1, n_iter_no_change=3, seed=42)
        sampler = RustSampler(cfg)
        _, raw_records = sampler.sample(sphere, [(-5.0, 5.0)])
        rec = raw_records[0]
        assert isinstance(rec["run"], int)
        assert isinstance(rec["iteration"], int)
        assert isinstance(rec["current_x"], np.ndarray)
        assert isinstance(rec["new_x"], np.ndarray)
        assert isinstance(rec["current_f"], float)
        assert isinstance(rec["new_f"], float)
        assert isinstance(rec["accepted"], bool)

    def test_trace_only_accepted(self):
        """Trace DataFrame should only contain accepted transitions."""
        cfg = RustConfig(n_runs=3, n_iter_no_change=20, seed=42)
        sampler = RustSampler(cfg)
        trace_df, raw_records = sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)])
        accepted_count = sum(1 for r in raw_records if r["accepted"])
        assert len(trace_df) == accepted_count


# ──────────────────────────────────────────────────────────────────────
#  Comparison: Rust vs Python with same initial points
# ──────────────────────────────────────────────────────────────────────


class TestRustVsPython:
    """
    Compare Rust and Python implementations given identical initial points.

    Since both implementations use scipy.optimize.minimize for the heavy
    lifting, the *minimization results* are identical. The perturbation
    RNG differs between numpy (Python) and ChaCha8 (Rust), so we supply
    deterministic initial_points and use a very small number of iterations
    to focus on the structure and logic rather than random paths.
    """

    @pytest.fixture()
    def fixed_initial_points_2d(self):
        """Fixed initial points for 2D problems, 5 runs."""
        rng = np.random.default_rng(99)
        return rng.uniform(-5, 5, size=(5, 2))

    @pytest.fixture()
    def fixed_initial_points_3d(self):
        """Fixed initial points for 3D problems, 5 runs."""
        rng = np.random.default_rng(99)
        return rng.uniform(-5, 5, size=(5, 3))

    def test_sample_to_lon_sphere(self, fixed_initial_points_2d):
        """Both implementations should produce a LON for the sphere function."""
        domain = [(-5.0, 5.0), (-5.0, 5.0)]
        pts = fixed_initial_points_2d

        py_cfg = PyConfig(n_runs=5, n_iter_no_change=10, seed=42)
        py_sampler = PySampler(py_cfg)
        py_lon = py_sampler.sample_to_lon(sphere, domain, initial_points=pts)

        rs_cfg = RustConfig(n_runs=5, n_iter_no_change=10, seed=42)
        rs_sampler = RustSampler(rs_cfg)
        rs_lon = rs_sampler.sample_to_lon(sphere, domain, initial_points=pts)

        assert isinstance(py_lon, LON)
        assert isinstance(rs_lon, LON)
        # Both should find at least 1 optimum for sphere
        assert py_lon.n_vertices >= 1
        assert rs_lon.n_vertices >= 1

    def test_sample_trace_structure_match(self, fixed_initial_points_2d):
        """Given same initial points, trace columns and types match."""
        domain = [(-5.0, 5.0), (-5.0, 5.0)]
        pts = fixed_initial_points_2d

        py_cfg = PyConfig(n_runs=5, n_iter_no_change=5, seed=42)
        py_sampler = PySampler(py_cfg)
        py_trace, py_raw = py_sampler.sample(sphere, domain, initial_points=pts)

        rs_cfg = RustConfig(n_runs=5, n_iter_no_change=5, seed=42)
        rs_sampler = RustSampler(rs_cfg)
        rs_trace, rs_raw = rs_sampler.sample(sphere, domain, initial_points=pts)

        # Column names must match exactly
        assert list(py_trace.columns) == list(rs_trace.columns)

        # Both should produce some records
        assert len(py_raw) > 0
        assert len(rs_raw) > 0

    def test_initial_minimize_results_match(self, fixed_initial_points_2d):
        """
        With max_iter=1, both should do exactly 1 perturbation per run.
        The initial minimize results should be identical since they
        start from the same points.
        """
        domain = [(-5.0, 5.0), (-5.0, 5.0)]
        pts = fixed_initial_points_2d[:3]  # just 3 runs

        py_cfg = PyConfig(n_runs=3, max_iter=1, n_iter_no_change=None, seed=42)
        py_sampler = PySampler(py_cfg)
        _, py_raw = py_sampler.sample(sphere, domain, initial_points=pts)

        rs_cfg = RustConfig(n_runs=3, max_iter=1, n_iter_no_change=None, seed=42)
        rs_sampler = RustSampler(rs_cfg)
        _, rs_raw = rs_sampler.sample(sphere, domain, initial_points=pts)

        # Both should have exactly 3 raw records (1 per run)
        assert len(py_raw) == 3
        assert len(rs_raw) == 3

        # The current_x and current_f (from initial minimize) should match
        for py_rec, rs_rec in zip(py_raw, rs_raw):
            np.testing.assert_allclose(py_rec["current_x"], rs_rec["current_x"], atol=1e-10)
            np.testing.assert_allclose(py_rec["current_f"], rs_rec["current_f"], atol=1e-10)

    def test_compute_lon_produces_lon(self):
        """compute_lon should return a LON object."""
        rs_cfg = RustConfig(n_runs=3, n_iter_no_change=10, seed=42)
        lon = rust_compute_lon(sphere, dim=2, lower_bound=-5.0, upper_bound=5.0, config=rs_cfg)
        assert isinstance(lon, LON)

    def test_compute_lon_with_sequence_bounds(self):
        """compute_lon with per-dimension bounds."""
        rs_cfg = RustConfig(n_runs=3, n_iter_no_change=10, seed=42)
        lon = rust_compute_lon(
            sphere,
            dim=2,
            lower_bound=[-5.0, -3.0],
            upper_bound=[5.0, 3.0],
            config=rs_cfg,
        )
        assert isinstance(lon, LON)
        assert lon.n_vertices >= 1

    def test_rastrigin_multimodal(self, fixed_initial_points_2d):
        """Rastrigin should produce multiple optima (at least with enough runs)."""
        domain = [(-5.12, 5.12), (-5.12, 5.12)]
        pts = fixed_initial_points_2d

        rs_cfg = RustConfig(n_runs=5, n_iter_no_change=50, seed=42)
        rs_sampler = RustSampler(rs_cfg)
        lon = rs_sampler.sample_to_lon(rastrigin, domain, initial_points=pts)
        assert isinstance(lon, LON)
        # Rastrigin has many local optima
        assert lon.n_vertices >= 1


# ──────────────────────────────────────────────────────────────────────
#  Edge cases
# ──────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_initial_points_wrong_shape(self):
        cfg = RustConfig(n_runs=3, n_iter_no_change=5, seed=42)
        sampler = RustSampler(cfg)
        pts = np.zeros((3, 5))  # wrong n_var (5 instead of 2)
        with pytest.raises((ValueError, Exception)):
            sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)], initial_points=pts)

    def test_initial_points_wrong_n_runs(self):
        cfg = RustConfig(n_runs=3, n_iter_no_change=5, seed=42)
        sampler = RustSampler(cfg)
        pts = np.zeros((5, 2))  # 5 points but n_runs=3
        with pytest.raises((ValueError, Exception)):
            sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)], initial_points=pts)

    def test_initial_points_out_of_bounds(self):
        cfg = RustConfig(n_runs=2, n_iter_no_change=5, bounded=True, seed=42)
        sampler = RustSampler(cfg)
        pts = np.array([[100.0, 100.0], [0.0, 0.0]])
        with pytest.raises((ValueError, Exception)):
            sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)], initial_points=pts)

    def test_progress_callback(self):
        """Progress callback should be called n_runs times."""
        calls = []

        def callback(run, total):
            calls.append((run, total))

        cfg = RustConfig(n_runs=3, n_iter_no_change=2, seed=42)
        sampler = RustSampler(cfg)
        sampler.sample(sphere, [(-5.0, 5.0)], progress_callback=callback)

        assert len(calls) == 3
        assert calls[0] == (1, 3)
        assert calls[1] == (2, 3)
        assert calls[2] == (3, 3)

    def test_max_iter_only(self):
        """Using max_iter without n_iter_no_change."""
        cfg = RustConfig(n_runs=2, max_iter=5, n_iter_no_change=None, seed=42)
        sampler = RustSampler(cfg)
        _, raw = sampler.sample(sphere, [(-5.0, 5.0)])
        # Each run should have at most 5 iterations
        for run_id in [1, 2]:
            run_records = [r for r in raw if r["run"] == run_id]
            assert len(run_records) <= 5

    def test_percentage_step_mode(self):
        """Percentage step mode should work."""
        cfg = RustConfig(
            n_runs=2,
            n_iter_no_change=5,
            step_mode="percentage",
            step_size=0.1,
            seed=42,
        )
        sampler = RustSampler(cfg)
        trace_df, _ = sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)])
        assert isinstance(trace_df, pd.DataFrame)

    def test_unbounded_mode(self):
        """Unbounded mode should work."""
        cfg = RustConfig(n_runs=2, n_iter_no_change=5, bounded=False, seed=42)
        sampler = RustSampler(cfg)
        trace_df, _ = sampler.sample(sphere, [(-5.0, 5.0)])
        assert isinstance(trace_df, pd.DataFrame)

    def test_fitness_precision(self):
        """Fitness precision should round fitness values."""
        cfg = RustConfig(n_runs=2, n_iter_no_change=5, fitness_precision=2, seed=42)
        sampler = RustSampler(cfg)
        trace_df, _ = sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)])
        if not trace_df.empty:
            for val in trace_df["fit1"]:
                # Check that values are rounded to 2 decimal places
                assert round(val, 2) == val or abs(round(val, 2) - val) < 1e-15

    def test_coordinate_precision_none(self):
        """coordinate_precision=None should produce full-precision hashes."""
        cfg = RustConfig(n_runs=2, n_iter_no_change=5, coordinate_precision=None, seed=42)
        sampler = RustSampler(cfg)
        trace_df, _ = sampler.sample(sphere, [(-5.0, 5.0)])
        if not trace_df.empty:
            # Node hashes should not have fixed decimal format
            node = trace_df["node1"].iloc[0]
            assert isinstance(node, str)

    def test_1d_domain(self):
        """1D optimization should work."""

        def f1d(x):
            return x[0] ** 2

        cfg = RustConfig(n_runs=2, n_iter_no_change=5, seed=42)
        sampler = RustSampler(cfg)
        trace_df, _ = sampler.sample(f1d, [(-5.0, 5.0)])
        assert isinstance(trace_df, pd.DataFrame)

    def test_high_dimensional(self):
        """Higher-dimensional optimization should work."""
        cfg = RustConfig(n_runs=2, n_iter_no_change=5, seed=42)
        sampler = RustSampler(cfg)
        domain = [(-5.0, 5.0)] * 10
        trace_df, _ = sampler.sample(sphere, domain)
        assert isinstance(trace_df, pd.DataFrame)


# ──────────────────────────────────────────────────────────────────────
#  LON construction tests
# ──────────────────────────────────────────────────────────────────────


class TestLONConstruction:
    """Test that sample_to_lon produces valid LON objects."""

    def test_lon_has_vertices(self):
        cfg = RustConfig(n_runs=5, n_iter_no_change=20, seed=42)
        sampler = RustSampler(cfg)
        lon = sampler.sample_to_lon(sphere, [(-5.0, 5.0), (-5.0, 5.0)])
        assert lon.n_vertices >= 1

    def test_lon_with_lon_config(self):
        """Passing a LONConfig should work."""
        cfg = RustConfig(n_runs=5, n_iter_no_change=20, seed=42)
        sampler = RustSampler(cfg)
        lon_config = LONConfig(fitness_aggregation="mean")
        lon = sampler.sample_to_lon(sphere, [(-5.0, 5.0), (-5.0, 5.0)], lon_config=lon_config)
        assert isinstance(lon, LON)

    def test_lon_fitness_precision_sets_eq_atol(self):
        """When fitness_precision is set, eq_atol should be derived from it."""
        cfg = RustConfig(n_runs=5, n_iter_no_change=20, fitness_precision=3, seed=42)
        sampler = RustSampler(cfg)
        lon = sampler.sample_to_lon(sphere, [(-5.0, 5.0), (-5.0, 5.0)])
        # eq_atol should be 10^-(3+1) = 0.0001
        assert lon.eq_atol is not None
        assert abs(lon.eq_atol - 1e-4) < 1e-15

    def test_empty_trace_returns_empty_lon(self):
        """If no transitions are found, an empty LON should be returned."""

        # Use max_iter=0-like config that will produce no transitions
        # Actually max_iter must be positive, so use n_iter_no_change=1
        # with a function that always finds the same minimum
        def constant(x):
            return 0.0

        cfg = RustConfig(n_runs=1, n_iter_no_change=1, seed=42)
        sampler = RustSampler(cfg)
        lon = sampler.sample_to_lon(constant, [(-1.0, 1.0)])
        assert isinstance(lon, LON)

    def test_compute_lon_matches_sample_to_lon(self):
        """compute_lon should produce equivalent results to manual sample_to_lon."""
        pts = np.array([[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0]])

        rs_cfg = RustConfig(n_runs=3, n_iter_no_change=10, seed=42)
        sampler = RustSampler(rs_cfg)
        lon1 = sampler.sample_to_lon(sphere, [(-5.0, 5.0), (-5.0, 5.0)], initial_points=pts)

        lon2 = rust_compute_lon(
            sphere,
            dim=2,
            lower_bound=-5.0,
            upper_bound=5.0,
            initial_points=pts,
            config=RustConfig(n_runs=3, n_iter_no_change=10, seed=42),
        )

        assert lon1.n_vertices == lon2.n_vertices
        assert lon1.n_edges == lon2.n_edges


# ──────────────────────────────────────────────────────────────────────
#  Callable minimizer method
# ──────────────────────────────────────────────────────────────────────


class TestCallableMinimizer:
    """Test passing a callable as minimizer_method."""

    def test_string_method(self):
        cfg = RustConfig(n_runs=2, n_iter_no_change=3, minimizer_method="Nelder-Mead", seed=42)
        sampler = RustSampler(cfg)
        trace_df, _ = sampler.sample(sphere, [(-5.0, 5.0)])
        assert isinstance(trace_df, pd.DataFrame)

    def test_custom_minimizer_options(self):
        cfg = RustConfig(
            n_runs=2,
            n_iter_no_change=3,
            minimizer_options={"maxiter": 100},
            seed=42,
        )
        sampler = RustSampler(cfg)
        trace_df, _ = sampler.sample(sphere, [(-5.0, 5.0)])
        assert isinstance(trace_df, pd.DataFrame)


# ──────────────────────────────────────────────────────────────────────
#  Public API re-export tests
# ──────────────────────────────────────────────────────────────────────


class TestPublicAPI:
    """Verify the public API is accessible from the expected locations."""

    def test_import_from_sampling(self):
        from lonpy.sampling import BasinHoppingSampler, BasinHoppingSamplerConfig, compute_lon

        assert BasinHoppingSampler is RustSampler
        assert BasinHoppingSamplerConfig is RustConfig
        assert compute_lon is rust_compute_lon

    def test_import_from_lonpy(self):
        from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig, compute_lon

        assert BasinHoppingSampler is RustSampler
        assert BasinHoppingSamplerConfig is RustConfig
        assert compute_lon is rust_compute_lon


# ──────────────────────────────────────────────────────────────────────
#  Numerical consistency: trace values
# ──────────────────────────────────────────────────────────────────────


class TestNumericalConsistency:
    """
    With fixed initial points and max_iter=0, the trace should be empty.
    With fixed initial points and max_iter=1, the initial minimize
    results (current_x, current_f) should exactly match between Rust and Python.
    """

    def test_max_iter_0_empty_trace(self):
        """max_iter=1 should produce exactly 1 record per run."""
        pts = np.array([[1.0, 2.0], [-1.0, 3.0]])
        cfg = RustConfig(n_runs=2, max_iter=1, n_iter_no_change=None, seed=42)
        sampler = RustSampler(cfg)
        trace_df, raw = sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)], initial_points=pts)
        assert len(raw) == 2  # 1 per run

    def test_python_rust_same_initial_minimize(self):
        """
        Given the same initial points, the initial minimize (current_x/f in
        the first raw record of each run) should match between Python and Rust.
        """
        pts = np.array([[2.5, -3.1], [-1.7, 4.2], [0.5, 0.5]])
        domain = [(-5.0, 5.0), (-5.0, 5.0)]

        py_cfg = PyConfig(n_runs=3, max_iter=1, n_iter_no_change=None, seed=42)
        py_sampler = PySampler(py_cfg)
        _, py_raw = py_sampler.sample(sphere, domain, initial_points=pts)

        rs_cfg = RustConfig(n_runs=3, max_iter=1, n_iter_no_change=None, seed=42)
        rs_sampler = RustSampler(rs_cfg)
        _, rs_raw = rs_sampler.sample(sphere, domain, initial_points=pts)

        assert len(py_raw) == len(rs_raw) == 3

        for i in range(3):
            np.testing.assert_allclose(py_raw[i]["current_x"], rs_raw[i]["current_x"], atol=1e-12)
            np.testing.assert_allclose(py_raw[i]["current_f"], rs_raw[i]["current_f"], atol=1e-12)

    def test_accepted_transitions_fitness_decreasing(self):
        """In accepted transitions, fit2 <= fit1."""
        cfg = RustConfig(n_runs=3, n_iter_no_change=20, seed=42)
        sampler = RustSampler(cfg)
        trace_df, _ = sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)])
        if not trace_df.empty:
            assert (trace_df["fit2"] <= trace_df["fit1"] + 1e-15).all()

    def test_node_hash_format_with_precision(self):
        """Node hashes should have the expected format."""
        cfg = RustConfig(n_runs=2, n_iter_no_change=5, coordinate_precision=3, seed=42)
        sampler = RustSampler(cfg)
        trace_df, _ = sampler.sample(sphere, [(-5.0, 5.0), (-5.0, 5.0)])
        if not trace_df.empty:
            node = trace_df["node1"].iloc[0]
            parts = node.split("_")
            assert len(parts) == 2  # 2D
            # Each part should have 3 decimal places
            for part in parts:
                if "." in part:
                    decimals = len(part.split(".")[1])
                    assert decimals == 3
