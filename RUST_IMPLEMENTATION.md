# Rust Implementation of `lonpy.sampling`

This document describes the rewrite of the `lonpy.sampling` module from pure Python to Rust using [PyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs/). The public API is unchanged -- existing code that imports from `lonpy` or `lonpy.sampling` works without modification.

## Table of Contents

- [Overview](#overview)
- [What Changed](#what-changed)
- [Architecture](#architecture)
- [Building](#building)
- [Development Workflow](#development-workflow)
- [Running Tests](#running-tests)
- [Design Decisions](#design-decisions)
- [File Reference](#file-reference)

---

## Overview

The `sampling.py` module implements Basin-Hopping sampling for constructing Local Optima Networks (LONs). The core loop involves:

1. Generating initial points in the search domain
2. Running `scipy.optimize.minimize` to find local optima
3. Perturbing solutions and re-minimizing
4. Tracking transitions between local optima (the "trace")
5. Constructing a LON graph from the trace data

The Rust rewrite moves the loop control, perturbation, rounding, hashing, and trace construction logic into compiled Rust code. The heavy numerical work (`scipy.optimize.minimize`) is still called back into Python, since scipy has no Rust equivalent.

### What stays in Python

- `scipy.optimize.minimize` -- called from Rust via PyO3 callbacks
- User-supplied objective functions (`func` parameter)
- pandas DataFrame construction for the trace output
- LON graph construction (`lonpy.lon.LON.from_trace_data`)

### What runs in Rust

- The Basin-Hopping main loop (iteration, stopping criteria)
- Perturbation of candidate solutions
- Solution rounding (`round_value`, `round_array`)
- Solution hashing (`hash_solution`)
- Step size computation (`compute_step_sizes`)
- Random number generation (ChaCha8 via `rand_chacha`)
- Input validation (bounds checking, shape checking)
- Config management (`BasinHoppingSamplerConfig`)

---

## What Changed

### Modified files

| File | Change |
|---|---|
| `pyproject.toml` | Build backend changed from `hatchling` to `maturin`. Added `[tool.maturin]` section. |
| `src/lonpy/sampling.py` | Rewritten as a thin re-export wrapper importing from `lonpy._lonpy_rust`. |

### New files

| File | Purpose |
|---|---|
| `rust/Cargo.toml` | Rust package configuration (PyO3, numpy, rand, rand_chacha dependencies) |
| `rust/src/lib.rs` | Main Rust implementation (~817 lines) |
| `src/lonpy/_sampling_py.py` | Copy of the original pure-Python implementation, preserved for comparison testing |
| `tests/test_sampling_rust.py` | 45 tests verifying the Rust implementation |

### Unchanged files

| File | Notes |
|---|---|
| `src/lonpy/__init__.py` | Public exports (`BasinHoppingSampler`, `BasinHoppingSamplerConfig`, `compute_lon`) continue to work through the new `sampling.py` wrapper |
| `src/lonpy/lon.py` | LON/LONConfig/CMLON classes -- untouched |
| All other modules | No changes |

---

## Architecture

```
User code
    |
    v
lonpy/__init__.py          (re-exports public API)
    |
    v
lonpy/sampling.py          (thin wrapper: re-exports from _lonpy_rust)
    |
    v
lonpy/_lonpy_rust.so       (compiled Rust extension module)
    |
    +---> Pure Rust: perturbation, rounding, hashing, loop control, RNG
    |
    +---> Python callbacks: scipy.optimize.minimize, user objective function
    |
    v
lonpy/lon.py               (LON.from_trace_data builds the graph)
```

### Key Rust types

- **`BasinHoppingSamplerConfig`** (`#[pyclass]`) -- All configuration fields with getters/setters. Constructor validates inputs and sets defaults identical to the original Python dataclass.

- **`BasinHoppingSampler`** (`#[pyclass]`) -- Holds a `Py<BasinHoppingSamplerConfig>`. Exposes `sample()` and `sample_to_lon()` methods.

- **`ConfigSnapshot`** (internal) -- Plain Rust struct that copies config values before the sampling loop. This avoids holding a Python borrow (`PyRef`) across calls back into Python (which would violate PyO3's borrow rules).

- **`TraceRecord`** (internal) -- Accumulates trace data in Rust memory before converting to a pandas DataFrame at the end.

### RNG difference

The Python implementation uses `numpy.random.default_rng` (PCG64). The Rust implementation uses `ChaCha8Rng` from `rand_chacha`. This means that even with the same seed, the sequence of random perturbations will differ. The deterministic behavior (same seed = same results) is preserved within each implementation, but results won't match between Python and Rust for the random perturbation steps.

When fixed `initial_points` are provided, the initial `scipy.optimize.minimize` calls produce identical results (since no randomness is involved). The divergence starts at the first perturbation step.

---

## Building

### Prerequisites

- **Rust** (stable, 1.70+): Install via [rustup](https://rustup.rs/)
- **Python** (3.10+): With a virtual environment
- **maturin**: Installed as a Python package (`pip install maturin`)

### Development build (debug mode)

```bash
# Activate the virtual environment
source .venv/bin/activate

# Build and install in development mode
maturin develop
```

This compiles the Rust code in debug mode and installs the extension module into the virtual environment. Fast to compile, slower at runtime.

### Release build (optimized)

```bash
source .venv/bin/activate

# Build with optimizations
maturin develop --release
```

This enables Rust compiler optimizations (`-O3`, LTO, etc.). Slower to compile, but the resulting module runs at full speed. **Always use `--release` for benchmarks or production.**

### Using uv (this project's package manager)

If using `uv` instead of pip/maturin directly:

```bash
# Install all dependencies including the Rust extension
uv sync

# Or with dev dependencies
uv sync --dev
```

`uv` reads `pyproject.toml`, detects the maturin build backend, and handles the Rust compilation automatically.

### Building a wheel

```bash
# Build a distributable wheel
maturin build --release

# The wheel will be in target/wheels/
```

### Verifying the build

```python
# Quick check that the Rust module loads
python -c "from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig, compute_lon; print('OK')"

# Check that it's actually the Rust implementation
python -c "import lonpy._lonpy_rust; print(type(lonpy._lonpy_rust.BasinHoppingSampler))"
# Should print: <class 'type'>  (a PyO3 class, not a Python class)
```

---

## Development Workflow

### Edit-compile-test cycle

1. Edit `rust/src/lib.rs`
2. Run `maturin develop --release` (or without `--release` for faster iteration)
3. Run tests: `python -m pytest tests/test_sampling_rust.py -v`

### Common build errors

| Error | Cause | Fix |
|---|---|---|
| `cannot find -lpython3.x` | Python dev headers missing | Install `python3-dev` or use the venv Python |
| `pyo3 ... requires rustc >= 1.63` | Rust too old | `rustup update stable` |
| `error[E0502]: cannot borrow...` | Holding `PyRef` across a Python call | Extract values into a `ConfigSnapshot` first |

---

## Running Tests

```bash
source .venv/bin/activate

# Run all Rust implementation tests
python -m pytest tests/test_sampling_rust.py -v

# Run with output visible
python -m pytest tests/test_sampling_rust.py -v -s

# Run a specific test class
python -m pytest tests/test_sampling_rust.py::TestConfigDefaults -v

# Run all project tests
python -m pytest tests/ -v
```

### Test structure

The test file `tests/test_sampling_rust.py` contains 45 tests across 8 classes:

| Class | What it tests |
|---|---|
| `TestConfigDefaults` | Default values match between Rust and Python configs |
| `TestConfigValidation` | Error handling for invalid config parameters |
| `TestSampleOutput` | DataFrame columns, raw record structure, output types |
| `TestRustVsPython` | Identical initial minimize results when given fixed initial points |
| `TestEdgeCases` | Wrong shapes, out-of-bounds points, callbacks, bounded/unbounded, precision, dimensionality |
| `TestLONConstruction` | `sample_to_lon`, `compute_lon`, LONConfig integration, `eq_atol` derivation |
| `TestCallableMinimizer` | String vs callable minimizer methods |
| `TestPublicAPI` | Verifies re-exports from `lonpy` and `lonpy.sampling` |

---

## Design Decisions

### Why maturin?

Maturin is the standard build tool for PyO3 projects. It integrates with Python packaging (`pyproject.toml`), handles the Rust compilation, and produces standard wheels. It replaced `hatchling` as the build backend.

### Why ConfigSnapshot?

PyO3's `Py<T>` requires a GIL-bound borrow (`borrow(py)`) to access fields. This borrow cannot be held across calls back into Python (like `scipy.optimize.minimize`), because Python might trigger garbage collection or other operations that need mutable access.

The `ConfigSnapshot` pattern extracts all needed config values into a plain Rust struct before the loop starts, eliminating the borrow conflict entirely.

### Why ChaCha8 instead of numpy's RNG?

Using numpy's RNG from Rust would require calling back into Python for every random number, defeating the purpose of the Rust rewrite. ChaCha8 is a fast, cryptographically-inspired PRNG available in pure Rust. The tradeoff is that random sequences differ from the Python implementation, but deterministic reproducibility (same seed = same results) is preserved within the Rust implementation.

### Why keep scipy.optimize.minimize in Python?

There is no Rust equivalent of scipy's optimization suite. The minimize call is the computational bottleneck per iteration, so wrapping it in Rust wouldn't speed it up. The Rust speedup comes from everything around the minimize call: loop control, perturbation generation, solution hashing/rounding, and trace construction.

---

## File Reference

```
lonpy/
├── pyproject.toml                  # Build config (maturin backend)
├── rust/
│   ├── Cargo.toml                  # Rust dependencies
│   └── src/
│       └── lib.rs                  # Rust implementation (817 lines)
├── src/
│   └── lonpy/
│       ├── __init__.py             # Public API re-exports (unchanged)
│       ├── sampling.py             # Thin wrapper re-exporting from _lonpy_rust
│       ├── _sampling_py.py         # Original Python implementation (for testing)
│       └── lon.py                  # LON/LONConfig/CMLON (unchanged)
└── tests/
    └── test_sampling_rust.py       # 45 tests for the Rust implementation
```
