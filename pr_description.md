# PR: Add `max_iter` stopping criterion and refactor `sampling` API

## Description

This PR introduces a `max_iter` stopping criterion for Basin-Hopping sampling, which caps the total number of iterations per run, and renames `max_perturbations_without_improvement` to `n_iter_no_change` to align API naming with the scikit-learn convention.

It also simplifies the `compute_lon` public API by merging configuration parameters into a single `BasinHoppingSamplerConfig` object, consistent with how `BasinHoppingSampler` is already used directly.


## Fixes [#13](https://github.com/helix-agh/lonpy/issues/13)

## Changes made

### `src/lonpy/sampling.py`

#### `BasinHoppingSamplerConfig`

- Renamed `max_perturbations_without_improvement` to `n_iter_no_change` (`int | None`, default `1000`):
  - The new name aligns with the scikit-learn convention and describes the parameter more clearly (consecutive non-improving iterations, not raw perturbation count).
  - Made nullable (`None`) to allow disabling this criterion entirely when `max_iter` is set.

- Added `max_iter: int | None = None`:
  - A new optional cap on the total number of perturbation steps per run, regardless of improvement.

- Added `__post_init__` validation:
  - Raises `ValueError` if both `n_iter_no_change` and `max_iter` are `None`, ensuring at least one stopping criterion is always active.

- Improved docstrings for `minimizer_method` and `minimizer_options` to include cross-references to `scipy.optimize.minimize` documentation and clarify accepted argument types.

#### `BasinHoppingSampler._basin_hopping_sampling()`

- Replaced the fixed `while perturbations_without_improvement < ...` loop condition with `while True:` and explicit `break` statements for both stopping criteria.

#### `compute_lon()`

- Replaced the following flat parameters:
  - `seed`, `step_size`, `step_mode`, `n_runs`, `max_perturbations_without_improvement`, `fitness_precision`, `coordinate_precision`, `bounded`

  with a single `config: BasinHoppingSamplerConfig | None = None` parameter.

  This makes `compute_lon` consistent with `BasinHoppingSampler.sample_to_lon()` and removes the risk of the two APIs diverging during further development.

- Updated docstring accordingly.

### `README.md`

- Updated Quick Start example to use `BasinHoppingSamplerConfig` with `n_iter_no_change` and pass it via `config=` to `compute_lon()`.
- Updated Custom Sampling Configuration example to use `n_iter_no_change`.

### `examples/bioma/`

- Renamed `FunctionConfig.max_perturbations_without_improvement` field to `n_iter_no_change` in `utils.py`.
- Updated `BasinHoppingSamplerConfig` instantiation in `utils.py` to pass `n_iter_no_change=`.
- Updated all `FunctionConfig` definitions in `fig3.py`, `fig4.py`, and `fig6.py` to use `n_iter_no_change=`.

### Docs

- `docs/api/sampling.md`: Removed stale `members` entries (`hash_solution`, `fitness_to_int`, `bounded_perturbation`, `unbounded_perturbation`) from the `BasinHoppingSampler` autodoc block — these methods do not exist in the public API.
- `docs/user-guide/sampling.md`: Updated all parameter names, added `max_iter` to the parameters table and usage examples, fixed `compute_lon()` call in the Custom Initial Points section (was incorrectly passing `n_runs` and `seed` as top-level kwargs), updated Best Practices examples.
- `docs/user-guide/analysis.md`, `docs/user-guide/examples.md`, `docs/getting-started/quickstart.md`, `docs/index.md`: Updated all `compute_lon()` calls to use `config=BasinHoppingSamplerConfig(...)` pattern.
