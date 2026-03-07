# cython: language_level=3
import numpy as np
cimport numpy as cnp
import warnings
from scipy.optimize import minimize

def _single_bh_run(
    int run,
    object func,               # User's Python callback
    object initial_point,      # Kept as object/ndarray for Scipy compatibility
    object p,
    object bounds_array,
    object config,
    object seed
):
    """Run one independent Basin-Hopping chain and return its raw records."""
    
    # Typed local variables for faster loop control
    cdef int iters_without_improvement = 0
    cdef int iter_index = 0
    cdef double current_f, new_f
    cdef bint is_bounded = config.bounded and bounds_array is not None
    
    # Pre-extract config values to avoid Python attribute lookups in the loop
    cdef int max_iter = config.max_iter if config.max_iter is not None else -1
    cdef int n_iter_no_change = config.n_iter_no_change if config.n_iter_no_change is not None else -1
    cdef object minimizer_method = config.minimizer_method
    cdef object minimizer_options = config.minimizer_options
    cdef object bounds = bounds_array if is_bounded else None

    # Declare Python objects used in the loop
    cdef object rng = np.random.default_rng(seed)
    cdef list records = []
    cdef object res, current_x, new_x, y, x_perturbed

    try:
        res = minimize(
            func,
            initial_point,
            method=minimizer_method,
            options=minimizer_options,
            bounds=bounds,
        )
    except ValueError as e:
        warnings.warn(
            f"Run {run}: initial minimize failed with ValueError: {e}. "
            f"Starting point: {initial_point}. Skipping run.",
            stacklevel=2,
        )
        return records

    current_x = res.x
    current_f = res.fun

    while True:
        # Fast C-level integer comparisons
        if max_iter != -1 and iter_index >= max_iter:
            break
        if n_iter_no_change != -1 and iters_without_improvement >= n_iter_no_change:
            break

        y = current_x + rng.uniform(low=-p, high=p)
        
        if is_bounded:
            x_perturbed = np.clip(y, bounds_array[:, 0], bounds_array[:, 1])
        else:
            x_perturbed = y

        try:
            res = minimize(
                func,
                x_perturbed,
                method=minimizer_method,
                options=minimizer_options,
                bounds=bounds,
            )
        except ValueError as e:
            warnings.warn(
                f"Run {run}, iteration {iter_index}: minimize after perturbation "
                f"failed with ValueError: {e}. "
                f"Perturbed point: {x_perturbed}. Skipping perturbation.",
                stacklevel=2,
            )
            iters_without_improvement += 1
            iter_index += 1
            continue

        new_x = res.x
        new_f = res.fun

        records.append({
            "run": run,
            "iteration": iter_index,
            "current_x": current_x.copy(),
            "current_f": current_f,
            "new_x": new_x.copy(),
            "new_f": new_f,
            "accepted": new_f <= current_f,
        })

        if n_iter_no_change != -1:
            if new_f < current_f:
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1

        if new_f <= current_f:
            current_x = new_x.copy()
            current_f = new_f

        iter_index += 1

    return records