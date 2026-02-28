# Plan: Dual Stopping Criteria for BasinHopping Inner Loop

The goal is to add `max_iterations` as an alternative (or complementary) stopping criterion alongside the existing `max_perturbations_without_improvement`. The cleanest model — directly inspired by scikit-learn — is to make both parameters independently optional (`None` = disabled), where the `while` loop stops as soon as **any enabled condition** is met. This mirrors how sklearn's `SGDClassifier` and `MLPClassifier` combine `max_iter` (hard cap) with `n_iter_no_change` (patience-based early stopping): both guard independently, neither implies the other.

**Naming proposal:** `max_iterations` over `n_iterations` — it keeps the `max_` prefix convention already used in the class and clearly conveys "ceiling", not "exact count".

---

## How "both" works

```
while True:
    if max_iterations is not None and run_index >= max_iterations:
        break   # hard cap reached
    if max_perturbations_without_improvement is not None
            and patience_counter >= max_perturbations_without_improvement:
        break   # no improvement for too long
    # … perturbation + minimization …
```

The loop exits on the first condition that fires. This is the standard scikit-learn OR-semantics for multiple termination guards.

---

## Scikit-learn precedents

| sklearn estimator | Hard-cap param | Patience param | Both active? |
|---|---|---|---|
| `SGDClassifier` | `max_iter` | `n_iter_no_change` | Yes — stop on first |
| `MLPClassifier` | `max_iter` | `n_iter_no_change` | Yes — stop on first |
| `GradientBoostingClassifier` | `n_estimators` | `n_iter_no_change` | Yes — stop on first |
| `EarlyStopping` (keras callback) | `max_queue_size` | `patience` | Yes — stop on first |

Usage examples that would be possible after the change:

```python
# Current behaviour preserved (patience only):
BasinHoppingSamplerConfig(max_perturbations_without_improvement=1000)

# Fixed budget (hard cap only, no patience):
BasinHoppingSamplerConfig(max_iterations=500, max_perturbations_without_improvement=None)

# Both — stop as soon as either fires (sklearn pattern):
BasinHoppingSamplerConfig(max_iterations=500, max_perturbations_without_improvement=200)

# Neither — invalid, caught at construction:
BasinHoppingSamplerConfig(max_iterations=None, max_perturbations_without_improvement=None)
# → ValueError: at least one stopping criterion must be set
```

---

## Steps

1. In `src/lonpy/sampling.py`, change `max_perturbations_without_improvement: int = 1000` to `max_perturbations_without_improvement: int | None = 1000` (no default behaviour change).

2. Add `max_iterations: int | None = None` field to `BasinHoppingSamplerConfig`, directly below `n_runs`. Update the docstring to document both fields and the OR-semantics.

3. Add `__post_init__` to `BasinHoppingSamplerConfig` that raises `ValueError` if both are `None`.

4. In `_basin_hopping_sampling` replace the `while perturbations_without_improvement < ...` loop with a `while True:` that checks both guards at the top (as shown above). The patience counter is still only incremented/reset when `max_perturbations_without_improvement` is active (not `None`), otherwise it is never consulted.

5. Update `compute_lon` flat signature: add `max_iterations: int | None = None`, change `max_perturbations_without_improvement` to `int | None = 1000`, pass both through to `BasinHoppingSamplerConfig`.

6. Update docstrings: `BasinHoppingSamplerConfig`, `BasinHoppingSampler`, and `compute_lon` to document the new field, the OR-semantics, and the `ValueError` guard.

---

## Verification

- Instantiate with only `max_perturbations_without_improvement` → existing tests pass unchanged.
- Instantiate with only `max_iterations=N` → each run does exactly N inner iterations.
- Instantiate with both → run terminates on whichever fires first; verify with a flat function (patience never fires) and a strictly decreasing sequence (hard cap fires first).
- Instantiate with both `None` → `ValueError` raised immediately.
- Run `python -m pytest` (or equivalent) after changes.

---

## Decisions

- Chose `max_iterations` over `n_iterations` to match the `max_` prefix convention already in `max_perturbations_without_improvement`.
- Chose OR-semantics (stop on first) over AND-semantics (stop only when both fire), following scikit-learn's established pattern where each guard is an independent ceiling.
- `max_perturbations_without_improvement` default stays `1000` (not changed to `None`) for full backward compatibility; users who want hard-cap-only must explicitly pass `max_perturbations_without_improvement=None`.
- Validation placed in `__post_init__` (dataclass hook) so it fires at construction, not at sampling time, matching scikit-learn's fail-fast philosophy.
