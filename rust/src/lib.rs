use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use numpy::ndarray::{Array1, Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// ──────────────────────────────────────────────────────────────────────
//  Raw record produced by one perturbation step
// ──────────────────────────────────────────────────────────────────────

fn raw_record_to_pydict<'py>(
    py: Python<'py>,
    run: usize,
    iteration: usize,
    current_x: &Array1<f64>,
    current_f: f64,
    new_x: &Array1<f64>,
    new_f: f64,
    accepted: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("run", run)?;
    d.set_item("iteration", iteration)?;
    d.set_item("current_x", current_x.to_pyarray(py))?;
    d.set_item("current_f", current_f)?;
    d.set_item("new_x", new_x.to_pyarray(py))?;
    d.set_item("new_f", new_f)?;
    d.set_item("accepted", accepted)?;
    Ok(d)
}

// ──────────────────────────────────────────────────────────────────────
//  Helper: call scipy.optimize.minimize from Rust
// ──────────────────────────────────────────────────────────────────────

struct MinimizeResult {
    x: Array1<f64>,
    fun: f64,
}

fn call_scipy_minimize<'py>(
    py: Python<'py>,
    scipy_minimize: &Bound<'py, PyAny>,
    func: &Bound<'py, PyAny>,
    x0: &Array1<f64>,
    method: &Bound<'py, PyAny>,
    options: &Bound<'py, PyAny>,
    bounds: Option<&Bound<'py, PyAny>>,
) -> PyResult<MinimizeResult> {
    let x0_py = x0.to_pyarray(py);
    let kwargs = PyDict::new(py);
    kwargs.set_item("method", method)?;
    kwargs.set_item("options", options)?;
    if let Some(b) = bounds {
        kwargs.set_item("bounds", b)?;
    } else {
        kwargs.set_item("bounds", py.None())?;
    }
    let res = scipy_minimize.call((func, x0_py), Some(&kwargs))?;
    let x_py: Bound<'_, PyAny> = res.getattr("x")?;
    let fun: f64 = res.getattr("fun")?.extract()?;

    let x_arr: PyReadonlyArray1<f64> = x_py.extract()?;
    let x = x_arr.as_array().to_owned();

    Ok(MinimizeResult { x, fun })
}

// ──────────────────────────────────────────────────────────────────────
//  BasinHoppingSamplerConfig
// ──────────────────────────────────────────────────────────────────────

#[pyclass(module = "lonpy._lonpy_rust")]
struct BasinHoppingSamplerConfig {
    #[pyo3(get, set)]
    n_runs: usize,
    #[pyo3(get, set)]
    n_iter_no_change: Option<usize>,
    #[pyo3(get, set)]
    max_iter: Option<usize>,
    #[pyo3(get, set)]
    step_mode: String,
    #[pyo3(get, set)]
    step_size: f64,
    #[pyo3(get, set)]
    fitness_precision: Option<i32>,
    #[pyo3(get, set)]
    coordinate_precision: Option<i32>,
    #[pyo3(get, set)]
    bounded: bool,
    #[pyo3(get, set)]
    minimizer_method: PyObject,
    #[pyo3(get, set)]
    minimizer_options: PyObject,
    #[pyo3(get, set)]
    seed: Option<u64>,
}

#[pymethods]
impl BasinHoppingSamplerConfig {
    #[new]
    #[pyo3(signature = (
        n_runs = 100,
        n_iter_no_change = Some(1000),
        max_iter = None,
        step_mode = "fixed".to_string(),
        step_size = 0.01,
        fitness_precision = None,
        coordinate_precision = Some(5),
        bounded = true,
        minimizer_method = None,
        minimizer_options = None,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        n_runs: usize,
        n_iter_no_change: Option<usize>,
        max_iter: Option<usize>,
        step_mode: String,
        step_size: f64,
        fitness_precision: Option<i32>,
        coordinate_precision: Option<i32>,
        bounded: bool,
        minimizer_method: Option<PyObject>,
        minimizer_options: Option<PyObject>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if let Some(n) = n_iter_no_change {
            if n == 0 {
                return Err(PyValueError::new_err(
                    "n_iter_no_change must be positive or None.",
                ));
            }
        }
        if let Some(m) = max_iter {
            if m == 0 {
                return Err(PyValueError::new_err("max_iter must be positive or None."));
            }
        }
        if n_iter_no_change.is_none() && max_iter.is_none() {
            return Err(PyValueError::new_err(
                "At least one stopping criterion must be set: n_iter_no_change and/or max_iter.",
            ));
        }

        let method = match minimizer_method {
            Some(m) => {
                if m.bind(py).is_none() {
                    "L-BFGS-B".into_pyobject(py)?.unbind().into()
                } else {
                    m
                }
            }
            None => "L-BFGS-B".into_pyobject(py)?.unbind().into(),
        };

        let opts = match minimizer_options {
            Some(o) => {
                if o.bind(py).is_none() {
                    let d = PyDict::new(py);
                    d.set_item("ftol", 1e-07)?;
                    d.set_item("gtol", 0)?;
                    d.set_item("maxiter", 15000)?;
                    d.unbind().into()
                } else {
                    o
                }
            }
            None => {
                let d = PyDict::new(py);
                d.set_item("ftol", 1e-07)?;
                d.set_item("gtol", 0)?;
                d.set_item("maxiter", 15000)?;
                d.unbind().into()
            }
        };

        Ok(BasinHoppingSamplerConfig {
            n_runs,
            n_iter_no_change,
            max_iter,
            step_mode,
            step_size,
            fitness_precision,
            coordinate_precision,
            bounded,
            minimizer_method: method,
            minimizer_options: opts,
            seed,
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let method_repr: String = self.minimizer_method.bind(py).repr()?.extract()?;
        let opts_repr: String = self.minimizer_options.bind(py).repr()?.extract()?;
        Ok(format!(
            "BasinHoppingSamplerConfig(n_runs={}, n_iter_no_change={:?}, max_iter={:?}, \
             step_mode='{}', step_size={}, fitness_precision={:?}, \
             coordinate_precision={:?}, bounded={}, minimizer_method={}, \
             minimizer_options={}, seed={:?})",
            self.n_runs,
            self.n_iter_no_change,
            self.max_iter,
            self.step_mode,
            self.step_size,
            self.fitness_precision,
            self.coordinate_precision,
            self.bounded,
            method_repr,
            opts_repr,
            self.seed,
        ))
    }
}

// ──────────────────────────────────────────────────────────────────────
//  Pure-Rust helpers (no Python calls)
// ──────────────────────────────────────────────────────────────────────

fn round_value(value: f64, precision: Option<i32>) -> f64 {
    match precision {
        Some(p) if p >= 0 => {
            let factor = 10f64.powi(p);
            (value * factor).round() / factor
        }
        _ => value,
    }
}

fn round_array(arr: &Array1<f64>, precision: Option<i32>) -> Array1<f64> {
    match precision {
        Some(p) if p >= 0 => {
            let factor = 10f64.powi(p);
            arr.mapv(|v| (v * factor).round() / factor)
        }
        _ => arr.clone(),
    }
}

fn hash_solution(x: &Array1<f64>, precision: Option<i32>) -> String {
    let x_clean: Array1<f64> = x.mapv(|v| v + 0.0); // -0.0 -> 0.0

    match precision {
        Some(p) if p >= 0 => {
            let p_usize = p as usize;
            let mut parts: Vec<String> = Vec::with_capacity(x_clean.len());
            for &v in x_clean.iter() {
                parts.push(format!("{:.prec$}", v, prec = p_usize));
            }
            parts.join("_")
        }
        _ => {
            let mut parts: Vec<String> = Vec::with_capacity(x_clean.len());
            for &v in x_clean.iter() {
                parts.push(v.to_string());
            }
            parts.join("_")
        }
    }
}

fn perturbation(
    rng: &mut ChaCha8Rng,
    x: &Array1<f64>,
    p: &Array1<f64>,
    bounds: Option<&Array2<f64>>,
    bounded: bool,
) -> Array1<f64> {
    let n = x.len();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let delta: f64 = rng.random_range(-p[i]..=p[i]);
        y[i] = x[i] + delta;
    }
    if bounded {
        if let Some(b) = bounds {
            for i in 0..n {
                y[i] = y[i].clamp(b[[i, 0]], b[[i, 1]]);
            }
        }
    }
    y
}

fn compute_step_sizes(
    step_mode: &str,
    step_size: f64,
    domain_arr: &Array2<f64>,
    n_var: usize,
) -> Array1<f64> {
    if step_mode == "percentage" {
        let mut p = Array1::<f64>::zeros(n_var);
        for i in 0..n_var {
            p[i] = step_size * (domain_arr[[i, 1]] - domain_arr[[i, 0]]).abs();
        }
        p
    } else {
        Array1::from_elem(n_var, step_size)
    }
}

// ──────────────────────────────────────────────────────────────────────
//  Snapshot of config fields needed during sampling (avoids repeated borrows)
// ──────────────────────────────────────────────────────────────────────

struct ConfigSnapshot {
    n_runs: usize,
    n_iter_no_change: Option<usize>,
    max_iter: Option<usize>,
    step_mode: String,
    step_size: f64,
    fitness_precision: Option<i32>,
    coordinate_precision: Option<i32>,
    bounded: bool,
    seed: Option<u64>,
}

impl ConfigSnapshot {
    fn from_pyref(cfg: &BasinHoppingSamplerConfig) -> Self {
        ConfigSnapshot {
            n_runs: cfg.n_runs,
            n_iter_no_change: cfg.n_iter_no_change,
            max_iter: cfg.max_iter,
            step_mode: cfg.step_mode.clone(),
            step_size: cfg.step_size,
            fitness_precision: cfg.fitness_precision,
            coordinate_precision: cfg.coordinate_precision,
            bounded: cfg.bounded,
            seed: cfg.seed,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
//  Trace record (for building the DataFrame in Python)
// ──────────────────────────────────────────────────────────────────────

struct TraceRecord {
    run: usize,
    fit1: f64,
    node1: String,
    fit2: f64,
    node2: String,
}

// ──────────────────────────────────────────────────────────────────────
//  BasinHoppingSampler
// ──────────────────────────────────────────────────────────────────────

#[pyclass(module = "lonpy._lonpy_rust")]
struct BasinHoppingSampler {
    config: Py<BasinHoppingSamplerConfig>,
}

#[pymethods]
impl BasinHoppingSampler {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<Py<BasinHoppingSamplerConfig>>, py: Python<'_>) -> PyResult<Self> {
        let config = match config {
            Some(c) => c,
            None => {
                let default_config = BasinHoppingSamplerConfig::new(
                    py,
                    100,
                    Some(1000),
                    None,
                    "fixed".to_string(),
                    0.01,
                    None,
                    Some(5),
                    true,
                    None,
                    None,
                    None,
                )?;
                Py::new(py, default_config)?
            }
        };
        Ok(BasinHoppingSampler { config })
    }

    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> Bound<'py, BasinHoppingSamplerConfig> {
        self.config.bind(py).clone()
    }

    /// Run Basin-Hopping sampling and construct trace data.
    ///
    /// Returns: (trace_df, raw_records)
    #[pyo3(signature = (func, domain, initial_points=None, progress_callback=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        func: Bound<'py, PyAny>,
        domain: Vec<(f64, f64)>,
        initial_points: Option<Bound<'py, PyAny>>,
        progress_callback: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        // Snapshot config fields so we don't borrow across Python calls
        let snap: ConfigSnapshot;
        let method_obj: PyObject;
        let options_obj: PyObject;
        {
            let cfg = self.config.borrow(py);
            snap = ConfigSnapshot::from_pyref(&cfg);
            method_obj = cfg.minimizer_method.clone_ref(py);
            options_obj = cfg.minimizer_options.clone_ref(py);
        }

        let n_var = domain.len();
        let n_runs = snap.n_runs;

        // Build domain array (n_var x 2)
        let mut domain_arr = Array2::<f64>::zeros((n_var, 2));
        for (i, &(lo, hi)) in domain.iter().enumerate() {
            domain_arr[[i, 0]] = lo;
            domain_arr[[i, 1]] = hi;
        }

        // Resolve initial points
        let resolved_pts = resolve_initial_points(
            py,
            &initial_points,
            &domain_arr,
            n_runs,
            n_var,
            snap.bounded,
        )?;

        // Compute step size
        let p = compute_step_sizes(&snap.step_mode, snap.step_size, &domain_arr, n_var);

        // Build bounds for scipy.optimize.minimize
        let bounds_for_minimize: Option<Bound<'py, PyAny>> = if snap.bounded {
            let bounds_list = PyList::empty(py);
            for &(lo, hi) in &domain {
                let tup = PyTuple::new(py, &[lo, hi])?;
                bounds_list.append(tup)?;
            }
            Some(bounds_list.into_any())
        } else {
            None
        };

        let bounds_arr: Option<Array2<f64>> = if snap.bounded {
            Some(domain_arr.clone())
        } else {
            None
        };

        // Import scipy.optimize.minimize and warnings
        let scipy_opt = py.import("scipy.optimize")?;
        let scipy_minimize = scipy_opt.getattr("minimize")?;
        let warnings_mod = py.import("warnings")?;
        let warn_fn = warnings_mod.getattr("warn")?;

        let method = method_obj.bind(py);
        let options = options_obj.bind(py);

        // Create RNG
        let mut rng = match snap.seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_os_rng(),
        };

        // Generate random initial points if not provided
        let initial_pts = if initial_points.is_none() {
            let mut pts = Array2::<f64>::zeros((n_runs, n_var));
            for r in 0..n_runs {
                for d in 0..n_var {
                    let lo = domain_arr[[d, 0]];
                    let hi = domain_arr[[d, 1]];
                    pts[[r, d]] = rng.random_range(lo..=hi);
                }
            }
            pts
        } else {
            resolved_pts
        };

        let mut raw_records: Vec<Bound<'py, PyDict>> = Vec::new();
        let mut trace_records: Vec<TraceRecord> = Vec::new();

        let coord_prec = snap.coordinate_precision;
        let fit_prec = snap.fitness_precision;

        for run in 1..=n_runs {
            if let Some(ref cb) = progress_callback {
                cb.call1((run, n_runs))?;
            }

            // Initial minimize
            let x0 = initial_pts.row(run - 1).to_owned();
            let min_result = call_scipy_minimize(
                py,
                &scipy_minimize,
                &func,
                &x0,
                method,
                options,
                bounds_for_minimize.as_ref(),
            );

            let (mut current_x, mut current_f) = match min_result {
                Ok(r) => (r.x, r.fun),
                Err(e) => {
                    if e.is_instance_of::<PyValueError>(py) {
                        let msg = format!(
                            "Run {}: initial minimize failed with ValueError: {}. \
                             Starting point: {:?}. Skipping run.",
                            run, e, x0,
                        );
                        warn_fn.call1((msg,))?;
                        continue;
                    } else {
                        return Err(e);
                    }
                }
            };

            let mut iters_without_improvement: usize = 0;
            let mut iter_index: usize = 0;

            loop {
                if let Some(max_it) = snap.max_iter {
                    if iter_index >= max_it {
                        break;
                    }
                }
                if let Some(nic) = snap.n_iter_no_change {
                    if iters_without_improvement >= nic {
                        break;
                    }
                }

                let x_perturbed =
                    perturbation(&mut rng, &current_x, &p, bounds_arr.as_ref(), snap.bounded);

                let min_result = call_scipy_minimize(
                    py,
                    &scipy_minimize,
                    &func,
                    &x_perturbed,
                    method,
                    options,
                    bounds_for_minimize.as_ref(),
                );

                let (new_x, new_f) = match min_result {
                    Ok(r) => (r.x, r.fun),
                    Err(e) => {
                        if e.is_instance_of::<PyValueError>(py) {
                            let msg = format!(
                                "Run {}, iteration {}: minimize after perturbation \
                                 failed with ValueError: {}. \
                                 Perturbed point: {:?}. Skipping perturbation.",
                                run, iter_index, e, x_perturbed,
                            );
                            warn_fn.call1((msg,))?;
                            iters_without_improvement += 1;
                            iter_index += 1;
                            continue;
                        } else {
                            return Err(e);
                        }
                    }
                };

                let accepted = new_f <= current_f;

                raw_records.push(raw_record_to_pydict(
                    py, run, iter_index, &current_x, current_f, &new_x, new_f, accepted,
                )?);

                if accepted {
                    let from_x_rounded = round_array(&current_x, coord_prec);
                    let to_x_rounded = round_array(&new_x, coord_prec);
                    let node1 = hash_solution(&from_x_rounded, coord_prec);
                    let node2 = hash_solution(&to_x_rounded, coord_prec);
                    let fit1 = round_value(current_f, fit_prec);
                    let fit2 = round_value(new_f, fit_prec);

                    trace_records.push(TraceRecord {
                        run,
                        fit1,
                        node1,
                        fit2,
                        node2,
                    });
                }

                if snap.n_iter_no_change.is_some() {
                    if new_f < current_f {
                        iters_without_improvement = 0;
                    } else {
                        iters_without_improvement += 1;
                    }
                }

                if new_f <= current_f {
                    current_x = new_x;
                    current_f = new_f;
                }

                iter_index += 1;
            }
        }

        // Build trace DataFrame
        let pd = py.import("pandas")?;
        let trace_df = if trace_records.is_empty() {
            let cols = PyList::new(py, &["run", "fit1", "node1", "fit2", "node2"])?;
            let kw = PyDict::new(py);
            kw.set_item("columns", cols)?;
            pd.getattr("DataFrame")?.call((), Some(&kw))?
        } else {
            let n = trace_records.len();
            let mut runs = Vec::with_capacity(n);
            let mut fit1s = Vec::with_capacity(n);
            let mut node1s = Vec::with_capacity(n);
            let mut fit2s = Vec::with_capacity(n);
            let mut node2s = Vec::with_capacity(n);
            for tr in &trace_records {
                runs.push(tr.run);
                fit1s.push(tr.fit1);
                node1s.push(tr.node1.clone());
                fit2s.push(tr.fit2);
                node2s.push(tr.node2.clone());
            }
            let data = PyDict::new(py);
            data.set_item("run", runs)?;
            data.set_item("fit1", fit1s)?;
            data.set_item("node1", node1s)?;
            data.set_item("fit2", fit2s)?;
            data.set_item("node2", node2s)?;
            pd.getattr("DataFrame")?.call1((data,))?
        };

        // Reorder columns
        let cols = PyList::new(py, &["run", "fit1", "node1", "fit2", "node2"])?;
        let trace_df = trace_df.get_item(cols)?;

        let raw_list = PyList::new(py, &raw_records)?;

        Ok((trace_df.unbind(), raw_list.into_any().unbind()))
    }

    /// sample_to_lon(func, domain, ...)
    #[pyo3(signature = (func, domain, initial_points=None, progress_callback=None, lon_config=None))]
    fn sample_to_lon<'py>(
        &self,
        py: Python<'py>,
        func: Bound<'py, PyAny>,
        domain: Vec<(f64, f64)>,
        initial_points: Option<Bound<'py, PyAny>>,
        progress_callback: Option<Bound<'py, PyAny>>,
        lon_config: Option<Bound<'py, PyAny>>,
    ) -> PyResult<PyObject> {
        let (trace_df_obj, _raw) =
            self.sample(py, func, domain, initial_points, progress_callback)?;
        let trace_df = trace_df_obj.bind(py);

        let is_empty: bool = trace_df.getattr("empty")?.extract()?;
        if is_empty {
            let lon_mod = py.import("lonpy.lon")?;
            let lon_cls = lon_mod.getattr("LON")?;
            return Ok(lon_cls.call0()?.unbind());
        }

        let lon_mod = py.import("lonpy.lon")?;
        let lon_cls = lon_mod.getattr("LON")?;
        let lon_config_cls = lon_mod.getattr("LONConfig")?;

        let effective_config = if let Some(cfg) = lon_config {
            let dc_mod = py.import("dataclasses")?;
            dc_mod.getattr("replace")?.call1((&cfg,))?
        } else {
            lon_config_cls.call0()?
        };

        // Set eq_atol from fitness_precision if not already set
        let eq_atol = effective_config.getattr("eq_atol")?;
        if eq_atol.is_none() {
            let fit_prec = self.config.borrow(py).fitness_precision;
            if let Some(p) = fit_prec {
                if p >= 0 {
                    let atol = 10f64.powi(-(p + 1));
                    effective_config.setattr("eq_atol", atol)?;
                }
            }
        }

        let kw = PyDict::new(py);
        kw.set_item("config", effective_config)?;
        let result = lon_cls.call_method("from_trace_data", (trace_df,), Some(&kw))?;

        Ok(result.unbind())
    }
}

// ──────────────────────────────────────────────────────────────────────
//  Standalone helper: resolve initial points
// ──────────────────────────────────────────────────────────────────────

fn resolve_initial_points(
    py: Python<'_>,
    initial_points: &Option<Bound<'_, PyAny>>,
    domain_arr: &Array2<f64>,
    n_runs: usize,
    n_var: usize,
    bounded: bool,
) -> PyResult<Array2<f64>> {
    match initial_points {
        None => Ok(Array2::zeros((n_runs, n_var))),
        Some(pts) => {
            let np = py.import("numpy")?;
            let kw = PyDict::new(py);
            kw.set_item("dtype", np.getattr("float64")?)?;
            let arr = np.call_method("asarray", (pts,), Some(&kw))?;

            let arr: PyReadonlyArray2<f64> = arr.extract()?;
            let shape = arr.shape();

            if shape.len() != 2 || shape[1] != n_var {
                return Err(PyValueError::new_err(format!(
                    "initial_points must have shape (n_runs, {}), got {:?}.",
                    n_var, shape,
                )));
            }
            if shape[0] != n_runs {
                return Err(PyValueError::new_err(format!(
                    "initial_points has {} points, but n_runs is {}. These must match.",
                    shape[0], n_runs,
                )));
            }

            let arr_owned = arr.as_array().to_owned();

            if bounded {
                for d in 0..n_var {
                    let lo = domain_arr[[d, 0]];
                    let hi = domain_arr[[d, 1]];
                    for r in 0..n_runs {
                        if arr_owned[[r, d]] < lo || arr_owned[[r, d]] > hi {
                            return Err(PyValueError::new_err(
                                "initial_points contains values outside the domain bounds. \
                                 All points must satisfy lower_bound <= x <= upper_bound \
                                 when bounded=True.",
                            ));
                        }
                    }
                }
            }

            Ok(arr_owned)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
//  compute_lon top-level function
// ──────────────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (func, dim, lower_bound, upper_bound, initial_points=None, config=None, lon_config=None))]
fn compute_lon<'py>(
    py: Python<'py>,
    func: Bound<'py, PyAny>,
    dim: usize,
    lower_bound: Bound<'py, PyAny>,
    upper_bound: Bound<'py, PyAny>,
    initial_points: Option<Bound<'py, PyAny>>,
    config: Option<Py<BasinHoppingSamplerConfig>>,
    lon_config: Option<Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let lower_bounds: Vec<f64> = if let Ok(v) = lower_bound.extract::<f64>() {
        vec![v; dim]
    } else {
        lower_bound.extract::<Vec<f64>>()?
    };

    let upper_bounds: Vec<f64> = if let Ok(v) = upper_bound.extract::<f64>() {
        vec![v; dim]
    } else {
        upper_bound.extract::<Vec<f64>>()?
    };

    if lower_bounds.len() != dim || upper_bounds.len() != dim {
        return Err(PyValueError::new_err(
            "lower_bound and upper_bound must have length equal to dim.",
        ));
    }

    let domain: Vec<(f64, f64)> = lower_bounds.into_iter().zip(upper_bounds).collect();

    let sampler = BasinHoppingSampler::new(config, py)?;
    sampler.sample_to_lon(py, func, domain, initial_points, None, lon_config)
}

// ──────────────────────────────────────────────────────────────────────
//  Module definition
// ──────────────────────────────────────────────────────────────────────

#[pymodule]
fn _lonpy_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BasinHoppingSamplerConfig>()?;
    m.add_class::<BasinHoppingSampler>()?;
    m.add_function(wrap_pyfunction!(compute_lon, m)?)?;
    Ok(())
}
