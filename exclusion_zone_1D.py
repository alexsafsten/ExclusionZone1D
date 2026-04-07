"""
Numerical tools for a 1D predator-prey reaction-diffusion model with an exclusion zone.

This module implements numerical experiments for the one-dimensional exclusion-zone
predator-prey model studied in the accompanying paper

    Berestycki, Fagan, and Safsten,
    "The Influence of Exclusion Zones on the Coexistence of Predator and Prey
    with an Allee Effect."
    
    https://arxiv.org/abs/2602.21414

Model overview
--------------
The prey population occupies the full interval (0, L), while the predator population
is confined to the subinterval (0, a), leaving the region (a, L) as a predator-free
exclusion zone. The prey obey bistable / strong-Allee-effect growth, and both species
diffuse in space with homogeneous Neumann boundary conditions. In the notation of
the paper, the 1D steady-state problem corresponds to equation (26), while the
time-dependent system is the one-dimensional analogue of the general model introduced
in Section 1.2. The exclusion zone can serve as a refuge for prey and, paradoxically,
may also support predator persistence by preventing over-exploitation of the prey.

This script provides three main user-facing functions:

    solve_EZ
        Solve the time-dependent 1D exclusion-zone model by finite differences in
        space and scipy.integrate.solve_ivp in time.

    plot_EZ
        Visualize a computed solution, including spatial population densities with
        an interactive time slider and total populations over time.

    a_sweep
        Sweep over values of the predator-domain length a and summarize long-time
        predator and prey population levels.

Numerical approach
------------------
Space is discretized by finite differences on a piecewise-uniform grid adapted to the
interface x = a. The resulting semi-discrete ODE system is integrated in time with
SciPy's stiff solvers (default: Radau), using an analytic sparse Jacobian.

Output conventions
------------------
The solver returns a dictionary containing spatial profiles, the spatial grid, time
values, and optionally total prey and predator populations. The prey density is
returned on the full grid. The predator density is also returned on the full grid,
with zeros in the predator-free region so that plotting and population integration
can be done directly on the same spatial mesh.

Intended use
------------
This code is intended as companion software for the paper and for exploratory
numerical experiments. It is written to be readable and modifiable by researchers
and students with some familiarity with reaction-diffusion equations and Python
scientific computing.

Typical workflow
----------------
1. Define a parameter dictionary `params`.
2. Call `solve_EZ(params, ...)` to compute a solution.
3. Call `plot_EZ(result)` to inspect the dynamics visually.
4. Call `a_sweep(params, a_list, ...)` to study how long-time behavior changes as
   the predator-domain size varies.

Note
----
This module is designed for the 1D setting. It is not a general-purpose PDE solver,
but rather a focused implementation of the exclusion-zone model used in the paper.
"""

import numpy as np
from scipy.sparse import dia_matrix
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

_EZ_TIME_SLIDER = None

def solve_EZ(
    params,
    tf=20,
    N1=200,
    N2=200,
    *,
    u0=1.0,
    v0=0.5,
    method="Radau",
    max_step=0.1,
    t_eval=None,
    rtol=1e-3,
    atol=1e-6,
    first_step=None,
    events=None,
    compute_total_pops=False,
    verbose=False,
    solve_ivp_kwargs=None,
):
    """
    Solve the 1D predator-prey exclusion-zone system using a finite-difference
    spatial discretization and SciPy's solve_ivp for time integration.

    Parameters
    ----------
    params : dict
        Model parameters. Required keys:
            'length'                    : total domain length L
            'a'                         : predator-occupied subdomain length
            'prey growth rate'          : r
            'carrying capacity'         : k
            'prey allee threshold'      : theta
            'predator conversion rate'  : alpha
            'prey consumption rate'     : beta
            'predator death rate'       : gamma
            'prey diffusion rate'       : du
            'predator diffusion rate'   : dv

    tf : float, default=20
        Final time.

    N1 : int, default=200
        Number of grid subintervals on [0, a].

    N2 : int, default=200
        Number of grid subintervals on [a, L].

    u0 : float or callable, default=1.0
        Initial condition for prey. If callable, must accept a spatial point x
        and return a scalar value.

    v0 : float or callable, default=0.5
        Initial condition for predator. If callable, must accept a spatial point x
        and return a scalar value.

    method : str, default="Radau"
        Time-integration method passed to scipy.integrate.solve_ivp.

    max_step : float, default=0.1
        Maximum time step passed to solve_ivp.

    t_eval : array-like or None, default=None
        Time points at which to store the computed solution.

    rtol : float, default=1e-3
        Relative tolerance for solve_ivp.

    atol : float or array-like, default=1e-6
        Absolute tolerance for solve_ivp.

    first_step : float or None, default=None
        Optional initial step size for solve_ivp.

    events : callable or list of callables or None, default=None
        Event function(s) passed to solve_ivp.

    compute_total_pops : bool, default=False
        If True, compute total prey and predator populations at each returned
        time point and return them as arrays PU and PV.

    verbose : bool, default=False
        If True, print progress information and a short solver summary.

    solve_ivp_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to solve_ivp. These will override the
        defaults above if the same keys are provided.

    Returns
    -------
    If compute_total_pops is False:
        {'U': U, 'V': V, 'G': G, 'T': T, 'E': E}

    If compute_total_pops is True:
        {'U': U, 'V': V, 'G': G, 'T': T, 'E': E, 'PU': PU, 'PV': PV}

    where
    -----
    U : ndarray
        Prey solution array with boundary padding included, shape
        (num_times, N1 + N2 + 1).

    V : ndarray
        Predator solution array padded to the full spatial grid, shape
        (num_times, N1 + N2 + 1).

    G : ndarray
        Full spatial grid.

    T : ndarray
        Time values returned by solve_ivp.

    E : object
        Event data returned by solve_ivp (S['t_events']).

    PU : ndarray
        Total prey population at each returned time, only returned if
        compute_total_pops=True.

    PV : ndarray
        Total predator population at each returned time, only returned if
        compute_total_pops=True.

    Notes
    -----
    This version preserves the core logic and output format of the original
    implementation while adding validation, configurability, and clearer structure.
    """
    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _print(msg):
        if verbose:
            print(msg)

    def _require_param(dictionary, key):
        if key not in dictionary:
            raise KeyError(
                f"Missing required parameter '{key}' in params. "
                f"Expected keys are: {required_keys}."
            )
        return dictionary[key]

    def _validate_nonnegative(name, value, strictly_positive=False):
        if not np.isscalar(value):
            raise TypeError(f"Parameter '{name}' must be a real scalar, got {type(value).__name__}.")
        if not np.isfinite(value):
            raise ValueError(f"Parameter '{name}' must be finite, got {value}.")
        if strictly_positive:
            if value <= 0:
                raise ValueError(f"Parameter '{name}' must be strictly positive, got {value}.")
        else:
            if value < 0:
                raise ValueError(f"Parameter '{name}' must be nonnegative, got {value}.")

    def _validate_positive_integer(name, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"'{name}' must be a positive integer, got {type(value).__name__}.")
        if value <= 0:
            raise ValueError(f"'{name}' must be a positive integer, got {value}.")

    def _build_initial_condition(ic, grid, label):
        """
        Construct an initial-condition vector from either:
        - a scalar constant, or
        - a callable evaluated pointwise on the grid.
        """
        if callable(ic):
            try:
                values = np.array([ic(x) for x in grid], dtype=np.float64)
            except Exception as exc:
                raise ValueError(
                    f"Initial condition '{label}' is callable, but evaluating it on the spatial grid failed."
                ) from exc
        else:
            if not np.isscalar(ic):
                raise TypeError(
                    f"Initial condition '{label}' must be either a scalar or a callable. "
                    f"Got object of type {type(ic).__name__}."
                )
            if not np.isfinite(ic):
                raise ValueError(
                    f"Initial condition '{label}' must be finite if given as a scalar, got {ic}."
                )
            values = np.full(len(grid), float(ic), dtype=np.float64)

        if values.shape != (len(grid),):
            raise ValueError(
                f"Initial condition '{label}' produced an array of shape {values.shape}, "
                f"but expected shape ({len(grid)},)."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Initial condition '{label}' produced non-finite values."
            )

        return values

    # ------------------------------------------------------------------
    # Validate top-level arguments
    # ------------------------------------------------------------------
    required_keys = [
        "length",
        "a",
        "prey growth rate",
        "carrying capacity",
        "prey allee threshold",
        "predator conversion rate",
        "prey consumption rate",
        "predator death rate",
        "prey diffusion rate",
        "predator diffusion rate",
    ]

    if not isinstance(params, dict):
        raise TypeError(f"'params' must be a dictionary, got {type(params).__name__}.")

    _validate_nonnegative("tf", tf)
    _validate_positive_integer("N1", N1)
    _validate_positive_integer("N2", N2)

    if not isinstance(compute_total_pops, bool):
        raise TypeError(
            f"'compute_total_pops' must be a boolean, got {type(compute_total_pops).__name__}."
        )

    if solve_ivp_kwargs is None:
        solve_ivp_kwargs = {}
    elif not isinstance(solve_ivp_kwargs, dict):
        raise TypeError(
            f"'solve_ivp_kwargs' must be a dictionary or None, got {type(solve_ivp_kwargs).__name__}."
        )

    # ------------------------------------------------------------------
    # Extract and validate model parameters
    # ------------------------------------------------------------------
    L = _require_param(params, "length")
    a = _require_param(params, "a")
    r = _require_param(params, "prey growth rate")
    k = _require_param(params, "carrying capacity")
    theta = _require_param(params, "prey allee threshold")
    alpha = _require_param(params, "predator conversion rate")
    beta = _require_param(params, "prey consumption rate")
    gamma = _require_param(params, "predator death rate")
    du = _require_param(params, "prey diffusion rate")
    dv = _require_param(params, "predator diffusion rate")

    _validate_nonnegative("length", L, strictly_positive=True)
    _validate_nonnegative("a", a, strictly_positive=True)
    _validate_nonnegative("prey growth rate", r)
    _validate_nonnegative("carrying capacity", k, strictly_positive=True)
    _validate_nonnegative("prey allee threshold", theta, strictly_positive=True)
    _validate_nonnegative("predator conversion rate", alpha)
    _validate_nonnegative("prey consumption rate", beta)
    _validate_nonnegative("predator death rate", gamma)
    _validate_nonnegative("prey diffusion rate", du)
    _validate_nonnegative("predator diffusion rate", dv)

    if a >= L:
        raise ValueError(
            f"Parameter 'a' must satisfy 0 < a < length. Got a={a} and length={L}."
        )

    if max_step is not None:
        _validate_nonnegative("max_step", max_step, strictly_positive=True)
    if first_step is not None:
        _validate_nonnegative("first_step", first_step, strictly_positive=True)
    _validate_nonnegative("rtol", rtol, strictly_positive=True)

    _print("Validated parameters and solver options.")

    # ------------------------------------------------------------------
    # Spatial grid construction
    # ------------------------------------------------------------------
    dx1 = a / N1
    dx2 = (L - a) / N2

    G1 = np.linspace(0, a, N1 + 1, dtype=np.float64)
    G2 = np.linspace(a, L, N2 + 1, dtype=np.float64)
    G = np.concatenate((G1, G2[1:]))

    Gu = G[1:-1]
    Gv = G[1:N1]

    _print(
        f"Constructed spatial grid: total points={len(G)}, "
        f"prey interior points={len(Gu)}, predator interior points={len(Gv)}."
    )

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    u0_vec = _build_initial_condition(u0, Gu, "u0")
    v0_vec = _build_initial_condition(v0, Gv, "v0")
    x0 = np.concatenate((u0_vec, v0_vec))

    _print("Built initial condition vectors.")

    # ------------------------------------------------------------------
    # Diffusion matrix construction
    # Core logic preserved from original implementation
    # ------------------------------------------------------------------
    d1 = np.full(N1 - 1, -2 / dx1**2, dtype=np.float64)
    d1[0] = -1 / dx1**2

    d2 = np.array([-2 / (dx1 * dx2)], dtype=np.float64)

    d3 = np.full(N2 - 1, -2 / dx2**2, dtype=np.float64)
    d3[-1] = -1 / dx2**2

    d4 = np.full(N1 - 1, -2 / dx1**2, dtype=np.float64)
    d4[0] = -1 / dx1**2
    d4[-1] = -1 / dx1**2

    dd = np.concatenate((du * d1, du * d2, du * d3, dv * d4))

    u1 = np.full(N1, 1 / dx1**2, dtype=np.float64)
    u2 = np.array([2 / (dx2 * (dx1 + dx2))], dtype=np.float64)
    u3 = np.full(N2 - 1, 1 / dx2**2, dtype=np.float64)
    u3[-1] = 0
    u4 = np.full(N1 - 2, 1 / dx1**2, dtype=np.float64)

    u_diag = np.concatenate((du * u1, du * u2, du * u3, dv * u4))

    l1 = np.full(N1 - 2, 1 / dx1**2, dtype=np.float64)
    l2 = np.array([2 / (dx1 * (dx1 + dx2))], dtype=np.float64)
    l3 = np.full(N2 - 1, 1 / dx2**2, dtype=np.float64)
    l4 = np.full(N1, 1 / dx1**2, dtype=np.float64)
    l4[0] = 0

    l_diag = np.concatenate((du * l1, du * l2, du * l3, dv * l4))

    data = np.array([l_diag, dd, u_diag], dtype=np.float64)
    offsets = (-1, 0, 1)

    system_size = 2 * N1 + N2 - 2
    D = dia_matrix((data, offsets), shape=(system_size, system_size), dtype=np.float64)

    _print(f"Constructed diffusion operator of size {system_size} x {system_size}.")

    # ------------------------------------------------------------------
    # RHS and Jacobian
    # Core logic preserved from original implementation
    # ------------------------------------------------------------------
    def f(t, x):
        U = x[:N1 + N2 - 1]
        v = x[N1 + N2 - 1:]

        u = U[:N1 - 1]
        V1 = v
        V2 = np.array([v[-1]], dtype=np.float64)
        V3 = np.zeros(N2 - 1, dtype=np.float64)
        V = np.concatenate((V1, V2, V3))

        du_reac = r * U * (U / theta - 1) * (1 - U / k) - beta * U * V
        dv_reac = -gamma * v + alpha * u * v

        Diff = D.dot(x)
        dx = Diff + np.concatenate((du_reac, dv_reac))

        return dx

    def jac(t, x):
        U = x[:N1 + N2 - 1]
        v = x[N1 + N2 - 1:]

        u = U[:N1 - 1]
        V1 = v
        V2 = np.array([v[-1]], dtype=np.float64)
        V3 = np.zeros(N2 - 1, dtype=np.float64)
        V = np.concatenate((V1, V2, V3))

        d1 = [
            r * (U[i] / theta - 1) * (1 - U[i] / k)
            + r * U[i] / theta * (1 - U[i] / k)
            - r * U[i] / k * (U[i] / theta - 1)
            - beta * V[i]
            for i in range(len(U))
        ]

        d2 = [-gamma + alpha * u[i] for i in range(len(v))]

        dd = np.concatenate((d1, d2))

        ud = np.pad([-beta * u[i] for i in range(len(v))], ((N1 + N2 - 1, 0)))
        ld = np.pad([alpha * v[i] for i in range(len(u))], ((0, N1 + N2 - 1)))

        data = np.array([ld, dd, ud], dtype=np.float64)
        offsets = (-(N1 + N2 - 1), 0, N1 + N2 - 1)

        J = D + dia_matrix((data, offsets), shape=(system_size, system_size), dtype=np.float64)
        return J

    # ------------------------------------------------------------------
    # Solve the ODE system
    # ------------------------------------------------------------------
    ivp_options = {
        "method": method,
        "jac": jac,
        "max_step": max_step,
        "t_eval": t_eval,
        "rtol": rtol,
        "atol": atol,
        "first_step": first_step,
        "events": events,
    }
    ivp_options.update(solve_ivp_kwargs)

    _print(f"Starting time integration on [0, {tf}] with method='{ivp_options.get('method', method)}'.")

    S = solve_ivp(f, (0, tf), x0, **ivp_options)

    if not S.success:
        raise RuntimeError(
            "solve_ivp failed to integrate the exclusion-zone system successfully. "
            f"Solver message: {S.message}"
        )

    _print(
        f"Integration complete. "
        f"Accepted time points: {len(S.t)}. "
        f"Solver message: {S.message}"
    )

    # ------------------------------------------------------------------
    # Reconstruct padded solution arrays
    # Core logic preserved from original implementation
    # ------------------------------------------------------------------
    T = S["t"]
    X = S["y"]
    U = np.transpose(np.pad(X[:N1 + N2 - 1], [(1, 1), (0, 0)], mode="edge"))
    V = np.pad(
        np.transpose(np.pad(X[N1 + N2 - 1:], [(1, 1), (0, 0)], mode="edge")),
        [(0, 0), (0, N2)]
    )
    E = S["t_events"]

    _print("Postprocessed solution arrays.")

    # ------------------------------------------------------------------
    # Optionally compute total populations
    # ------------------------------------------------------------------
    if compute_total_pops:
        _print("Computing total prey and predator populations over time.")

        PU = np.trapz(U, x=G, axis=1)
        PV = np.trapz(V, x=G, axis=1)

        _print("Total-population arrays computed. Returning results.")
        return {'U': U, 'V': V, 'G': G, 'T': T, 'E': E, 'PU': PU, 'PV': PV}

    _print("Returning results.")
    return {'U': U, 'V': V, 'G': G, 'T': T, 'E': E}

def plot_EZ(
    result,
    *,
    figsize=(12, 5),
    prey_label="Prey density",
    predator_label="Predator density",
    prey_total_label="Total prey population",
    predator_total_label="Total predator population",
    density_title_prefix="Population densities",
    total_title="Total populations vs. time",
    density_ylabel="Population density",
    total_ylabel="Total population",
):
    """
    Plot the output of solve_EZ in a single window with two side-by-side plots.

    Left plot:
        Population densities with a time slider underneath.

    Right plot:
        Total prey and predator populations as functions of time.

    Parameters
    ----------
    result : dict
        Dictionary output from solve_EZ. Must contain:
            'U', 'V', 'G', 'T'
        and may optionally contain:
            'PU', 'PV'

    figsize : tuple, default=(12, 5)
        Figure size.

    prey_label : str, default="Prey density"
        Legend label for prey density curve.

    predator_label : str, default="Predator density"
        Legend label for predator density curve.

    prey_total_label : str, default="Total prey population"
        Legend label for total prey population curve.

    predator_total_label : str, default="Total predator population"
        Legend label for total predator population curve.

    density_title_prefix : str, default="Population densities"
        Prefix for the left plot title. The time value is appended dynamically.

    total_title : str, default="Total populations vs. time"
        Title for the total-population plot.

    density_ylabel : str, default="Population density"
        Y-label for the density plot.

    total_ylabel : str, default="Total population"
        Y-label for the total-population plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.

    axes : tuple
        Tuple (ax_density, ax_total) of matplotlib axes.
    """
    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------
    if not isinstance(result, dict):
        raise TypeError(
            f"'result' must be a dictionary produced by solve_EZ, got {type(result).__name__}."
        )

    required_keys = ["U", "V", "G", "T"]
    missing = [key for key in required_keys if key not in result]
    if missing:
        raise KeyError(
            f"Result dictionary is missing required key(s): {missing}. "
            f"Expected keys include {required_keys}."
        )

    U = np.asarray(result["U"])
    V = np.asarray(result["V"])
    G = np.asarray(result["G"])
    T = np.asarray(result["T"])

    if U.ndim != 2:
        raise ValueError(f"result['U'] must be a 2D array, got shape {U.shape}.")
    if V.ndim != 2:
        raise ValueError(f"result['V'] must be a 2D array, got shape {V.shape}.")
    if G.ndim != 1:
        raise ValueError(f"result['G'] must be a 1D array, got shape {G.shape}.")
    if T.ndim != 1:
        raise ValueError(f"result['T'] must be a 1D array, got shape {T.shape}.")

    if U.shape[0] != len(T):
        raise ValueError(
            f"result['U'] has {U.shape[0]} time slices, but result['T'] has length {len(T)}."
        )
    if V.shape[0] != len(T):
        raise ValueError(
            f"result['V'] has {V.shape[0]} time slices, but result['T'] has length {len(T)}."
        )
    if U.shape[1] != len(G):
        raise ValueError(
            f"result['U'] has spatial dimension {U.shape[1]}, but result['G'] has length {len(G)}."
        )
    if V.shape[1] != len(G):
        raise ValueError(
            f"result['V'] has spatial dimension {V.shape[1]}, but result['G'] has length {len(G)}."
        )

    # ------------------------------------------------------------------
    # Compute total populations if not already present
    # ------------------------------------------------------------------
    if "PU" in result:
        PU = np.asarray(result["PU"])
        if PU.shape != (len(T),):
            raise ValueError(
                f"result['PU'] must have shape ({len(T)},), got {PU.shape}."
            )
    else:
        PU = np.trapz(U, x=G, axis=1)

    if "PV" in result:
        PV = np.asarray(result["PV"])
        if PV.shape != (len(T),):
            raise ValueError(
                f"result['PV'] must have shape ({len(T)},), got {PV.shape}."
            )
    else:
        PV = np.trapz(V, x=G, axis=1)

    # ------------------------------------------------------------------
    # Create figure and axes
    # ------------------------------------------------------------------
    fig, (ax_density, ax_total) = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(bottom=0.22, wspace=0.3)

    # ------------------------------------------------------------------
    # Left plot: densities
    # ------------------------------------------------------------------
    idx0 = 0
    prey_line, = ax_density.plot(G, U[idx0], label=prey_label)
    pred_line, = ax_density.plot(G, V[idx0], label=predator_label)

    density_ymax = max(np.max(U), np.max(V))
    if density_ymax <= 0:
        density_ymax = 1.0

    ax_density.set_xlim(G[0], G[-1])
    ax_density.set_ylim(0, density_ymax)
    ax_density.set_xlabel("x")
    ax_density.set_ylabel(density_ylabel)
    ax_density.set_title(f"{density_title_prefix} at t = {T[idx0]:.4g}")
    ax_density.legend()

    # ------------------------------------------------------------------
    # Right plot: total populations
    # ------------------------------------------------------------------
    ax_total.plot(T, PU, label=prey_total_label)
    ax_total.plot(T, PV, label=predator_total_label)
    ax_total.set_xlabel("t")
    ax_total.set_ylabel(total_ylabel)
    ax_total.set_title(total_title)
    ax_total.legend()

    # ------------------------------------------------------------------
    # Slider
    # ------------------------------------------------------------------
    slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
    time_slider = Slider(
        ax=slider_ax,
        label="Time index",
        valmin=0,
        valmax=len(T) - 1,
        valinit=idx0,
        valstep=1,
    )

    def _update(val):
        idx = int(time_slider.val)
        prey_line.set_ydata(U[idx])
        pred_line.set_ydata(V[idx])
        ax_density.set_title(f"{density_title_prefix} at t = {T[idx]:.4g}")
        fig.canvas.draw_idle()

    time_slider.on_changed(_update)

    # Save slider globally so it is not garbage collected
    global _EZ_TIME_SLIDER
    _EZ_TIME_SLIDER = time_slider

    plt.show()

    return fig, (ax_density, ax_total)

def a_sweep(
    params,
    a_list,
    tf=20,
    N1=200,
    N2=200,
    *,
    u0=1.0,
    v0=0.5,
    method="Radau",
    max_step=0.1,
    t_eval=None,
    rtol=1e-3,
    atol=1e-6,
    first_step=None,
    events=None,
    verbose=False,
    solve_ivp_kwargs=None,
):
    """
    Sweep over values of the exclusion-zone parameter a, solve the model for each,
    and plot asymptotic prey and predator population statistics.

    For each a in a_list, this function:
      1. Creates a copy of params with params['a'] replaced by that value.
      2. Calls solve_EZ(..., compute_total_pops=True).
      3. Extracts the last quarter of the PU and PV arrays.
      4. Computes:
            - liminf  = min(last quarter)
            - limmean = mean(last quarter)
            - limsup  = max(last quarter)
      5. Stores these values in arrays indexed by a.

    The resulting six arrays are then plotted against a.

    Parameters
    ----------
    params : dict
        Parameter dictionary for solve_EZ.

    a_list : array-like
        List or array of a-values, assumed sorted from smallest to largest.

    All remaining arguments match solve_EZ, except that compute_total_pops is
    always forced to True internally.

    Returns
    -------
    results : dict
        Dictionary containing:
            'a_list'
            'PU_liminf', 'PU_limmean', 'PU_limsup'
            'PV_liminf', 'PV_limmean', 'PV_limsup'

    fig : matplotlib.figure.Figure
        The created figure.

    ax : matplotlib.axes.Axes
        The plot axes.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not isinstance(params, dict):
        raise TypeError(f"'params' must be a dictionary, got {type(params).__name__}.")

    a_array = np.asarray(a_list, dtype=np.float64)
    if a_array.ndim != 1:
        raise ValueError(f"'a_list' must be one-dimensional, got shape {a_array.shape}.")
    if len(a_array) == 0:
        raise ValueError("'a_list' must not be empty.")
    if not np.all(np.isfinite(a_array)):
        raise ValueError("'a_list' must contain only finite values.")
    if np.any(a_array <= 0):
        raise ValueError("'a_list' must contain strictly positive values.")
    if np.any(np.diff(a_array) < 0):
        raise ValueError("'a_list' must be sorted from smallest to largest.")

    n = len(a_array)

    # ------------------------------------------------------------------
    # Allocate output arrays
    # ------------------------------------------------------------------
    PU_liminf = np.empty(n, dtype=np.float64)
    PU_limmean = np.empty(n, dtype=np.float64)
    PU_limsup = np.empty(n, dtype=np.float64)

    PV_liminf = np.empty(n, dtype=np.float64)
    PV_limmean = np.empty(n, dtype=np.float64)
    PV_limsup = np.empty(n, dtype=np.float64)

    # ------------------------------------------------------------------
    # Sweep over a-values
    # ------------------------------------------------------------------
    for i, a_val in enumerate(a_array):
        if verbose:
            print(f"[a_sweep] Solving {i + 1}/{n} with a = {a_val}")

        params_i = params.copy()
        params_i["a"] = float(a_val)

        result = solve_EZ(
            params_i,
            tf=tf,
            N1=N1,
            N2=N2,
            u0=u0,
            v0=v0,
            method=method,
            max_step=max_step,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            events=events,
            compute_total_pops=True,
            verbose=verbose,
            solve_ivp_kwargs=solve_ivp_kwargs,
        )

        PU = np.asarray(result["PU"], dtype=np.float64)
        PV = np.asarray(result["PV"], dtype=np.float64)

        if PU.ndim != 1 or PV.ndim != 1:
            raise ValueError("Returned 'PU' and 'PV' must be one-dimensional arrays.")
        if len(PU) == 0 or len(PV) == 0:
            raise ValueError("Returned 'PU' and 'PV' must be nonempty.")

        # Last quarter of each array
        start_idx = len(PU) * 3 // 4
        PU_tail = PU[start_idx:]
        PV_tail = PV[start_idx:]

        if len(PU_tail) == 0 or len(PV_tail) == 0:
            raise ValueError(
                "Unable to extract the last quarter of PU/PV. "
                "Returned arrays are unexpectedly short."
            )

        PU_liminf[i] = np.min(PU_tail)
        PU_limmean[i] = np.mean(PU_tail)
        PU_limsup[i] = np.max(PU_tail)

        PV_liminf[i] = np.min(PV_tail)
        PV_limmean[i] = np.mean(PV_tail)
        PV_limsup[i] = np.max(PV_tail)

    # ------------------------------------------------------------------
    # Plot the results
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    # Prey curves: default first matplotlib color (blue)
    ax.plot(a_array, PU_liminf, "--", color='C0', label="PU liminf")
    ax.plot(a_array, PU_limmean, "-", color='C0', label="PU limmean")
    ax.plot(a_array, PU_limsup, "--", color='C0', label="PU limsup")

    # Predator curves: default second matplotlib color (orange)
    ax.plot(a_array, PV_liminf, "--", color='C1', label="PV liminf")
    ax.plot(a_array, PV_limmean, "-", color='C1', label="PV limmean")
    ax.plot(a_array, PV_limsup, "--", color='C1', label="PV limsup")

    ax.set_xlabel("a")
    ax.set_ylabel("Population")
    ax.set_title("Asymptotic population statistics vs. a")
    ax.legend()

    plt.show()

    results = {
        "a_list": a_array,
        "PU_liminf": PU_liminf,
        "PU_limmean": PU_limmean,
        "PU_limsup": PU_limsup,
        "PV_liminf": PV_liminf,
        "PV_limmean": PV_limmean,
        "PV_limsup": PV_limsup,
    }

    return results, fig, ax

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Example workflow
    #
    # This block demonstrates a typical use of the module:
    #   1. define model parameters,
    #   2. solve the time-dependent exclusion-zone system,
    #   3. visualize the resulting dynamics,
    #   4. sweep over values of the predator-domain size a.
    #
    # You can run this file directly to reproduce a basic example.
    # ------------------------------------------------------------------

    # Model parameters
    #
    # The prey occupy the full interval (0, L), while predators occupy
    # only the subinterval (0, a). Thus, (a, L) is the predator-free
    # exclusion zone.
    a = 0.5
    L = 1.0
    alpha = 13.9
    beta = 10.0
    gamma = 5.0
    theta = 0.04
    du = 1.0
    dv = 0.52
    r = 0.904
    k = 1.0

    # Time horizon for the example simulation
    tf = 200

    # Number of a-values used in the sweep
    n = 99
    a_list = np.linspace(0.01 * L, 0.99 * L, n)

    # Bundle parameters into the dictionary expected by solve_EZ
    params = {
        "length": L,
        "a": a,
        "prey growth rate": r,
        "carrying capacity": k,
        "prey allee threshold": theta,
        "predator conversion rate": alpha,
        "prey consumption rate": beta,
        "predator death rate": gamma,
        "prey diffusion rate": du,
        "predator diffusion rate": dv,
    }

    print("Running a sample exclusion-zone simulation...")
    print(f"  Domain length L = {L}")
    print(f"  Predator-domain length a = {a}")
    print(f"  Final time tf = {tf}")

    # Solve the time-dependent PDE system with the default discretization
    # and solver settings.
    result = solve_EZ(params)

    print("Simulation complete.")
    print("Opening interactive plots...")

    # Visualize the solution:
    #   - left panel: spatial prey/predator densities with time slider
    #   - right panel: total prey/predator populations over time
    plot_EZ(result)

    print("Now running a sweep over predator-domain sizes a...")

    # Sweep over a range of predator-domain sizes and summarize the
    # long-time prey and predator population levels.
    #
    # This last line will likely take several minutes of computation on
    # most machines.
    a_sweep(params, a_list, tf=tf)

    print("Done.")