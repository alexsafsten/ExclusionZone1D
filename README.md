README
======

exclusion_zone_1D.py
--------------------

This repository contains numerical tools for a one-dimensional predator-prey reaction-diffusion
model with a predator-free exclusion zone. The code is intended as companion software for the paper

    Berestycki, Fagan, and Safsten,
    "The Influence of Exclusion Zones on the Coexistence of Predator and Prey with an Allee Effect."

    https://arxiv.org/abs/2602.21414

In this model, the prey occupy the full interval (0, L), while the predators are confined to the
subinterval (0, a). The remaining region (a, L) is a predator-free exclusion zone. The prey obey
bistable / strong-Allee-effect growth, and both species diffuse subject to homogeneous Neumann
boundary conditions.

Main features
-------------

The script provides three main public functions:

1. solve_EZ
   Solves the time-dependent 1D exclusion-zone model using a finite-difference discretization in
   space and scipy.integrate.solve_ivp in time.

2. plot_EZ
   Produces a figure with:
     - spatial prey and predator densities, controlled by an interactive time slider
     - total prey and predator populations as functions of time

3. a_sweep
   Sweeps over a list of predator-domain sizes a and summarizes long-time prey and predator
   population levels by plotting liminf, mean, and limsup values computed from the final quarter
   of each simulation.

Requirements
------------

This script requires:

- Python 3
- NumPy
- SciPy
- Matplotlib

A typical installation is:

    pip install numpy scipy matplotlib

Files
-----

- exclusion_zone_1D.py
    Main script containing the solver, plotting helper, parameter sweep routine, and a runnable
    example in the __main__ block.

Typical workflow
----------------

1. Define a parameter dictionary.
2. Call solve_EZ(params, ...) to compute a time-dependent solution.
3. Call plot_EZ(result) to inspect the solution visually.
4. Call a_sweep(params, a_list, ...) to study how the long-time behavior depends on the predator
   domain size a.

Example
-------

Here is a minimal example:

    import numpy as np
    from exclusion_zone_1D import solve_EZ, plot_EZ, a_sweep

    params = {
        "length": 1.0,
        "a": 0.5,
        "prey growth rate": 0.904,
        "carrying capacity": 1.0,
        "prey allee threshold": 0.04,
        "predator conversion rate": 13.9,
        "prey consumption rate": 10.0,
        "predator death rate": 5.0,
        "prey diffusion rate": 1.0,
        "predator diffusion rate": 0.52,
    }

    result = solve_EZ(params, tf=200)
    plot_EZ(result)

    a_list = np.linspace(0.01, 0.99, 50)
    sweep_results, fig, ax = a_sweep(params, a_list, tf=200)

Notes on output
---------------

solve_EZ returns a dictionary with keys:

- "U" : prey density on the full spatial grid
- "V" : predator density on the full spatial grid
- "G" : spatial grid
- "T" : time values
- "E" : event data returned by solve_ivp

If compute_total_pops=True is passed to solve_EZ, the result also contains:

- "PU" : total prey population over time
- "PV" : total predator population over time

The predator density V is returned on the same full grid as U, with zeros in the predator-free
region. This makes plotting and numerical integration straightforward.

About the numerics
------------------

The code uses a piecewise-uniform finite-difference grid adapted to the interface x = a. The
resulting semi-discrete ODE system is integrated in time with SciPy's stiff ODE solvers
(default: Radau), using an analytic sparse Jacobian.

Intended audience
-----------------

This code is meant to be readable and modifiable by researchers and students with some
familiarity with reaction-diffusion equations, mathematical biology, and Python scientific
computing. It is not a general-purpose PDE solver; it is a focused implementation for the 1D
exclusion-zone model studied in the accompanying paper.

Running the script directly
---------------------------

If you run exclusion_zone_1D.py directly, the script will:

1. define a sample parameter set
2. solve the time-dependent model
3. open an interactive visualization
4. run a sweep over predator-domain sizes

License / citation
------------------

If you use this code in academic work, please cite the accompanying paper.
