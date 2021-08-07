# Single-Surface-Multipactor-Simulation
Monte Carlo simulation of multipactor discharge phenomena for a single surface dielectric.

Program developed in python to support simulation of multipactor phenomena on a single surface dielectric with varying DC/RF electric fields.

Usage:
Call the run_simulation function with parameters; E0m (peak emission energy), N (granularity of the graph), charts (number of graphs that will 
be average together), filename (of picture), Edc_max, Erf_max.

Areas that are colored black are values of Edc and Erf that result in a multipactor discharge, areas colored in white have values of Edc, Erf
that do not result in multipactor.
