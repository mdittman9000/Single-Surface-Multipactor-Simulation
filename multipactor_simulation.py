#
# \file multipactor_simulation.py
#
# \author Michael Dittman
#

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
from scipy import special
from mpmath import *

# Analytically determined random emission energy
def random_emission_energy(U, E0m):
    """
    Random generation of an emission energy, U must be a uniform
    Distribution from the calling area
    :param U: Value to evaluate inverse CDF at
    :param E0m: Peak of distribution emission energies
    :return: Value at U
    """
    W = -scipy.special.lambertw((U-1)/math.e, -1)

    result = re((E0m*W) - E0m)

    return result

# Analytically determined random emission angle
def random_emission_angle(U):
    """
    Random genereration of emission angle, U must be a uniform
    Distribution from the calling area
    :param U: Value to evaluate inverse CDF at
    :return: Value at U
    """
    return math.acos(1-2*U)

# Function that calculates flight time of a macroparticle
def flight_time(mass, charge, Edc, v0, phi):
    """
    :param mass: The mass of an electron
    :param charge: The charge of the macroparticle
    :param Edc: The perpendicular electric field (constant for a single run)
    :param v0: The initial velocity magnitude
    :param phi: The emission angle of the macroparticle
    """

    return (2 * mass * v0 * (math.sin(phi))) / (Edc * charge)

# Function that calculates impact energy
def calculate_impact_energy(mass, charge, Edc, Erf, v0, phi, flight_time, omega, theta):
    """"
    :param mass: The mass of the macroparticle
    :param charge: The charge of the macroparticle
    :param Edc: The perpendicular electric field (constant for a single run)
    :param v0: The initial velocity magnitude
    :param phi: The emission angle of the macroparticle
    :return: The impact energy in eV
    """
    # Calculate the impact energy
    vx = ((-charge / mass) * Edc * flight_time) + v0 * math.sin(phi)

    vy = ((charge / (mass * omega)) * Erf * (math.cos((omega * flight_time) + theta) - math.cos(theta))) + v0 * (
        math.cos(phi))

    v_mag = math.sqrt((vx*vx) + (vy*vy))

    # Convert to Ev
    return .5*9.109E-31*(v_mag * v_mag) /(1.602E-19)

# Function to calculate impact angles
def calculate_impact_angle(mass, charge, Edc, Erf, v0, phi, flight_time, omega, theta):
    """
    :param mass: The mass of the macroparticle
    :param charge: The charge of the macroparticle
    :param Edc: The DC electric field strength at this point
    :param Erf: The RF electric field strength at this point
    :param v0: The emission velocity of this macroparticle
    :param phi: The emission angle of this macroparticle
    :param flight_time: The flight time of this macroparticle as determined by the force law equation
    :param omega: The angular frequency of this RF field
    :param theta: The phase of the rf field (uniform distribution)
    :return: The angle of impact of the macroparticle
    """

    # Calculate the x component of velocity (multipactor force law equation)
    vx = ((-charge / mass) * Edc * flight_time) + v0 * math.sin(phi)

    # Calculate the y component of velocity (multipactor force law equation)
    vy = ((charge / (mass * omega)) * Erf * (math.cos((omega * flight_time) + theta) - math.cos(theta))) + v0 * (
        math.cos(phi))

    # return the angle
    return math.atan(vy/vx)

# Average the SEY array
def calculate_average_sey(SEY):
    """
    Simple function to average the sey value
    :param SEY: The array of SEY values
    """
    N = len(SEY)

    product = 1

    # Average the SEY array
    for i in range(N):
        product = product * SEY[i]

    result = product ** (1 / N)

    val = result[0].item()

    if (math.isnan(val) or val < 1.00):
        return 0

    return 1.0

# Calcuate the sey value with the updated vaughn empircal formula
def calculate_SEY_experimental(Ei, Xi):
    """
    Calculate the empirical sey
    :param Ei : impact energy
    :param Xi : impact angle
    """

    ks = 1

    # Parameters for when impact angle Xi = 0
    Emax0 = 400  # [ev]
    dmax0 = 3.0  # [ev]

    # Parameter adjustment
    Emax = Emax0 * (1 + ((ks * Xi * Xi) / (2 * math.pi)))
    dmax = dmax0 * (1 + ((ks * Xi * Xi) / (2 * math.pi)))

    # definition of w
    w = (Ei) / (Emax)

    # k = .62 for w < 1 and .25 for w = 1
    if w < 1:
        k = .56
    if (1 <= w) and (w < 3.6):
        k = .25

    # dmax is the maximum value of d, Emax is the impact energy which yields dmax

    if w < 3.6:
        return dmax * ((w * (math.e ** (1 - w))) ** k)
    else:
        return dmax * (1.125/(w**.35))

# Calculate the sey with the non-updated vaughn empircal formula
def calculate_sey(Ei, Xi):
    """
    Calculate the empirical sey
    :param Ei : impact energy
    :param Xi : impact angle
    """

    ks = 1

    # Parameters for when impact angle Xi = 0
    Emax0 = 400  # [ev]
    dmax0 = 3  # [ev]

    # Parameter adjustment
    Emax = Emax0 * (1 + ((ks * Xi * Xi) / (2 * math.pi)))
    dmax = dmax0 * (1 + ((ks * Xi * Xi) / (2 * math.pi)))

    # definition of w
    w = (Ei) / (Emax)

    # k = .62 for w < 1 and .25 for w = 1

    k = 0

    if w < 1:
        k = .62
    if w >= 1:
        k = .25

    # dmax is the maximum value of d, Emax is the impact energy which yields dmax

    return dmax * ((w * (math.e ** (1 - w))) ** k)


# Returns the average SEY for a specific value of Edc and Erf in N runs
# Do a single run with one value of Edc and Erf
def single_run(N, E0m, Edc, Erf, frequency):
    """
    :param N: the number of times to do the run
    :param E0m: The peak emission energy
    :param Edc: The perpendicular dc field
    :param Erf: The magnitude of the rf field
    :param frequency: The frequency of the rf field
    """

    # Electron parameter constants
    charge_electron = 1.602E-19  # [C]
    mass_electron = 9.109E-31  # [kg]

    # Angular frequency of the RF field
    omega = frequency * (2 * math.pi)

    # Macroparticle arrays
    mass = np.zeros((N + 1, 1), float)
    charge = np.zeros((N + 1, 1), float)

    # SEY array
    SEY = np.zeros((N, 1), float)

    # Initial values
    mass[0] = mass_electron
    charge[0] = charge_electron

    # Calculate the SEY array
    for i in range(N):
        # Try catch here

        # Uniformly distributed RF phase from 0 to 2pi
        theta = random.uniform(0, 2 * math.pi)

        # Generate an emission angle for the particle
        emission_angle = random_emission_angle(random.uniform(0, 1))

        # Generate an emission energy for the particle
        emission_energy = random_emission_energy(random.uniform(0, 1), E0m)

        # Convert the emission energy to Joules
        emission_energy_joules = emission_energy * 1.6022E-19  # ev to Joules

        # Get the initial velocity
        v0 = math.sqrt(2 * emission_energy_joules / mass_electron)  # [kg]

        # Calculate its flight time (mass, charge, Edc, v0, phi)
        time_of_flight = flight_time(mass[i], charge[i], Edc, v0,
                                     emission_angle)

        # Calculate the impact angle (mass, charge, Edc, Erf, v0, phi, flight_time, omega, theta)
        Xi = calculate_impact_angle(mass[i], charge[i], Edc, Erf, v0, emission_angle,
                                    time_of_flight, omega, theta)  # Radians

        # Calculate the impact energy (mass, charge, Edc, Erf, v0, phi, flight_time, omega, theta)
        Ei = calculate_impact_energy(mass[i], charge[i], Edc, Erf, v0, emission_angle, time_of_flight, omega,
                                     theta)  # Joules

        # Calculate SEY
        SEY[i] = calculate_sey(Ei, Xi)

        # Increase the mass and charge
        # Fix
        mass[i + 1] = SEY[i] * mass[i]  # [kg]
        charge[i + 1] = SEY[i] * charge[i]  # [C]

    return calculate_average_sey(SEY)

def sey_calculator(E0m, N, charts, Edc_max, Erf_max):
    """
    Plot the susceptibility associated with the Peak emission energy
    :param E0m: The peak emission energy
    :param N: The number of items in array
    :param charts: The number of times to average
    """

    # Create an NxNxN cube for the Sey values
    # (NxN)xN -> (NxN)x1 stores the values of the first complete plot...(NxN)x2 stores for the second...
    sey_cube = np.array([[[0 for k in range(N)] for j in range(N)] for i in range(charts)])

    # For each calculation create an Edc, Erf array
    Edc = np.linspace(0.01, Edc_max, N)
    Erf = np.linspace(0, Erf_max, N)

    ## get the values for the susceptibility chart ##
    for i in range(charts):

        for j in range(N):
            for k in range(N):
                # Fill the cube. i is a plane of Seys j,k are numbers in that plane
                sey_cube[i, j, k] = single_run(100, E0m, Edc[j], Erf[k], 1E+9)

        #plot_chart(sey_cube[i,:,:], E0m, N, "temp.csv", Edc_max, Erf_max)

    return sey_cube

def single_chart(E0m, N, filename, Edc_max, Erf_max):
    """
    Plot the susceptibility associated with the Peak emission energy
    :param E0m: The peak emission energy
    :param N: The number of items in array
    """
    sey_plot = np.array([[0.0 for i in range(N)] for j in range(N)])
    Edc = np.linspace(0.01, Edc_max, N)
    Erf = np.linspace(0, Erf_max, N)

    A, B = np.meshgrid(Edc, Erf)

    ## get the values for the susceptibility chart ##
    for j in range(N):
        for k in range(N):
            sey_plot[j][k] = single_run(100, E0m, Edc[j], Erf[k], 1E+9)

    fig, axs = plt.subplots(1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plot_chart(sey_plot, 2, N, filename, Edc_max, Erf_max)


def median_values(sey_cube, charts, N):
    """
    Take an NxNxN cube of sey values and return the median value of each iteration
    :param sey_cube: The NxNxN datastructure of sey values
    :param N: The number of items in an array
    :return: The median sey NxN array
    """

    sey_plot = np.array([[0.0 for i in range(N)] for j in range(N)])

    for j in range(N):
        for k in range(N):
            temp = sorted(sey_cube[:, j, k])
            res = temp[int(charts / 2)]

            sey_plot[j][k] = res

    return sey_plot


def plot_chart(sey_plot, E0m, N, filename, image_name, Edc_max, Erf_max):
    """
    Plot the sey values
    :param sey_plot:
    :return:
    """

    Edc = np.linspace(0.01, Edc_max, N)
    Erf = np.linspace(0, Erf_max, N)

    A, B = np.meshgrid(Edc, Erf)

    fig, axs = plt.subplots(1)

    # Font size for the tick marks and tick numbers
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # What power will the axis be in?
    axs.ticklabel_format(scilimits=(6, 6))
    axs.contourf(A, B, np.transpose(sey_plot), cmap='binary')
    axs.set_ylabel("$E_{rf}$ [MV/m] × $(f/1GHz)^{-1}$ × $(E_{max0}/400eV)^{-1/2}$", fontsize=24)
    axs.set_xlabel("$E_{dc}$ [MV/m] × $(f/1GHz)^{-1}$ × $(E_{max0}/400eV)^{-1/2}$", fontsize=24)
    #axs.set_title("Multipactor susceptibility chart" + " E0m = " + str(E0m), fontsize=18)

    fig.set_size_inches(18, 18, forward=True)

    np.savetxt(filename, sey_plot, delimiter=",")
    plt.savefig(image_name)
    plt.show()

def run_simulation(E0m, N, charts, filename, image_name, Edc_max, Erf_max ):
    """
    Run the simulation of the macroparticle
    :param E0m:
    :param N:
    :param filename:
    :param Edc_max:
    :param Erf_max:
    :return:
    """

    # Calculate all sey values
    sey_cube = sey_calculator(E0m, N, charts, Edc_max, Erf_max)

    # Get the median values
    sey_array = median_values(sey_cube, charts, N)

    # Plot the sey array PARAMS : plot_chart(sey_plot, E0m, N, filename, Edc_max, Erf_max)
    plot_chart(sey_array, E0m, N, filename, image_name,  Edc_max, Erf_max)

# Run the simulation PARAMS : run_simulation(E0m, N, charts, filename, Edc_max, Erf_max ):
run_simulation(2, 300, 30,"sey6.csv", "final_sey_plot5.png", .6E+6, 6E+6)

