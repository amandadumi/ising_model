#!/usr/bin/env python
# coding: utf-8

# In this lab we will look at modeling an Ising system with Monte Carlo.

# # Ising Model
#
# The Ising model is a lattice of $k$ interacting sites. Each site has a spin $\sigma_k$ which can have a value of +1 or -1. The spins are allowed to interact with their nearest-neighbor spins. The Hamiltonian for the system is given by:
#
# $$
# H(\sigma)=-\sum_{\langle i j\rangle} J_{i j} \sigma_{i} \sigma_{j}
# $$
#
# where $\langle i j\rangle$ indicate a sum over only nearest neighbors.
# The interaction between spins is captured in $J$.
# (If $J$, is positive the system will behave ferromagnetically while a negative $J$ favors antiferromagnetic interactions. (maybe this sentence will be better in the conclusions?))
#
# We will start by introducing some code for this model.


# think about umbrella sampling between points
# 1-D
# 2-D phase transitions. Given equations, but not derived in class. Expectd 1/2, but actual answer is 1/8th


# To begin, we provide a python class for you to add your functions to. This will help keep variables organized. Initial functions have been provided to initialize your model, print the data, and plot the model.

import numpy as np
import matplotlib.pyplot as plt
import pyblock
import pandas as pd
import time
import ising_utils as iu


# a dictionary data structure to store all relevant parameters
ip = {"J": None, "h": None, "M": None, "N": None}


def initialize_lattice_state(ip):
    """
    Initializing the lattice of size MxN to all ones
    """
    lattice_state = np.ones((ip["M"], ip["N"]))
    return lattice_state


def flip_spin(lattice_state, i, j):
    """
    Flip spin (i, j)
    i.e. -1 ---> 1
          1 ---> -1
    """
    new_lattice_state = np.copy(lattice_state)
    new_lattice_state[i, j] = -new_lattice_state[i, j]
    return new_lattice_state


# Add simple test here to make sure they are interacting with class correctly.
# a = IsingModel(3, 3, 1.0, 0.0)
# a.plot_lattice()


def calculate_energy_of_site(lattice_state, ip, i, j):
    """
    Calculate energy of spin (i, j)

    Periodic boundary conditions implemented
    """
    spin_here = lattice_state[i, j]  # value of spin here

    # value of spin above, below, left, and right of spin (i, j)
    # for each, if on boundary, we wrap around to the other side
    # of the lattice for periodic boundary conditions
    if j == 0:
        spin_above = lattice_state[i, ip["N"] - 1]
    else:
        spin_above = lattice_state[i, j - 1]

    if j == ip["N"] - 1:
        spin_below = lattice_state[i, 0]
    else:
        spin_below = lattice_state[i, j + 1]

    if i == ip["M"] - 1:
        spin_right = lattice_state[0, j]
    else:
        spin_right = lattice_state[i + 1, j]

    if i == 0:
        spin_left = lattice_state[ip["M"] - 1, j]
    else:
        spin_left = lattice_state[i - 1, j]

    return -ip["h"] * spin_here - ip["J"] * spin_here * (
        spin_above + spin_below + spin_left + spin_right
    )


# add test here for correct behavior


def calculate_total_spin(lattice_state):
    """
    Calculate total spin of the lattice
    """
    return np.sum(lattice_state)


def calculate_total_spin_per_spin(lattice_state, ip):
    """
    Calculate total spin of the lattice
    """
    return calculate_total_spin(lattice_state) / (ip["M"] * ip["N"])


def calculate_total_energy(lattice_state, ip):
    """
    Calculate total energy of the lattice
    """
    E = 0.0
    for i in range(ip["M"]):
        for j in range(ip["N"]):
            E += calculate_energy_of_site(lattice_state, ip, i, j)
    # factor of two for overcounting neighboring interactions.
    # but then need to add back -1/2 h \sum s_i
    return (E - (ip["h"] * calculate_total_spin(lattice_state))) / 2.0


def calculate_total_energy_per_spin(lattice_state, ip):
    """
    Calculate energy of lattice normalized by the number of spins
    """
    return calculate_total_energy(lattice_state, ip) / (ip["M"] * ip["N"])


# # Monte Carlo Simulations
#
# Theoretically, we have made a very simple model for describing interacting systems.
# Additionally, we can now start to use this model for predicting observables.
# For example, for the expectation value of the energy of an $M*N$ lattice, the expectation value would be
#
# $$
# \langle E\rangle=\sum_{\alpha} E(\alpha) P(\alpha)
# $$
#
# where $E(\alpha)$ is the energy of a fixed state $\alpha$, and  $P(\alpha)$ is the probability of being in that fixed state.
# However, the number of fixed states grows as $2^{(N*M)}$ where $N*M$ is the total number of lattice points.
# This quickly becomes impractical as the lattice size grows.
#
# To deal with this, we use Monte Carlo sampling to sample states $\alpha$ with probability $P(\alpha)$.

# #References:
#
# code: https://github.com/CorySimon/IsingModel/blob/master/Ising%20Model.ipynb
#
# theory: https://arxiv.org/pdf/0803.0217.pdf

# simulation parameters
sp = {"num_equil_sweeps": None, "num_sweeps": None, "kT": None}
# simulation results:
sr = {"energy_list": [], "total_spin_list": []}


def metropolis_test(sp, E_new, E_old):
    if np.random.random() <= np.exp(-(E_new - E_old) / sp["kT"]):
        return True
    else:
        return False


def sweep(lattice_state, ip, sp):
    for site_i in range(ip["M"]):
        for site_j in range(ip["N"]):
            E_old = calculate_energy_of_site(lattice_state, ip, site_i, site_j)
            # flip spin i and jN
            new_lattice_state = flip_spin(lattice_state, site_i, site_j)
            # calculate updated energy
            E_new = calculate_energy_of_site(new_lattice_state, ip, site_i, site_j)
            # Monte Carlo step
            if metropolis_test(sp, E_new, E_old):
                # accept move
                E_old = E_new
                lattice_state = np.copy(new_lattice_state)
            else:
                # reject move
                # do nothing just continue
                continue
    return lattice_state


def get_average_energy(sr):
    return np.mean(sr["energy_list"])


def get_stderr_energy(sr, sp):
    return np.std(sr["energy_list"]) / np.sqrt(sp["num_sweeps"])


def get_average_spin(sr):
    return np.mean(sr["total_spin_list"])


def get_stderr_spin(sr, sp):
    return np.std(sr["total_spin_list"]) / np.sqrt(sp["num_sweeps"])


def run_calculation(ip, sp):
    sr = {"energy_list": [], "total_spin_list": []}
    lattice_state = initialize_lattice_state(ip)
    for step in range(sp["num_equil_sweeps"]):
        lattice_state = sweep(lattice_state, ip, sp)
    for step in range(sp["num_sweeps"]):
        lattice_state = sweep(lattice_state, ip, sp)
        # calculate total energy
        e = calculate_total_energy_per_spin(lattice_state, ip)
        sr["energy_list"].append(e)

        # calculate total s2
        s = calculate_total_spin_per_spin(lattice_state, ip)
        sr["total_spin_list"].append(s)
    return sr


'''
class Wolff_Calculation(Calculation):
    def __init__(self, ising_model, kT=1, num_equil_sweeps=1000, num_sweeps=1000):
         Calculation.__init__(self, ising_model, kT, num_equil_sweeps, num_sweeps)
         self.p_add = 1 - np.exp(-2*self.ising_model.J/self.kT)
         #print(self.p_add)
         #exit()

    def sweep(self):
        # randomly choose site
        # from [0, argument), exclusive
        n_rand = np.random.randint(self.ising_model.N)
        m_rand = np.random.randint(self.ising_model.M)
        # create cluster from original site
        site = [n_rand, m_rand]
        self.cluster = [site]
        self.determine_added_sites(site)
        # flip cluster
        for site in self.cluster:
            self.ising_model.flip_spin(site[0], site[1])
        self.record_observables()

    def find_all_neighbors_site(self, site):
        """
        Calculate neighbors of site (i, j)

        Periodic boundary conditions implemented
        """
        # value of spin above, below, left, and right of spin (i, j)
        # for each, if on boundary, we wrap around to the other side
        # of the lattice for periodic boundary conditions
        i = site[0]
        j = site[1]
        neighbor_list = []
        if j == 0:
            neighbor_above = [i, self.ising_model.N - 1]
        else:
            neighbor_above = [i, j - 1]
        neighbor_list.append(neighbor_above)
        if j == self.ising_model.N - 1:
            neighbor_below = [i, 0]
        else:
            neighbor_below = [i, j + 1]
        neighbor_list.append(neighbor_below)
        if i == self.ising_model.M - 1:
            neighbor_right = [0, j]
        else:
            neighbor_right = [i + 1, j]
        neighbor_list.append(neighbor_right)
        if i == 0:
            neighbor_left = [self.ising_model.M - 1, j]
        else:
            neighbor_left = [i - 1, j]
        neighbor_list.append(neighbor_left)
        return neighbor_list
    
    def add_neighbor(self, site, spin):
        if self.ising_model.lattice_state[site[0],site[1]] != spin:
            return False
        
        if np.random.random() < self.p_add:
            return True
        else:
            return False

    def determine_added_sites(self, site):
        """
        determining bonds to site
        """
        # find all neighbors of site
        neighbors = self.find_all_neighbors_site(site)
        # if neighboring site is of the same spin
        for neighbor in neighbors:
            if neighbor in self.cluster:
                continue
            elif self.add_neighbor(neighbor, self.ising_model.lattice_state[site[0],site[1]]):
                self.cluster.append(neighbor)
                self.determine_added_sites(neighbor)

    def grow_cluster(self, site):
        """
        explores all sites connected 
        """
        pass
'''


# create cluster from all connected sites

# repeat until no bonds are created

# flip entire cluster

#
# a = IsingModel(10, 10, 0, h=1)
# a2 = Calculation(a, kT=1, num_sweeps=100)
# a2.run_calculation()
# plt.figure()
# plt.plot(range(a2.num_sweeps), a2.energy_list,label ='E')
# plt.plot(range(a2.num_sweeps), a2.total_spin_list,label ='S')
# print(np.average(a2.energy_list))
# print(np.average(a2.total_spin_list))
# plt.legend()
# plt.show()
# plt.savefig('sweeps_es.png',dpi=300)


def analytical(x, J):
    analytical_solution = []
    for i in x:
        if i < 2.269:
            analytical_solution.append(
                (1 - np.sinh((2 * J) / i) ** (-4)) ** (1.0 / 8.0)
            )
        else:
            analytical_solution.append(0)

    return analytical_solution


start_time = time.time()
kT_list = np.arange(1.2, 3.0, 0.1)
analytical_kT_list = np.arange(1.2, 3.0, 0.001)
avg_s_list = []
stderr_s_list = []
num_equil = 10
num_sample = 150
ip = {"N": 10, "M": 10, "J": 1, "h": 0.01}
s_analytical = analytical(analytical_kT_list, ip["J"])
for kT in kT_list:
    print("kT={}".format(kT))
    sp = {"num_equil_sweeps": num_equil, "num_sweeps": num_sample, "kT": kT}

    sr = run_calculation(ip, sp)
    print(sr)
    mean, std_err = iu.get_reblocked_avg_stderr_spin(sr, sp)
    print("kT={}    mean {}    std_err {}".format(kT, mean, std_err))

    avg_s_list.append(np.abs(mean))
    stderr_s_list.append(std_err)
    fig, ax = plt.subplots(tight_layout=True)
    fig.set_size_inches(8, 5)
    n_bins = 25
    bins = np.linspace(-1, 1, n_bins)
    ax.hist(sr["total_spin_list"], bins=bins)
    ax.set_title(
        "kT={:>4.2f} m={:>02d}, n={:>02d}, J={:>4.2f}, h={:>4.2f} samp={}".format(
            sp["kT"], ip["M"], ip["N"], ip["J"], ip["h"], sp["num_sweeps"]
        )
    )
    ax.set_xlim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig(
        "metropolis_s_hist_kT_{:4.2f}_n_{:>02d}_m_{:>02d}_J_{:>4.2f}_h_{:>4.2f}_samp_{}.png".format(
            sp["kT"], ip["M"], ip["N"], ip["J"], ip["h"], sp["num_sweeps"]
        ),
        dpi=300,
    )
    # plt.show()
    s = pd.Series(sr["total_spin_list"])
    print("avg s autocorr = {}".format(s.autocorr()))
plt.figure(figsize=(8, 5))
plt.errorbar(kT_list, avg_s_list, stderr_s_list, label="avg S")
plt.plot(analytical_kT_list, s_analytical, label="analytical")
plt.title(
    "m={:>02d}, n={:>02d}, J={:>4.2f}, h={:>4.2f} samp={}".format(
        ip["M"], ip["N"], ip["J"], ip["h"], sp["num_sweeps"]
    )
)

plt.legend()
plt.tight_layout()
plt.savefig(
    "metropolis_avg_s_m_{:>02d}_n_{:>02d}_J_{:>4.2f}_h_{:>4.2f}_samp_{}.png".format(
        ip["M"], ip["N"], ip["J"], ip["h"], sp["num_sweeps"]
    ),
    dpi=300,
)
# plt.show()

print("Runtime (s) = {}".format(time.time() - start_time))
# # Generating histogram of bins for T = 2.3
# n = 30
# m = 30
# J = 1
# h = 0.00
# a = IsingModel(n, m, J, h)
# kT = 2.3
# a3 = Calculation(a, kT=kT, num_equil_sweeps=2000, num_sweeps=10000)
# a3.run_calculation()
# s = a3.total_spin_list
# fig, ax = plt.subplots(tight_layout=True)
# n_bins = 25
# ax.hist(x, bins=n_bins)
# plt.savefig(
#     "metropolis_s_hist_T_{:4.2f}_{:>02d}_m_{:>02d}_J_{:>4.2f}_h_{:>4.2f}.png".format(kT, n, m, J, h),
#     dpi=300,
# )
# plt.show()

# s = pd.Series(s)
# s.autocorr()
