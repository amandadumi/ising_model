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


class IsingModel:
    """
    Store attributes of an Ising lattice model
    Provide abstractions to conveniently manipulate lattice for simulations
    """

    def __init__(self, M, N, J, h=0):
        """
        Initialization.

        parameters:
            lattice is M by N sites
            M: size of first dimension
            N: size of second dimension
            J: interaction between neighbors (units: kT)
            h: background (external) field (units: kT)
        """
        # store parameters for convenience:
        #   energetic parameters
        self.J = J
        self.h = h

        #   size of lattice
        self.M = M
        self.N = N

        # We will store the lattice as an M by N array of -1 or 1
        # initialize each site as -1 or 1 with equal probability

        # The np.random.randint initializes random ints
        # but does not include the high value so this initializes a
        # matrix of -1 and 0's
        lattice_state = np.random.randint(-1, high=1, size=(M, N))
        # then we change all the zeros to ones so we have a
        # matrix of -1 and 1
        lattice_state[lattice_state == 0] = 1
        self.lattice_state = lattice_state

    def print_params(self):
        """
        Print lattice attributes
        """
        print("\t{:d} by {:d} lattice".format((self.M, self.N)))
        print(
            "\tJ = {: 8.6f}   (positive means a favorable interaction)".format(self.J)
        )
        print("\th = {: 8.6f}   (external field aligned with spins)".format(self.h))

    def plot_lattice(self):
        """
        Plot lattice configuration
        """
        plt.figure()

        imgplot = plt.imshow(self.lattice_state)
        imgplot.set_interpolation("none")

        plt.xticks(range(self.M))
        plt.yticks(range(self.N))

        for i in range(self.M + 1):
            plt.plot([i - 0.5, i - 0.5], [0 - 0.5, self.N - 0.5], color="black")
        for j in range(self.N + 1):
            plt.plot([0 - 0.5, self.M - 0.5], [j - 0.5, j - 0.5], color="black")

        plt.show()


class IsingModel(IsingModel):
    def flip_spin(self, i, j):
        """
       Flip spin (i, j)
       i.e. -1 ---> 1
             1 ---> -1
       """
        self.lattice_state[i, j] = -self.lattice_state[i, j]


# Add simple test here to make sure they are interacting with class correctly.
# a = IsingModel(3, 3, 1.0, 0.0)
# a.plot_lattice()


class IsingModel(IsingModel):
    def calculate_energy_of_site(self, i, j):
        """
      Calculate energy of spin (i, j)

      Periodic boundary conditions implemented
      """
        spin_here = self.lattice_state[i, j]  # value of spin here

        # value of spin above, below, left, and right of spin (i, j)
        # for each, if on boundary, we wrap around to the other side
        # of the lattice for periodic boundary conditions
        if j == 0:
            spin_above = self.lattice_state[i, self.N - 1]
        else:
            spin_above = self.lattice_state[i, j - 1]

        if j == self.N - 1:
            spin_below = self.lattice_state[i, 0]
        else:
            spin_below = self.lattice_state[i, j + 1]

        if i == self.M - 1:
            spin_right = self.lattice_state[0, j]
        else:
            spin_right = self.lattice_state[i + 1, j]

        if i == 0:
            spin_left = self.lattice_state[self.M - 1, j]
        else:
            spin_left = self.lattice_state[i - 1, j]

        return -self.h * spin_here - self.J * spin_here * (
            spin_above + spin_below + spin_left + spin_right
        )


# add test here for correct behavior


class IsingModel(IsingModel):
    def calculate_total_spin(self):
        """
      Calculate total spin of the lattice
      """
        return np.sum(self.lattice_state)


class IsingModel(IsingModel):
    def calculate_total_spin_per_spin(self):
        """
      Calculate total spin of the lattice
      """
        return self.calculate_total_spin() / (self.M * self.N)


class IsingModel(IsingModel):
    def calculate_total_energy(self):
        """
      Calculate total energy of the lattice
      """
        E = 0.0
        for i in range(self.M):
            for j in range(self.N):
                E += self.calculate_energy_of_site(i, j)
        # factor of two for overcounting neighboring interactions.
        # but then need to add back -1/2 h \sum s_i
        return (E - (self.h * self.calculate_total_spin())) / 2.0


class IsingModel(IsingModel):
    def calculate_total_energy_per_spin(self):
        """
      Calculate energy of lattice normalized by the number of spins
      """
        return self.calculate_total_energy() / (self.M * self.N)


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


class Calculation:
    def __init__(self, ising_model, kT=1, num_equil_sweeps=1000, num_sweeps=1000):
        """
        Initializing
        """
        self.num_equil_sweeps = num_equil_sweeps
        self.num_sweeps = num_sweeps
        self.kT = kT
        self.ising_model = ising_model

        self.energies_list = []
        self.total_spin_list = []

    def sweep(self):
        for site_i in range(self.ising_model.M):
            for site_j in range(self.ising_model.N):
                E_old = self.ising_model.calculate_energy_of_site(site_i, site_j)
                # flip spin i and j
                self.ising_model.flip_spin(site_i, site_j)
                # calculate updated energy
                E_new = self.ising_model.calculate_energy_of_site(site_i, site_j)
                # Monte Carlo step
                if np.random.random() <= np.exp(-(E_new - E_old) / self.kT):
                    # accept move
                    E_old = E_new
                else:
                    # reject move
                    # flip spin i and j back
                    self.ising_model.flip_spin(site_i, site_j)

    def record_observables(self):
        # calculate total energy
        e = self.ising_model.calculate_total_energy_per_spin()
        self.energies_list.append(e)

        # calculate total s2
        s = self.ising_model.calculate_total_spin_per_spin()
        self.total_spin_list.append(s)

    def get_average_energy(self):
        return np.mean(self.energies_list)

    def get_stderr_energy(self):
        return npd.std(self.energies_list) / np.sqrt(self.num_sweeps)

    def get_average_spin(self):
        return np.mean(self.total_spin_list)

    def get_stderr_spin(self):
        return np.std(self.total_spin_list) / np.sqrt(self.num_sweeps)

    def get_reblocked_avg_stderr_energy(self):
        reblock_data = pyblock.blocking.reblock(np.array(self.energies_list))
        opt = pyblock.blocking.find_optimal_block(self.num_sweeps, reblock_data)
        reblocked_data = reblock_data[opt[0]]
        return reblocked_data.mean, reblocked_data.std_err

    def get_reblocked_avg_stderr_spin(self):
        reblock_data = pyblock.blocking.reblock(np.array(self.total_spin_list))
        opt = pyblock.blocking.find_optimal_block(self.num_sweeps, reblock_data)
        if np.isnan(opt[0]):
            # reblocked_data = reblock_data[-1]
            # return reblocked_data.mean, reblocked_data.std_err
            means = []
            start = 0
            end = len(self.total_spin_list) // 5
            for i in range(4):
                means.append(np.mean(self.total_spin_list[start:end]))
                start = end
                end = start + len(self.total_spin_list) // 5
                print('!!')
                print(start,end)
            means.append(np.mean(self.total_spin_list[start::]))
            return np.mean(means), np.std(means) / np.sqrt(len(means))
        else:
            reblocked_data = reblock_data[opt[0]]
            return reblocked_data.mean, reblocked_data.std_err

    def run_calculation(self):
        for sweep in range(self.num_equil_sweeps):
            self.sweep()
        for sweep in range(self.num_sweeps):
            self.sweep()
            self.record_observables()


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
# plt.plot(range(a2.num_sweeps), a2.energies_list,label ='E')
# plt.plot(range(a2.num_sweeps), a2.total_spin_list,label ='S')
# print(np.average(a2.energies_list))
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
kT_list = np.arange(2.2, 2.5, 0.1)
# kT_list = np.arange(2.24, 2.3, 0.01)
# plot_kT_list = np.arange(2.24, 2.3, 0.001)
plot_kT_list = np.arange(1.2, 3.0, 0.001)
avg_s_list = []
stderr_s_list = []
n = 10
m = 10
J = 1
h = 0.00
num_equil=100
num_sample=1500
s_analytical = analytical(plot_kT_list, J)
a = IsingModel(n, m, J, h)
for kT in kT_list:
    print("kT={}".format(kT))
    a2 = Calculation(a, kT=kT, num_equil_sweeps=num_equil, num_sweeps=num_sample)
    # a2 = Wolff_Calculation(a, kT=kT, num_equil_sweeps=2000,num_sweeps= 1000)
    a2.run_calculation()
    mean, std_err = a2.get_reblocked_avg_stderr_spin()
    print("kT={}    mean {}    std_err {}".format(kT,mean,std_err))

    avg_s_list.append(np.abs(mean))
    stderr_s_list.append(std_err)
    s = a2.total_spin_list
    fig, ax = plt.subplots(tight_layout=True)
    fig.set_size_inches(8,5)
    n_bins = 25
    bins = np.linspace(-1,1,n_bins)
    ax.hist(s, bins=bins)
    ax.set_title("kT={:>4.2f} n={:>02d}, m={:>02d}, J={:>4.2f}, h={:>4.2f} samp={}".format(kT, n, m, J, h, num_sample))
    ax.set_xlim(-1.1,1.1)
    plt.tight_layout()
    plt.savefig(
        "metropolis_s_hist_kT_{:4.2f}_n_{:>02d}_m_{:>02d}_J_{:>4.2f}_h_{:>4.2f}_samp_{}.png".format(
            kT, n, m, J, h, num_sample
        ),
        dpi=300,
    )
    # plt.show()
    s = pd.Series(s)
    print("avg s autocorr = {}".format(s.autocorr()))
plt.figure(figsize=(8,5))
plt.errorbar(kT_list, avg_s_list, stderr_s_list, label="avg S")
plt.plot(plot_kT_list, s_analytical, label="analytical")
plt.title("n={:>02d}, m={:>02d}, J={:>4.2f}, h={:>4.2f} samp={}".format(n, m, J, h, num_sample))

plt.legend()
plt.tight_layout()
plt.savefig(
    "metropolis_avg_s_n_{:>02d}_m_{:>02d}_J_{:>4.2f}_h_{:>4.2f}_samp_{}.png".format(n, m, J, h, num_sample),
    dpi=300,
)
#plt.show()

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
