import numpy as np
import matplotlib.pyplot as plt
import pyblock


def plot_lattice(lattice_state):
    """
    Plot lattice configuration
    """
    plt.figure()

    imgplot = plt.imshow(lattice_state)
    imgplot.set_interpolation('none')

    M = len(lattice_state[0])
    N = len(lattice_state)
    plt.xticks(range(N))
    plt.yticks(range(M))

    for i in range(N+1):
        plt.plot([i-0.5, i-0.5], [0-0.5, M-0.5], color='black')
    for j in range(M+1):
        plt.plot([0-0.5, N-0.5], [j-0.5, j-0.5], color='black')

    plt.show()

def print_params(ising_parameters):
    """
    Print lattice attributes
    """
    print("\t{} by {} lattice".format(ising_parameters['M'], ising_parameters['N']))
    print(
        "\tJ = {}   (positive means a favorable interaction)".format(ising_parameters['J']))
    print("\th = {}   (external field aligned with spins)".format(ising_parameters['h']))

def test_calculate_energy_of_sites():
    pass

def test_calculate_lattice_energy_per_spin():
    pass

def get_reblocked_avg_stderr_energy(sr,sp):
    reblock_data = pyblock.blocking.reblock(np.array(sr['energies_list']))
    opt = pyblock.blocking.find_optimal_block(sp['num_sample_sweeps'], reblock_data)
    if np.isnan(opt[0]):
        means = []
        start = 0
        end = len(sr['energies_list']) // 5
        for i in range(4):
            means.append(np.mean(sr['energies_list'][start:end]))
            start = end
            end = start + len(sr['energies_list']) // 5
        means.append(np.mean(sr['energies_list'][start::]))
        return np.mean(means), np.std(means) / np.sqrt(len(means))
    else:
        reblocked_data = reblock_data[opt[0]]
        return reblocked_data.mean, reblocked_data.std_err
    reblocked_data = reblock_data[opt[0]]
    return reblocked_data.mean, reblocked_data.std_err

def get_reblocked_avg_stderr_spin(sr,sp):
    reblock_data = pyblock.blocking.reblock(np.array(sr['total_spin_list']))
    opt = pyblock.blocking.find_optimal_block(sp['num_sample_sweeps'], reblock_data)
    if np.isnan(opt[0]):
        means = []
        start = 0
        end = len(sr['total_spin_list']) // 5
        for i in range(4):
            means.append(np.mean(sr['total_spin_list'][start:end]))
            start = end
            end = start + len(sr['total_spin_list']) // 5
        means.append(np.mean(sr['total_spin_list'][start::]))
        return np.mean(means), np.std(means) / np.sqrt(len(means))
    else:
        reblocked_data = reblock_data[opt[0]]
        return reblocked_data.mean, reblocked_data.std_err
