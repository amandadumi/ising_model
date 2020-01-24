import numpy as np
import matplotlib.pyplot as plt


def plot_lattice(IsingModel):
    """
    Plot lattice configuration
    """
    plt.figure()

    imgplot = plt.imshow(IsingModel.lattice_state)
    imgplot.set_interpolation('none')

    plt.xticks(range(IsingModel.N))
    plt.yticks(range(IsingModel.M))

    for i in range(IsingModel.N+1):
        plt.plot([i-0.5, i-0.5], [0-0.5, IsingModel.M-0.5], color='black')
    for j in range(IsingModel.M+1):
        plt.plot([0-0.5, IsingModel.N-0.5], [j-0.5, j-0.5], color='black')

    plt.show()


def test_calculate_energy_of_sites():
    pass

def test_calculate_lattice_energy_per_spin():
    pass