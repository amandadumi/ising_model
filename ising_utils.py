import numpy as np
import matplotlib.pyplot as plt
import pyblock


def plot_lattice(lattice_state):
    """
    Plot lattice configuration
    """
    plt.figure()

    imgplot = plt.imshow(lattice_state, vmin=-1, vmax=1)
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

def check_exercise_1(ip, lattice):
    # hide this
    try: 
        ip
        print("Check: OK")
        print("The variable 'ip' exists.")
    except NameError:
        print("Check: FAILED")
        print("The variable 'ip' is not defined.")
    
    try: 
        assert type(ip) is dict
        print("Check: OK")
        print("The variable 'ip' is a dictionary.")
    except:
        print("Check: FAILED")
        print("The variable 'ip' is not a dictionary. It has been defined as a {}".format(type(ip)))
        
    try: 
        lattice
        print("Check: OK")
        print("The variable 'lattice' exists.")
    except NameError:
        print("Check: FAILED")
        print("The variable 'lattice' is not defined.\nDid you forget to store the result of the function in a variable named 'lattice'?")
        
    try: 
        assert type(lattice) is np.ndarray
        print("Check: OK")
        print("The variable 'lattice' is a numpy array.")
    except:
        print("Check: FAILED")
        print("The variable 'lattice' is not a numpy array. It has been defined as a {}.".format(type(lattice)))
    
    try:
        assert list(ip.keys()) == ["J","h","M","N"]
        for i in ["J","h","M","N"]:
            assert ip[i] != None
        print("Check: OK")
        print("The dictionary 'ip' has the right keys and they aren't 'None'.")
    except:
        print("Check: FAILED")
        print("The parameters dictionary 'ip' doesn't have all of the parameters defined or at least one of them is 'None'. ['J','h','M','N']")
        iu.print_params(ip)
        
    try:
        assert ip['M'] == 5
        assert ip['N'] == 8
        assert ip['J'] == 2
        assert ip['h'] == 0.01
        print("Check: OK")
        print("The variable 'ip' has all the right values in it.")
    except:
        print("Check: FAILED")
        print("At least one of the parameters in the dictionary 'ip' is incorrect.")
        iu.print_params(ip)
    
    try:
        assert np.shape(lattice)[0] == 5
        assert np.shape(lattice)[1] == 8
        assert lattice.all() == 1
        print("Check: OK")
        print("The shape of 'lattice' is correct and is all set to 1.")
    except:
        print("Check: FAILED")
        print("The shape of 'lattice' is incorrect or all values are not 1.")
        iu.print_params(ip)

def check_exercise_2(ip, example_lattice):
    try:
        assert ip['M'] == 3
        assert ip['N'] == 3
        assert ip['J'] == 1
        assert ip['h'] == 0
        print("Check: OK")
        print("The variable 'ip' has all the right values in it.")
    except:
        print("Check: FAILED")
        print("At least one of the parameters in the dictionary 'ip' is incorrect.")
        iu.print_params(ip)
    
    try:
        assert np.shape(example_lattice)[0] == 3
        assert np.shape(example_lattice)[1] == 3
        print("Check: OK")
        print("The shape of 'lattice' is correct.")
    except:
        print("Check: FAILED")
        print("The shape of 'lattice' is incorrect.")
    
    try:
        assert example_lattice[0,1] == -1
        assert example_lattice[2,2] == -1
        print("Check: OK")
        print("The correct spins of 'lattice' were flipped.")
    except:
        print("Check: FAILED")
        print("Spins (0,1) and/or (2,2) were not flipped.")
    
    try:
        assert np.sum(example_lattice) == ((ip['M']*ip['N'] )-4)
        print("Check: OK")
        print("No incorrect spins were flipped.")
    except:
        print("Check: FAILED")
        print("Extra spins besides (0,1) and/or (2,2) were flipped!")

def check_exercise_3(initialize_lattice_state, flip_spin, calculate_energy_of_site):
    try:
        print("Testing function with no external field")
        ip = {"J": 1, "h": 0.0, "M": 3, "N": 3}
        example_lattice = initialize_lattice_state(ip) 
        example_lattice = flip_spin(example_lattice,1,1)
        print_params(ip)
        plot_lattice(example_lattice)
        correct = np.array([[-2.0, -1.0, -2.0],[-1.0,2.0,-1.0],[-2.0,-1.0,-2.0]])
        calculated = []
        for i in range(ip["M"]):
            for j in range(ip["N"]):
                calculated.append(calculate_energy_of_site(example_lattice,ip,i,j))
        calculated = np.array(calculated).reshape(ip["M"],ip["N"])
        print("Expected:")
        print(correct)
        print("Calculated:")
        print(calculated)
        assert np.allclose(calculated, correct)
        print("Test passed!")
    except:
        print("Test failed!")
    print("")
    try:
        print("Testing function with an external field")
        ip = {"J": 1, "h": 0.01, "M": 3, "N": 3}
        example_lattice = initialize_lattice_state(ip) 
        example_lattice = flip_spin(example_lattice,1,1)
        print_params(ip)
        plot_lattice(example_lattice)
        correct = np.array([[-2.01, -1.01, -2.01],[-1.01,2.01,-1.01],[-2.01,-1.01,-2.01]])
        calculated = []
        for i in range(ip["M"]):
            for j in range(ip["N"]):
                calculated.append(calculate_energy_of_site(example_lattice,ip,i,j))
        calculated = np.array(calculated).reshape(ip["M"],ip["N"])
        print("Expected:")
        print(correct)
        print("Calculated:")
        print(calculated)
        assert np.allclose(calculated, correct)
        print("Test passed!")
    except:
        print("Test failed!")
def check_exercise_4(metropolis_test):
    # THIS IS THE TEST FOR THE FUNCTION THE STUDENTS WRITE
    # set some parameters
    sp = {"num_equil_sweeps": None, "num_sweeps": None, "kT": 1}
    E_new = 0.47
    E_old = 0.1
    # since this function is probabilistic we will sample it ~N times 
    # and check that we are within 3 sigma of the expected value
    nsamp = 100000
    # expected
    P_accept = np.exp(-(0.47 - 0.1) / sp["kT"])
    # calculated (metropolis_test is run nsamp times and the mean of the trues (1) and falses (0) gives the probability)
    P_func = np.mean(np.array([metropolis_test(sp, E_new, E_old) for i in range(nsamp)]))
    # assert it is within 3 sigma
    assert np.abs(P_func-P_accept) <= 3*(1.0/np.sqrt(nsamp))

