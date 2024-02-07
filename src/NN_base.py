import numpy as np
import math
import os, sys; sys.path.append(os.path.abspath('.\..'))
print(os.path.abspath('.\..'))

# from utils.activation_functions import *
# from utils.propagation_function import *
from utils.initialization import initialize_parameters_deep

class NN_base():
    """
    Custom neural network base class
    """
    def __init__(self, hidden_layers_size: list = [2,2], activation: 'str'= 'sigmoid', 
                 solver: 'str' = 'adam', learning_rate_init: float = 0.001,
                 max_iter: int = 100, seed: int = 23) -> None:
        """
        Args:
            hidden_layers_size(list): Number of neuron of hidden layers. [n_neur_layer_1, n_neur_layer_2, ..., n_neur_layer_n]
            activation(str): Activation function of the neurons
            solver(str): Solver to use in the weights optimization
            learning_rate_init(float): Learning rate of the weights
            max_iter(int): Max number of iterations of the gradiend descent
            seed(int): Seed used to initialize the weights
        """
        self.seed = seed
        self.parameters = initialize_parameters_deep(hidden_layers_size)
        pass

    def print_parameters(self):
        for key,value in self.parameters.items():
            print(f'{key}: {np.matrix(value)}\n')

    def fit(self, X: np.array, Y: np.array):
        pass


nn = NN_base(hidden_layers_size = [10,3,3,4,1], seed = 23)
nn.print_parameters()