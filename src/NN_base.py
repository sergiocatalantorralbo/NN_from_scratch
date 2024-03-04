import numpy as np
import math
import os, sys; sys.path.append(os.path.abspath('./.'))
# print(os.path.abspath('./.'))

# from utils.activation_functions import *
from utils.propagation_function import L_model_forward, compute_cost, update_parameters, L_model_backward
from utils.initialization import initialize_parameters_deep
import os, sys; sys.path.append(os.path.abspath('.\..\..'))

from utils.load_datasets import load_cats_dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_cats_dataset()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

class NNBase():
    """
    Custom neural network base class
    """
    def __init__(self, hidden_layers_size: list = [2,2], activation: 'str'= 'sigmoid', 
                 solver: 'str' = 'adam', learning_rate_init: float = 0.001,
                 max_iter: int = 100, seed: int = 23) -> None:
        """
        Args:
            hidden_layers_size(list): Number of neuron of hidden layers. 
            [n_neur_layer_1, n_neur_layer_2, ..., n_neur_layer_n]
            activation(str): Activation function of the neurons
            solver(str): Solver to use in the weights optimization
            learning_rate_init(float): Learning rate of the weights
            max_iter(int): Max number of iterations of the gradiend descent
            seed(int): Seed used to initialize the weights
        """
        self.seed = seed
        self.hidden_layers_size = hidden_layers_size
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter
        self.parameters = initialize_parameters_deep(self.hidden_layers_size)

    def _forward_propagation(self, X):
        output, activation_cache = L_model_forward(X, self.parameters)
        return output, activation_cache

    def _update_parameters(self, grads):
        parameters = update_parameters(self.parameters, grads, self.learning_rate)
        return parameters

    def _iteration(self, X: np.array, Y: np.array):
        output, activation_cache = self._forward_propagation(X)
        cost = compute_cost(output, Y)
        grads = L_model_backward(output, Y, activation_cache)
        self.parameters = self._update_parameters(grads)
        return cost
    
    def _iterator(self, X: np.array, Y: np.array, print_cost):
        for i in range(self.max_iter):
            cost = self._iteration(X, Y)
            self._iteration(X, Y)
            if print_cost and i % 10 == 0 or i == self.max_iter - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

    def fit(self, X: np.array, Y: np.array, print_cost: bool = True):
        self._iterator(X, Y, print_cost)


first_layer_size = train_set_x.shape[0]
layers_dims = [first_layer_size, 20, 7, 5, 1]
nn = NNBase(layers_dims, learning_rate_init = 0.0075, 
             max_iter = 100, seed = 23)
nn.fit(train_set_x, train_set_y)