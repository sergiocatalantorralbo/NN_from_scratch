import numpy as np
import math
import os, sys; sys.path.append(os.path.abspath('./.'))
# print(os.path.abspath('./.'))

# from utils.activation_functions import *
from utils.propagation_function import linear_forward,linear_backward
from utils.activation_functions import relu,sigmoid,relu_backward,sigmoid_backward

from utils.initialization import initialize_parameters_deep
import os, sys; sys.path.append(os.path.abspath('.\..\..'))

from utils.load_datasets import load_cats_dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_cats_dataset()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

class NeuralNetworkBase():
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



    def _linear_activation_forward(self,A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Args:
            A_prev(array): activations from previous layer (or input data): (size of previous layer, number of examples)
            W(array): weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b(array): bias vector, numpy array of shape (size of the current layer, 1)
            activation(str): the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
            A(array): the output of the activation function, also called the post-activation value 
            cache(tuple): a python dictionary containing "linear_cache" and "activation_cache" (arrays);
                    stored for computing the backward pass efficiently
        """
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
            
        else:
            print("\033[91mError! Please make sure you have passed the value correctly in the \"activation\" parameter")
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache


    def _forward_propagation(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Args:
            X(array): data, numpy array of shape (input size, number of examples)
        
        Returns:
            output(array): last post-activation value
            caches(list): list of caches containing:
                ·every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                ·the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """
        caches = []
        A = X
        L = len(self.parameters) // 2
        for l in range(1, L):
            A_prev = A
            l_parameter_W = self.parameters['W' + str(l)]
            l_parameter_b = self.parameters['b' + str(l)]
            A, cache = self._linear_activation_forward(A_prev, l_parameter_W, l_parameter_b, activation = "relu")
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        L_parameter_W = self.parameters['W' + str(L)]
        L_parameter_b = self.parameters['b' + str(L)]
        output, cache = self._linear_activation_forward(A, L_parameter_W, L_parameter_b, activation = "sigmoid")
        caches.append(cache)
        
        assert(output.shape == (1,X.shape[1]))

        return output, caches
    
    def _compute_cost(self, output, Y):
        """
        Implement the cost function.

        Args:
            output -- probability vector corresponding to your label predictions, shape (1, number of examples)
            Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
            cost -- cross-entropy cost
        """
        
        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = (1./m) * (-np.dot(Y,np.log(output).T) - np.dot(1-Y, np.log(1-output).T))
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost

    def _linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Args:
            dA -- post-activation gradient for current layer l 
            cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)
            
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)
            
        else:
            print("\033[91mError! Please make sure you have passed the value correctly in the \"activation\" parameter")
        
        return dA_prev, dW, db

    def _backward_propagation(self, output, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
            AL(array): probability vector, output of the forward propagation (L_model_forward())
            Y(array): true "label" vector (containing 0 if non-cat, 1 if cat)
            caches(list): list of caches containing:
                · every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                · the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
        
        Returns:
            grads(dict): A dictionary with the gradients
                ·grads["dA" + str(l)] = ... 
                ·grads["dW" + str(l)] = ...
                ·grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = output.shape[1]
        Y = Y.reshape(output.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = - (np.divide(Y, output) - np.divide(1 - Y, 1 - output))
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self._linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def _update_parameters(self, grads: dict):
        """
        Update parameters using gradient descent
        
        Arguments:
            grads(dict): python dictionary containing your gradients, output of L_model_backward
        
        Returns:
            parameters(dict): python dictionary containing your updated parameters 
                    parameters["W" + str(l)] = ... 
                    parameters["b" + str(l)] = ...
        """
        
        L = len(self.parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - self.learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - self.learning_rate * grads["db" + str(l+1)]

    def _iteration(self, X: np.array, Y: np.array):
        output, activation_cache = self._forward_propagation(X)
        cost = self._compute_cost(output, Y)
        grads = self._backward_propagation(output, Y, activation_cache)
        self._update_parameters(grads)
        return cost
    
    def _iterator(self, X: np.array, Y: np.array, print_cost):
        for i in range(self.max_iter):
            cost = self._iteration(X, Y)
            if print_cost and i % 10 == 0 or i == self.max_iter - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

    def fit(self, X: np.array, Y: np.array, print_cost: bool = True):
        self._iterator(X, Y, print_cost)


first_layer_size = train_set_x.shape[0]
layers_dims = [first_layer_size, 20, 7, 5, 1]
nn = NeuralNetworkBase(layers_dims, learning_rate_init = 0.0075, 
             max_iter = 2500, seed = 23)
nn.fit(train_set_x, train_set_y)