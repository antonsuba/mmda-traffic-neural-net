import pandas as pd
import numpy as np
import math
from random import uniform

class NeuralNetwork(object):
    "Neural network with feed forward and back propagation"

    def __init__(self, topology, inputs, output, activation_scheme, weight_seed=None, custom_weights=None):
        self.topology = topology
        self.layers = self.__setup_layers(topology, inputs)
        self.weights = self.__setup_weights(topology, weight_seed, custom_weights)
        self.output = output
        self.activation_scheme = activation_scheme

        self.guess_list = list()
        self.error_list = list()
        self.layers_with_weights = list()

        self.activation_functions = self.__setup_activation_functions()
        self.derivative_functions = self.__setup_derivative_functions()


    def __setup_layers(self, topology, inputs):
        topos = [None for x in topology]

        topos[0] = np.matrix([float(x) for x in inputs])
        return topos


    def __setup_weights(self, topology, weight_seed, custom_weights):
        weights = list()

        if custom_weights:
            weights = custom_weights

        else:
            starting_weight = weight_seed if weight_seed is not None else uniform(0.00, 0.99)

            for layer_a, layer_b in zip(topology, topology[1:]):
                weights_arr = np.full((layer_a, layer_b), starting_weight)
                weights_matrix = np.matrix(weights_arr)
                weights.append(weights_matrix)

        return weights


    def __setup_activation_functions(self):
        def __relu_activation(x):
            return x if x > 0 else 0

        def __sigmoid_activation(x):
            return 1 / (1 + math.exp(-1 * x))

        activation_functions = {
            'Relu' : __relu_activation,
            'Sigmoid' : __sigmoid_activation
        }

        return activation_functions


    def __setup_derivative_functions(self):
        def __relu_derivative(x):
            return 1 if x > 0 else 0

        def __sigmoid_derivative(x):
            return x * (1 - x)

        derivative_functions = {
            'Relu' : __relu_derivative,
            'Sigmoid' : __sigmoid_derivative
        }

        return derivative_functions


    def __generate_current_topology(self):
        "Generate list of layers and weights"

        layers_with_weights = [None] * (len(self.layers) + len(self.weights))
        layers_with_weights[::2] = self.layers
        layers_with_weights[1::2] = self.weights

        self.layers_with_weights = layers_with_weights


    def feed_forward(self):

        def __compute_error_rate(guess, actual):
            return (guess - actual) ** 2

        for i in range(0, len(self.weights)):
            result_matrix = self.layers[i] * self.weights[i]

            key = self.activation_scheme[i]
            activation_function = np.vectorize(self.activation_functions[key])
            activated_matrix = activation_function(result_matrix)

            self.layers[i + 1] = activated_matrix

        #Append last layer as guess
        self.guess_list.append(self.layers[-1])

        #Compute error rate
        error_rate_func = np.vectorize(__compute_error_rate)
        error_rate = error_rate_func(self.guess_list[-1], self.output)
        print('Error rate: %s' % str(error_rate))
        self.error_list.append(error_rate)

        self.__generate_current_topology()


    def back_propagation(self):

        def __compute_gradient(y, error):
            return y * error

        guess = self.guess_list[-1]
        error_rate = self.error_list[-1]

        #First part of back propagation
        prev_layer = self.layers[-2]
        prev_weight = self.weights[-1]

        key = self.activation_scheme[-1]
        y_derivative_func = np.vectorize(self.derivative_functions[key])
        y_derivative = y_derivative_func(guess)

        print('Y Derivative: %s' % str(y_derivative))

        gradient_func = np.vectorize(__compute_gradient)
        gradients = gradient_func(y_derivative, error_rate)
        # print('Gradients: %s' % str(gradients))
        gradients_tr = gradients.transpose()

        delta_w = gradients_tr * prev_layer
        # print('Delta Weights: %s' % str(delta_w))
        delta_w_tr = delta_w.transpose()

        new_weight = prev_weight - delta_w_tr

        self.weights[-1] = new_weight


        #Second Part of back propagation
        gradients_p = gradients
        weights_p = prev_weight

        for i in range(2, len(self.weights) + 1):
            layer = self.layers[-i]
            layer_next = self.layers[-(i+1)]

            key = self.activation_scheme[-i]
            derivative_function = np.vectorize(self.derivative_functions[key])
            z_hat = derivative_function(layer)

            weights_p_tr = weights_p.transpose()
            gradients_h = gradients_p * weights_p_tr
            gradients_h_activated = gradient_func(gradients_h, z_hat)

            layer_next_tr = layer_next.transpose()
            delta_w = layer_next_tr * gradients_h_activated

            original_weight = self.weights[-i]
            new_weight = original_weight - delta_w

            gradients_p = gradients_h_activated
            weights_p = original_weight

            self.weights[-i] = new_weight

        self.__generate_current_topology()


    def train(self, epochs):
        for i in range(0, epochs):
            self.feed_forward()
            print('Feed forward %i' % i)            
            print(str(self.layers_with_weights))

            self.back_propagation()
            print('Back propagation %i' % i)            
            print(str(self.layers_with_weights))


    def run(self, label):
        self.feed_forward()
        print('Final Feed Forward:')
        print(str(self.layers_with_weights))
        print('Final Guess:')
        print(str(self.guess_list[-1]))
        print('Final Error rate')
        print(str(self.error_list[-1]))

        # print('Error rate list: %s' % str(self.error_list))

        # with open('output.txt', 'w') as outfile:
        #     outfile.write('# %s:' % label)

        #     for i in range(0, len(self.layers_with_weights)):
        #         if i % 2 == 0:
        #             outfile.write('Layer:')
        #         else:
        #             outfile.write('Weight:')
                
        #         for data_slice in self.layers_with_weights[i]:
        #             arr = np.array(data_slice)
        #             np.savetxt(outfile, arr, fmt='%-7.5f')


def main():
    # print('Problem 1:')
    # #Problem 1
    # topology = [3, 2, 3, 2]
    # inputs = [1, 0, 1]
    # output = [1, 0]
    # activation_scheme = ['Relu', 'Relu', 'Sigmoid']

    # neural_net = NeuralNetwork(topology, inputs, output, activation_scheme)
    # neural_net.train(2)
    # neural_net.run('Problem 1')
    
    # print('\n')

    # print('Problem 2')
    # #Problem 2
    # topology = [5, 4, 3, 2]
    # inputs = [1, 0, 0.8, 0.9, 0.8]
    # output = [1, 0]
    # activation_scheme = ['Sigmoid', 'Sigmoid', 'Sigmoid']

    # neural_net = NeuralNetwork(topology, inputs, output, activation_scheme)
    # neural_net.train(2)
    # neural_net.run('Problem 2')

    # print('\n')

    # print('Problem 3')
    # #Problem 3
    # topology = [5, 3, 2, 3, 5]
    # inputs = [0.9, 0.8, 0.9, 0.2, 0.3]
    # output = [0.9, 0.8, 0.9, 0.2, 0.3]
    # activation_scheme = ['Relu', 'Relu', 'Relu', 'Relu']

    # neural_net = NeuralNetwork(topology, inputs, output, activation_scheme)
    # neural_net.train(2)
    # neural_net.run('Problem 3')

    print('Problem 1:')
    #Problem 1
    topology = [4, 3, 4]
    inputs = [0.9, 0.5, 0.1, 0.3]
    output = [0.9, 0.5, 0.1, 0.3]
    activation_scheme = ['Relu', 'Sigmoid']

    weight_1 = np.matrix([[0.7863690559, 0.4975437665, 0.9521735073],
                          [0.5775275116, 0.2028151628, 0.7669216083],
                          [0.07380019736, 0.4244236388, 0.1051142052],
                          [0.05272444907, 0.9716144298, 0.4978612697]])

    weight_2 = np.matrix([[0.8641701841, 0.7390354543, 0.5996454166, 0.4113275868],
                          [0.890385264, 0.9254203403, 0.5906074863, 0.3691760936],
                          [0.5945421458, 0.7382485887, 0.6031191925, 0.6168419889]])

    custom_weights = [weight_1, weight_2]

    neural_net = NeuralNetwork(topology, inputs, output, activation_scheme, None, custom_weights)
    neural_net.train(2)
    neural_net.run('Problem 1')


if __name__ == "__main__":
    main()
