import pandas as pd
import numpy as np
import math
from random import uniform

class NeuralNetwork(object):
    "Neural network with feed forward and back propagation"

    def __init__(self, topology, activation_scheme, momentum=1, learning_rate=1, weight_seed=None, custom_weights=None):
        self.topology = topology
        self.layers = self.__setup_layers(topology)
        self.weights = self.__setup_weights(topology, weight_seed, custom_weights)
        # self.output = output
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.activation_scheme = activation_scheme

        self.guess_list = list()
        self.error_list = list()
        self.overall_error_list = list()
        self.latent_layers = list()
        self.layers_with_weights = list()

        self.activation_functions = self.__setup_activation_functions()
        self.derivative_functions = self.__setup_derivative_functions()


    def __setup_layers(self, topology):
        topos = [None for x in topology]

        # topos[0] = np.matrix([float(x) for x in inputs])
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


    def feed_forward(self, inputs, output, activate_logging=False):
        "Perform one feed forward pass"

        def __compute_error_rate(guess, actual):
            return (guess - actual)

        # self.layers[0] = np.matrix([float(x) for x in inputs])
        self.layers[0] = np.matrix(inputs.astype(float))

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
        error_rate = error_rate_func(self.guess_list[-1], output)
        # print('Error rate: %s' % str(error_rate))
        self.error_list.append(error_rate)

        self.__generate_current_topology()

        # if activate_logging:
            # print('Guess: %s' % self.guess_list[-1])
            # print('Actual %s' % output)
            # print('Error rate: %s' % error_rate)

        return error_rate


    def back_propagation(self, data_point, error_rate, dp_counter, guess_pointer=-1, activate_logging=False):
        "Update weights using back prop"

        def __compute_gradient(y, error):
            return y * error

        # self.layers[0] = np.matrix([float(x) for x in data_point])
        self.layers[0] = np.matrix(data_point.astype(float))

        guess = self.guess_list[guess_pointer]

        #
        # Output to Hidden back propagation
        #
        prev_layer = self.layers[-2]
        prev_weight = self.weights[-1]

        key = self.activation_scheme[-1]
        y_derivative_func = np.vectorize(self.derivative_functions[key])
        y_derivative = y_derivative_func(guess)

        gradient_func = np.vectorize(__compute_gradient)
        gradients = gradient_func(y_derivative, error_rate)
        gradients_tr = gradients.transpose()

        delta_w = gradients_tr * prev_layer
        delta_w_tr = delta_w.transpose()

        new_weight = (self.momentum * prev_weight) - (self.learning_rate * delta_w_tr)

        self.weights[-1] = new_weight

        #
        # Hidden - Hidden / Hidden - Input back propagation
        #
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
            new_weight = (self.momentum * original_weight) - (self.learning_rate * delta_w)

            gradients_p = gradients_h_activated
            weights_p = original_weight

            self.weights[-i] = new_weight

            try:
                self.latent_layers[dp_counter] = layer
            except IndexError:
                self.latent_layers.append(layer)

        self.__generate_current_topology()


    def train(self, inputs, outputs, epochs=600, train_method='sequential'):
        "Train neural net. Requires epoch count parameter"

        def __sequential(inputs, outputs):
            overall_error = 0

            dp_counter = 0
            for data_point, actual in zip(inputs, outputs):
                error_rate = self.feed_forward(data_point, actual)
                self.back_propagation(data_point, error_rate, dp_counter)

                overall_error += error_rate
                dp_counter += 1

            overall_error = np.sum(overall_error) / len(inputs)
            self.overall_error_list.append(overall_error)

            # print('Overall error rate: %s' % str(overall_error))

        def __deferred_bp(inputs, outputs, epoch):
            error_avg = 0
            for data_point, actual in zip(inputs, outputs):
                error_rate = self.feed_forward(data_point, actual)
                # print(error_rate)
                error_avg += error_rate

            # print(error_avg)
            error_avg = error_avg / len(inputs)
            # print(error_avg)

            guess_pointer = -1
            for data_point in inputs:
                self.back_propagation(data_point, error_avg, guess_pointer=guess_pointer)
                guess_pointer = guess_pointer - 1
                # print('\n')

        train_methods = {
            'sequential' : __sequential,
            'deferred_bp' : __deferred_bp
        }

        train_func = train_methods[train_method]

        for i in range(0, epochs):
            # print('Epoch %i' % i)
            train_func(inputs, outputs)


    def run(self, data_point, actual):
        "Performs one feed forward pass with the trained neural net"

        error_rate = self.feed_forward(data_point, actual)
        # print('Feed Forward:')
        # print(str(self.layers_with_weights))
        print('Actual: %s' % str(actual))
        print('Guess: %s' % str(self.guess_list[-1]))
        # print('Error Rate: %s' % str(self.error_list[-1]))
        # print('Error Total: %s' % np.sum(self.error_list[-1]))

        return error_rate
        


def main():
    # Sample Usage
    # print('Problem 1:')
    #Problem 1
    topology = [4, 3, 4]
    inputs = [[0.9, 0.5, 0.1, 0.3], [0.2, 0.6, 0.4, 0.1]]
    output = [[0.9, 0.5, 0.1, 0.3], [0.2, 0.6, 0.4, 0.1]]
    activation_scheme = ['Sigmoid', 'Sigmoid']

    weight_1 = np.matrix([[0.7863690559, 0.4975437665, 0.9521735073],
                          [0.5775275116, 0.2028151628, 0.7669216083],
                          [0.07380019736, 0.4244236388, 0.1051142052],
                          [0.05272444907, 0.9716144298, 0.4978612697]])

    weight_2 = np.matrix([[0.8641701841, 0.7390354543, 0.5996454166, 0.4113275868],
                          [0.890385264, 0.9254203403, 0.5906074863, 0.3691760936],
                          [0.5945421458, 0.7382485887, 0.6031191925, 0.6168419889]])

    custom_weights = [weight_1, weight_2]

    neural_net = NeuralNetwork(topology, activation_scheme, custom_weights=custom_weights)
    neural_net.train(inputs, output, epochs=600, train_method='sequential')
    neural_net.run([0.9, 0.5, 0.1, 0.3], [0.9, 0.5, 0.1, 0.3])

if __name__ == "__main__":
    main()
