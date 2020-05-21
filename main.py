import sys
import numpy as np
# import matplotlib


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

layer1 = LayerDense(4, 5) # n_inputs = len(X[0]) and n_neurons = any
layer2 = LayerDense(5, 2) # n_inputs = n_neurons of previous layer and n_neurons = any
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation1.forward(layer2.output)
print(activation1.output)
