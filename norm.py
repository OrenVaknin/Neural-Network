import sys
import numpy as np
class Norm:

    def forward(self, inputs):
        epsilon = sys.float_info.epsilon
        mean = inputs.mean(axis=0)
        var = inputs.var(axis=0)
        std = np.sqrt(var + epsilon)
        inputsNorm = (inputs - mean) / std
        return inputsNorm

    def backward(self, output_gradient, learning_rate):
        return output_gradient

    def Update(self, batch_size):
        pass
