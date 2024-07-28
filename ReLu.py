import numpy as np

class ReLu:
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, outputGradient, learningRate):
        self.inputGradient = outputGradient.copy()
        self.inputGradient[self.input<=0]=0
        return self.inputGradient
    def Update(self, batch_size):
        pass