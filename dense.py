import numpy as np
class dense:
    def __init__(self, inputSize,outputSize):
        self.weight = np.random.randn(outputSize,inputSize)/10.0
        self.bias = np.random.randn(outputSize,1)/10.0
        self.weightChange = np.zeros_like(self.weight)
        self.biasChange = np.zeros_like(self.bias)
    def forward(self, input):
        self.input= input
        return np.dot(self.weight, self.input)+self.bias
    def backward(self, outputGradient, learningRate):
        weightsGradient = np.dot(outputGradient, self.input.T)
        inputGradient = np.dot(self.weight.T, outputGradient)
        self.weightChange -= learningRate * weightsGradient
        self.biasChange -= learningRate * outputGradient
        return inputGradient

    def Update(self, batch_size):

        # normalizing the change by the num of samples
        self.weightChange /= batch_size
        self.biasChange /= batch_size

        # Updating the params
        self.weight += self.weightChange
        self.bias += self.biasChange

        # Resets the change
        self.weightChange = np.zeros_like(self.weight)
        self.biasChange = np.zeros_like(self.bias)