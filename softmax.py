import numpy as np
class softmax:
    def __init__(self):
        pass

    def forward(self,input):
        expValues = np.exp(input-np.max(input))
        self.output = expValues/np.sum(expValues)
        return self.output


    """def backward(self, y_true,learningRate):
        dOutput = self.output
        dOutput[y_true] -= 1
        return dOutput"""

    def backward(self, y_true, learningRate):
        y_true_one_hot = np.zeros_like(self.output)
        y_true_one_hot[y_true] = 1
        dOutput = self.output - y_true_one_hot
        return dOutput

    def Update(self, batch_size):
        pass

