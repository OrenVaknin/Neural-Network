import numpy as np
class flat:
    def __init__(self):
        pass



    def forward(self, input):
        self.input_shape = np.shape(input)
        final_shape=1
        for i in self.input_shape:
            final_shape *= i
        self.output_shape = (final_shape,1)
        return np.reshape(input, self.output_shape)



    def backward(self, y_true,learningRate):
        return y_true
    def Update(self, batch_size):
        pass

