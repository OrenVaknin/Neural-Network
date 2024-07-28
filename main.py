import numpy as np
import dense
import flat
import ReLu
import softmax
import norm
import pickle
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#taking a smaller database
x_train = x_train[:2000].astype('float32')
y_train = y_train[:2000]
x_test = x_test[:100].astype('float32')
y_test = y_test[:100]
#i dont know if neccesary
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = np.array(x_train)
y_train = np.array(y_train)
#Noramlizing
x_train /=255.0
x_test /=255.0
batch_size = 100
numOfBatches = x_train.shape[0] // batch_size
x_trainBatches = np.array(np.array_split(x_train, numOfBatches))
y_trainBatches = np.array(np.array_split(y_train, numOfBatches))
#Categorical Cross entropy
def categoricalCrossEntropy(y, y_pred):
    y_predClipped = np.clip(y_pred, 1e-100, 1 - 1e-100)
    y_hat = y_predClipped[y]
    return -np.log(y_hat)[0]

def neuralNetwork1():
    return [flat.flat(),dense.dense(784,30),ReLu.ReLu(),norm.Norm()
            ,dense.dense(30,35),ReLu.ReLu(), norm.Norm(),
            dense.dense(35,30),ReLu.ReLu(),norm.Norm(),dense.dense(30,10), softmax.softmax()]


def run(x_trainBatches, y_trainBatches, numOfBatches):
    network = neuralNetwork1()
    epochs =20000
    learningRate = 0.1
    for i in range (epochs):
        error = 0
        for batch in range(numOfBatches):
            for (x,y) in (zip(x_trainBatches[batch],y_trainBatches[batch])):
                output = x
                for layer in network:
                    output= layer.forward(output)
                error += categoricalCrossEntropy(y, output)
                grad = y
                for layer in reversed(network):
                    grad=layer.backward(grad, learningRate)
        for layer in network: layer.Update(batch_size)
        error/= len(x_train)
        if i%25 == 0 :print("error is:", error, "epoch is:", i)
    save(network)

def save(network):
    pickle_out = open("Network.pickle", "wb")
    pickle.dump(network, pickle_out)
    pickle_out.close()
def load():
    pickle_in = open("Network.pickle", "rb")
    network = pickle.load(pickle_in)
    pickle_in.close()
    return network
run(x_trainBatches, y_trainBatches, numOfBatches)