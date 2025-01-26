import numpy as np
from NeuralNetwork.NeuralNetworkUtils import *

class NeuralNetwork:
    def __init__(self, layers: list, params: dict = None):
        self.layers = layers

        inputLayerSize = layers[0]
        hiddenLayer1Size = layers[1]
        hiddenLayer2Size = layers[2]
        outputLayerSize = layers[3]

        if params is None:
            self.params = {
                "W1": np.random.randn(hiddenLayer1Size, inputLayerSize) * np.sqrt(1. / hiddenLayer1Size),
                "W2": np.random.randn(hiddenLayer2Size, hiddenLayer1Size) * np.sqrt(1. / hiddenLayer2Size),
                "W3": np.random.randn(outputLayerSize, hiddenLayer2Size) * np.sqrt(1. / outputLayerSize)     
            }
        else:
            self.params = params

    def forwardPass(self, x_train):
        paramsCopy = self.params

        paramsCopy["A0"] = x_train

        # input -> hidden 1
        paramsCopy["Z1"] = np.dot(paramsCopy["W1"], paramsCopy["A0"])
        paramsCopy["A1"] = sigmoid(paramsCopy["Z1"])

        # input -> hidden 1
        paramsCopy["Z2"] = np.dot(paramsCopy["W2"], paramsCopy["A1"])
        paramsCopy["A2"] = sigmoid(paramsCopy["Z2"])

        # hidden 2 -> output
        paramsCopy["Z3"] = np.dot(paramsCopy["W3"], paramsCopy["A2"])
        paramsCopy["A3"] = sigmoid(paramsCopy["Z3"])

        return paramsCopy["Z3"]
    
    def backwardPass(self, y_train, output):
        paramsCopy = self.params

        change_w = {}

        # calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * softMaxDerivative(paramsCopy["Z3"])
        change_w["W3"] = np.outer(error, paramsCopy["A2"])

        # calculate W2 update
        error = np.dot(paramsCopy["W3"].T, error) * sigmoidDerivative(paramsCopy["Z2"])
        change_w["W2"] = np.outer(error, paramsCopy["A1"])

        # calculate W1 update
        error = np.dot(paramsCopy["W2"].T, error) * sigmoidDerivative(paramsCopy["Z1"])
        change_w["W1"] = np.outer(error, paramsCopy["A0"])

        return change_w
    
    def updateWeights(self, change_w: dict, learningRate: float):
        for key, value in change_w.items():
            self.params[key] -= learningRate * value

    def computeAccuracy(self, testImages, testLabels):
        predictions = []

        for i in range(len(testLabels)):
            inputs = np.asarray(a = testImages[i], dtype = float).flatten() / 255.0 * 0.99 + 0.01
            targets = np.zeros(10) + 0.01
            targets[int(testLabels[i])] = 0.99
            output = self.forwardPass(inputs)
            predictions.append(np.argmax(output) == np.argmax(targets))
        
        return np.mean(predictions)
    
    def getPrediction(self, imgInput):
        output = self.forwardPass(imgInput)
        print("Output: {}".format(output))
        return np.argmax(output)

    def __eq__(self, value,) -> bool:
        equals: bool = True

        equals &= self.layers == value.layers
        for key, val in self.params.items():
            if (key.startswith("W")):
                equals &= np.array_equal(val, value.params[key])

        return equals