from NeuralNetwork.NeuralNetwork import NeuralNetwork
import struct
import numpy as np
from os.path import isfile

def generateTrainingNetwork(neuralNetworkWeightsPath: str) -> NeuralNetwork:
    if neuralNetworkWeightsPath is None or neuralNetworkWeightsPath == "":
        return NeuralNetwork([784, 100, 50, 10])
    else:
        if not isfile(neuralNetworkWeightsPath):
            raise FileNotFoundError("File not found")
        else:
            return neuralNetworkFromFile(neuralNetworkWeightsPath)

def saveWeights(neuralNetwork: NeuralNetwork, filename: str):
    with open(filename, 'wb') as f:
        # Write the dimensions and hidden layer count
        f.write(struct.pack('i', len(neuralNetwork.layers)))
        for layer_size in neuralNetwork.layers:
            f.write(struct.pack('i', layer_size))
        
        # Write the weights
        for key in ["W1", "W2", "W3"]:
            np.save(f, neuralNetwork.params[key])

def neuralNetworkFromFile(filename: str) -> NeuralNetwork:
    layers = []
    params = {}

    if not isfile(filename):
        raise FileNotFoundError("File not found")

    with open(filename, 'rb') as f:
        # Read the dimensions and hidden layer count
        num_layers = struct.unpack('i', f.read(4))[0]
        layers = [struct.unpack('i', f.read(4))[0] for _ in range(num_layers)]
        
        # Read the weights
        for key in ["W1", "W2", "W3"]:
            params[key] = np.load(f)

    # Initialize the network with the read dimensions
    return NeuralNetwork(layers, params)