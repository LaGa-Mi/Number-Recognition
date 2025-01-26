from ImageUtils import imageToInput
from os.path import isfile
from NeuralNetwork.NeuralNetwork import NeuralNetwork

def identify(neuralNetwork: NeuralNetwork, imagePath: str, neuralNetworkWeightsPath: str) -> int:
    if not isfile(imagePath):
        print("Image not found")
        return -1

    imgInput = imageToInput(imagePath)
    result = neuralNetwork.getPrediction(imgInput)

    return result