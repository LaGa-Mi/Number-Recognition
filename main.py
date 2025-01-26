from os.path import join
from NeuralNetwork.Training import Trainer
from NeuralNetwork.Using import identify
from NeuralNetwork.NeuralNetworkStorage import generateTrainingNetwork, saveWeights
from NeuralNetwork.NeuralNetworkStorage import neuralNetworkFromFile
from NeuralNetwork.NeuralNetwork import NeuralNetwork

train: bool = False

saveUpdatedWeights: bool = True
epochs: int = 10
learningRate: float = 0.2

imageFolder = "./Tests/"
imagePath = join(imageFolder, "tres2.png")

neuralNetworkWeightsFolderPath = "./Weights/"
neuralNetworkWeightsPath = join(neuralNetworkWeightsFolderPath, "weights.nnw")
saveNeuralNetworkWeightsPath = join(neuralNetworkWeightsFolderPath, "weights.nnw")

def main():
    inputPath = "./Dataset/"
    mnist_test_images = join(inputPath, "t10k-images.idx3-ubyte")
    mnist_test_labels = join(inputPath, "t10k-labels.idx1-ubyte")
    mnist_train_images = join(inputPath, "train-images.idx3-ubyte")
    mnist_train_labels = join(inputPath, "train-labels.idx1-ubyte")

    if train:
        networkToBeTrained: NeuralNetwork = generateTrainingNetwork(neuralNetworkWeightsPath)

        trainer: Trainer = Trainer(networkToBeTrained, mnist_train_images, mnist_train_labels, mnist_test_images, mnist_test_labels)

        trainer.train(epochs, learningRate)

        if saveUpdatedWeights:
            saveWeights(trainer.neuralNetwork, saveNeuralNetworkWeightsPath)
    else:
        neuralNetwork: NeuralNetwork = neuralNetworkFromFile(neuralNetworkWeightsPath)
        print("Image was identified as the number: {}".format(identify(neuralNetwork, imagePath, neuralNetworkWeightsPath)))

if __name__ == "__main__":
    main()