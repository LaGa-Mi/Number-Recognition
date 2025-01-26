import time
from NeuralNetwork.NeuralNetwork import NeuralNetwork
import numpy as np
from DataHandler import DataHandler

class Trainer:
    def __init__(self, neuralNetwork: NeuralNetwork, train_images_path: str, train_labels_path: str, test_images_path: str, test_labels_path: str):
        self.neuralNetwork = neuralNetwork
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path

    def train(self, epochs: int, trainingRate: float) -> None:
        (trainImages, trainLabels), (testImages, testLabels) = DataHandler.load_data(self.train_images_path, self.train_labels_path, self.test_images_path, self.test_labels_path)

        startTime = time.time()
        for epoch in range(epochs):
            for i in range(len(trainLabels)):
                inputs = np.asarray(a = trainImages[i], dtype = float).flatten() / 255.0 * 0.99 + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(trainLabels[i])] = 0.99
                output = self.neuralNetwork.forwardPass(inputs)
                change_w = self.neuralNetwork.backwardPass(targets, output)
                self.neuralNetwork.updateWeights(change_w, trainingRate)
            accuracy = self.neuralNetwork.computeAccuracy(testImages, testLabels)
            print("{:6.2f}s: Epoch {}, Accuracy {:2.2f}%".format(time.time() - startTime, str(epoch).zfill(len(str(epochs))), accuracy * 100))