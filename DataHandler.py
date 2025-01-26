import numpy as np
import struct
from array import array

# Source: https://www.kaggle.com/code/hojjatk/read-mnist-dataset

class DataHandler:
    def read_images_labels(images_filepath: str, labels_filepath: str) -> tuple[list[array], list[array]]:
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img
        
        return (images, labels)
    
    def load_data(training_images_filepath: str, training_labels_filepath: str, test_images_filepath: str, test_labels_filepath: str) -> tuple[tuple[array, array], tuple[array, array]]:
        x_train, y_train = DataHandler.read_images_labels(training_images_filepath, training_labels_filepath)
        x_test, y_test = DataHandler.read_images_labels(test_images_filepath, test_labels_filepath)
        return ((x_train, y_train), (x_test, y_test))