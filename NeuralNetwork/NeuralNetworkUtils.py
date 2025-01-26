import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x: np.ndarray) -> np.ndarray:
    return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)

def softMax(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

def softMaxDerivative(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))