import numpy as np

class NoiseGenerator(object):
    """Generates noise"""
    def __init__(self, batch_size, hidden_size):
        self.batch_size = batch_size,
        self.hidden_size = hidden_size
        self.type = type


    def generate():
        if type == "Gaussian":
            return np.random.normal(0, 1, batch_size, hidden_size)
        else:
            raise ValueError("Unsupported noise type")
