import numpy as np
import tensorflow as tf

class BaseChannel(object):
    def __init__(self):
        self.name = "channel_"

    def apply_noise(self, x):
        pass

    def grad(self, op, grad):
        pass

class AWGN(BaseChannel):
    """Generates noise"""
    def __init__(self, SNR, max_input_power):
        super().__init__()
        self.P = max_input_power
        self.SNR = SNR
        self.name += "AWGN"

    def apply_input_power_constraint(self, x):
        mean, variance = tf.nn.moments(x)
        return (x - mean) / tf.sqrt(variance)

    def apply_noise(self, x):
        # Find way to compute variance
        std = np.sqrt(1 / self.SNR)
        noisy = x + tf.random.normal(x.shape, stddev=std)
        return noisy

    def grad(self, op, grad):
        return grad
