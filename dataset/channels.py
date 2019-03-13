import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Layer
# from tensorflow.keras.layers.normalization import BatchNormalization

def get_AWGN(snr):
    std = np.sqrt(1. / snr)
    return K.layers.GaussianNoise(std)

class AWGN_modelfree(Layer):
    def __init__(self, batch_size, snr):
        super().__init__()
        self.std = np.sqrt(1. / snr)
        self.batch_size = batch_size

    def build(self, input_shape):
        self.shape = input_shape.as_list()
        self.shape[0] = self.batch_size

    def add_noise(self, inp):
        noise = self.std * np.random.standard_normal(self.shape)
        return (inp + noise).astype(np.float32)

    def identity_grad(self, op, grad):
        return grad

    def call(self, inp):
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(self.identity_grad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(self.add_noise, [inp], [tf.float32], stateful=True, name="Noise")

class PowerConstraint(Layer):
    def __init__(self, **kwargs):
        super(PowerConstraint, self).__init__(**kwargs)
    def build(self, input_shape):
        super(PowerConstraint, self).build(input_shape)
    def call(self, x):
        mean = K.backend.mean(x, axis=[1,2], keepdims=True)
        std = K.backend.std(x, axis=[1,2], keepdims=True)
        return (x - mean) / std
    def compute_output_shape(self, input_shape):
        return input_shape
