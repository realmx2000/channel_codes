import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Layer
# from tensorflow.keras.layers.normalization import BatchNormalization

def get_channel(name, modelfree, data_args):
    if not modelfree:
        if name == "AWGN":
            return get_AWGN(data_args.SNR)
        elif name == "RBF":
            return RBF(data_args.batch_size, data_args.block_length, data_args.scale)
        else:
            raise Exception("Invalid channel specified.")
    else:
        if name == "AWGN":
            return AWGN_modelfree(data_args.batch_size, data_args.block_length, data_args.SNR)
        if name == 'RBF':
            return RBF(data_args.batch_size, data_args.block_length, data_args.scale)
        elif name == "BSC":
            return BSC(data_args.batch_size, data_args.block_length, data_args.epsilon)
        elif name == "BEC":
            return BEC(data_args.batch_size, data_args.block_length, data_args.epsilon)
        else:
            raise Exception("Invalid channel specified.")

def get_AWGN(snr):
    std = np.sqrt(1. / snr)
    return K.layers.GaussianNoise(std)

class Channel(Layer):
    def __init__(self, batch_size, block_length):
        super().__init__()
        self.batch_size = batch_size
        self.block_length = block_length

    def build(self, input_shape):
        self.shape = input_shape.as_list()
        self.shape[0] = self.batch_size
        self.shape[1] = self.block_length

    def identity_grad(self, op, grad):
        return grad

    def apply_channel(self, inp):
        pass

    def call(self, inp):
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(self.identity_grad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(self.apply_channel, [inp], [tf.float32], stateful=True, name="Channel")

class RBF(Channel):
    def __init__(self, batch_size, block_length, sigma):
        super().__init__(batch_size, block_length)
        self.sigma = sigma

    def apply_channel(self, inp):
        var = np.random.rayleigh(self.sigma)
        noise = np.sqrt(var) * np.random.standard_normal(self.shape)
        return (inp + noise).astype(np.float32)

class AWGN_modelfree(Channel):
    def __init__(self, batch_size, block_length, snr):
        super().__init__(batch_size, block_length)
        self.std = np.sqrt(1. / snr)

    def apply_channel(self, inp):
        noise = self.std * np.random.standard_normal(self.shape)
        return (inp + noise).astype(np.float32)

class Discrete_noiseless(Channel):
    def __init__(self, batch_size, block_length, eps):
        super().__init__(batch_size, block_length)
        self.eps = eps

    def discretize(self, inp):
        return np.round(np.clip(inp, 0, 1)).astype(np.float32)

    def apply_channel(self, inp):
        return self.discretize(inp)

class BSC(Discrete_noiseless):
    def apply_channel(self, inp):
        inp_d = self.discretize(inp)
        noise = np.random.random_sample(self.shape)
        noise = (noise < self.eps)
        return np.mod(inp_d + noise, 2)

class BEC(Discrete_noiseless):
    def apply_channel(self, inp):
        inp_d = self.discretize(inp)
        noise = np.random.random_sample(self.shape)
        noise = (noise < self.eps)
        # -1 indicates erasure.
        inp_d[noise] = -1
        return inp_d

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
