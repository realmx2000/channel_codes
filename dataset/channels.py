import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Layer
# from tensorflow.keras.layers.normalization import BatchNormalization 

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
