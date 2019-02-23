import tensorflow as tf
import keras

class Encoder:
    def __init__(self, block_length, num_units, num_layers, rate, gpu):
        self.block_length = block_length
        self.num_units = num_units
        self.rate = rate
        if gpu:
            self.layer1 = keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True))
            self.layer2 = keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True))
        else:
            self.layer1 = keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True))
            self.layer2 = keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True))

        self.summarizer = keras.layers.Dense(int(1 / rate))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        fc_in = tf.reshape(out2, [-1, self.num_units])
        fc_out = self.summarizer(fc_in)
        encoded = tf.reshape(fc_out, [-1, self.block_length, int(1 / self.rate)])
        return encoded
