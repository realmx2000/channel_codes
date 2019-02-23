import tensorflow as tf
import keras

### THIS FILE NO LONGER USED ###

class Decoder:
    def __init__(self, block_length, num_units, num_layers, rate, gpu):
        self.block_length = block_length
        self.num_units = num_units
        self.rate = rate
        self.model = keras.Sequential()

        if gpu:
            self.model.add(keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True),
                                                                      input_shape=(block_length, int(1 / rate))))
            self.model.add(keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True)))
        else:
            self.model.add(keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True),
                                                                     input_shape=(block_length, int(1 / rate))))
            self.model.add(keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True)))

        self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


    def forward(self, x):
        return self.model(x)