import tensorflow.keras as keras
from .optim_util import get_optimizer, bitwise_error, blockwise_error

class AutoEncoder:
    def __init__(self, args, channel):
        opt1= get_optimizer(args.optimizer, args.lr, args.scheduler, args.decay,
                                         args.momentum, args.patience)
        opt2= get_optimizer(args.optimizer, args.lr, args.scheduler, args.decay,
                                         args.momentum, args.patience)
        self.compile_models(args.loss, opt1, opt2, args.block_lenth, args.num_units,
                            args.num_layers, args.rate, args.gpu, channel)
    def get_encoder(self, block_length, num_units, num_layers, rate, gpu):
        inp = keras.layers.Input((block_length, 1))
        if gpu:
            layer1 = keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True))(inp)
            layer2 = (keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True)))(layer1)
        else:
            layer1 = keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True))(inp)
            layer2 = keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True))(layer1)

        encodings = keras.layers.TimeDistributed(keras.layers.Dense(int(1 / rate)))(layer2)
        model = keras.Model(inp, encodings)
        return model

    def get_decoder(self, block_length, num_units, num_layers, rate, gpu):
        encodings = keras.layers.Input((block_length, int(1 / rate)))
        if gpu:
            layer1 = keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True))(encodings)
            layer2 = (keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True)))(layer1)
        else:
            layer1 = keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True))(encodings)
            layer2 = keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True))(layer1)

        decodings = keras.layers.TimeDistributed(keras.layers.Dense(1))(layer2)
        model = keras.Model(encodings, decodings)
        return model

    def compile_models(self, loss, opt1,  opt2, block_length, num_units, num_layers,
                       rate, gpu, channel):
        encoder = self.get_encoder(block_length, num_units, num_layers, rate, gpu)
        decoder = self.get_decoder(block_length, num_units, num_layers, rate, gpu)

        inp = keras.layers.Input((block_length, 1))
        encodings = encoder(inp)
        noisy = channel.apply_noise(encodings) # This might be a problem - not a Keras layer
        decodings = decoder(noisy)

        encoder.trainable = False
        decoder.trainable = True
        self.trainable_decoder = keras.Model(inp, decodings)
        self.trainable_decoder.compile(optimizer=opt1, loss=loss, metrics=[bitwise_error, blockwise_error, loss])

        encoder.trainable = True
        decoder.trainable = False
        self.trainable_encoder = keras.Model(inp, decodings)
        self.trainable_encoder.compile(optimizer=opt2, loss=loss, metrics=[bitwise_error, blockwise_error, loss])

    def train(self, iterator, train_ratio):
        # TODO: alternating training.
        pass

