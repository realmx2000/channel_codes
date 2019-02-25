import tensorflow.keras as keras
from .optim_util import get_optimizer, get_scheduler

class AutoEncoder:
    def __init__(self, args, channel):
        opt1= get_optimizer(args.optimizer, args.lr, args.scheduler, args.decay,
                                         args.momentum)
        opt2= get_optimizer(args.optimizer, args.lr, args.scheduler, args.decay,
                                         args.momentum)
        self.scheduler = get_scheduler(args.scheduler, args.decay, args.patience)

        self.compile_models(args.loss, opt1, opt2, args.block_length, args.num_units,
                            args.num_layers, args.rate, args.gpu, channel)


    def get_encoder(self, block_length, num_units, num_layers, rate, gpu):
        inp = keras.layers.Input((block_length, 1))
        if gpu:
            out = keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True))(inp)
            for _ in range(num_layers - 1):
                out = (keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True)))(out)
        else:
            out = keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True))(inp)
            for _ in range(num_layers - 1):
                out = keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True))(out)

        encodings = keras.layers.TimeDistributed(keras.layers.Dense(int(1 / rate)))(out)
        model = keras.Model(inp, encodings)
        return model


    def get_decoder(self, block_length, num_units, num_layers, rate, gpu):
        encodings = keras.layers.Input((block_length, int(1 / rate)))
        if gpu:
            out = keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True))(encodings)
            for _ in range(num_layers - 1):
                out = (keras.layers.Bidirectional(keras.layers.CuDNNGRU(num_units, return_sequences=True)))(out)
        else:
            out = keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True))(encodings)
            for _ in range(num_layers - 1):
                out = keras.layers.Bidirectional(keras.layers.GRU(num_units, return_sequences=True))(out)

        decodings = keras.layers.TimeDistributed(keras.layers.Dense(1))(out)
        model = keras.Model(encodings, decodings)
        return model


    def compile_models(self, loss, opt1,  opt2, block_length, num_units, num_layers,
                       rate, gpu, channel):
        encoder = self.get_encoder(block_length, num_units, num_layers, rate, gpu)
        decoder = self.get_decoder(block_length, num_units, num_layers, rate, gpu)

        inp = keras.layers.Input((block_length, 1))
        encodings = encoder(inp)
        constrained = channel.apply_input_power_constraint(encodings)
        noisy = channel.apply_noise(constrained) # This might be a problem - not a Keras layer
        decodings = decoder(noisy)

        encoder.trainable = False
        decoder.trainable = True
        self.trainable_decoder = keras.Model(inp, decodings)
        self.trainable_decoder.compile(optimizer=opt1, loss=loss, metrics=['accuracy', loss])

        encoder.trainable = True
        decoder.trainable = False
        self.trainable_encoder = keras.Model(inp, decodings)
        self.trainable_encoder.compile(optimizer=opt2, loss=loss, metrics=['accuracy', loss])


    def train_encoder(self, x):
        metrics = self.trainable_encoder.train_on_batch(x, x)
        return metrics

    def train_decoder(self, x):
        metrics = self.trainable_decoder.train_on_batch(x, x)
        return metrics
