import tensorflow.keras as keras
from .optim_util import get_optimizer

class AutoEncoder:
    def __init__(self, model_args, data_args, channel):
        opt1= get_optimizer(model_args.optimizer, model_args.lr, model_args.scheduler, model_args.decay,
                            model_args.momentum)
        opt2= get_optimizer(model_args.optimizer, model_args.lr, model_args.scheduler, model_args.decay,
                            model_args.momentum)

        self.compile_models(model_args.loss, opt1, opt2, data_args.block_length, model_args.enc_size,
                            model_args.dec_size, model_args.num_layers, data_args.rate, model_args.gpu, channel)


    def get_encoder(self, block_length, enc_size, num_layers, rate, gpu):
        inp = keras.layers.Input((block_length, 1))
        if gpu:
            out = keras.layers.Bidirectional(keras.layers.CuDNNGRU(enc_size, return_sequences=True))(inp)
            for _ in range(num_layers - 1):
                out = (keras.layers.Bidirectional(keras.layers.CuDNNGRU(enc_size, return_sequences=True)))(out)
        else:
            out = keras.layers.Bidirectional(keras.layers.GRU(enc_size, return_sequences=True))(inp)
            for _ in range(num_layers - 1):
                out = keras.layers.Bidirectional(keras.layers.GRU(enc_size, return_sequences=True))(out)

        encodings = keras.layers.TimeDistributed(keras.layers.Dense(int(1 / rate)))(out)
        model = keras.Model(inp, encodings)
        return model


    def get_decoder(self, block_length, dec_size, num_layers, rate, gpu):
        encodings = keras.layers.Input((block_length, int(1 / rate)))
        if gpu:
            out = keras.layers.Bidirectional(keras.layers.CuDNNGRU(dec_size, return_sequences=True))(encodings)
            for _ in range(num_layers - 1):
                out = (keras.layers.Bidirectional(keras.layers.CuDNNGRU(dec_size, return_sequences=True)))(out)
        else:
            out = keras.layers.Bidirectional(keras.layers.GRU(dec_size, return_sequences=True))(encodings)
            for _ in range(num_layers - 1):
                out = keras.layers.Bidirectional(keras.layers.GRU(dec_size, return_sequences=True))(out)

        decodings = keras.layers.TimeDistributed(keras.layers.Dense(1))(out)
        model = keras.Model(encodings, decodings)
        return model


    def compile_models(self, loss, opt1,  opt2, block_length, enc_size, dec_size, num_layers,
                       rate, gpu, channel):
        encoder = self.get_encoder(block_length, enc_size, num_layers, rate, gpu)
        decoder = self.get_decoder(block_length, dec_size, num_layers, rate, gpu)

        inp = keras.layers.Input((block_length, 1))
        encodings = encoder(inp)
        #constrained = channel.apply_input_power_constraint(encodings)
        #noisy = channel.apply_noise(constrained) # This might be a problem - not a Keras layer
        decodings = decoder(encodings)

        encoder.trainable = False
        decoder.trainable = True
        self.trainable_decoder = keras.Model(inp, decodings)
        self.trainable_decoder.compile(optimizer=opt1, loss=loss, metrics=['accuracy'])

        encoder.trainable = True
        decoder.trainable = False
        self.trainable_encoder = keras.Model(inp, decodings)
        self.trainable_encoder.compile(optimizer=opt2, loss=loss, metrics=['accuracy'])


    def train_encoder(self, x):
        metrics = self.trainable_encoder.train_on_batch(x, x)
        return metrics


    def train_decoder(self, x):
        metrics = self.trainable_decoder.train_on_batch(x, x)
        return metrics
