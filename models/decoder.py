import tensorflow as tf
import keras

class Decoder:
    def __init__(self, num_units, num_layers):
        self.base_fw = [keras.layers.GRUCell(num_units, reset_after=True) for _ in range(num_layers)]
        self.base_bw = [keras.layers.GRUCell(num_units, reset_after=True) for _ in range(num_layers)]
        self.summarizer = keras.layers.Dense(1, activation='sigmoid')

    def forward(self, x):
        (all_states, fw_state, bw_state) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.base_fw, self.base_bw, x)
        out = tf.squeeze(all_states)
        decoded = self.summarizer(out)