import numpy as np
import tensorflow as tf

class InputDataloader(object):
    """Loads inputs"""
    def __init__(self, batch_size, block_length, num_examples, is_training):
        self.batch_size = batch_size
        self.is_training = is_training
        self.block_length = block_length
        self.num_examples = num_examples

    def example_generator(self):
        for _ in range(self.num_examples):
            example = np.random.randint(0, 2, size=self.block_length)
            yield np.expand_dims(example, 2)

    def get_loader(self):
        loader = tf.data.Dataset.batch(self.batch_size).from_generator(self.example_generator,
                                                                       tf.int64, tf.TensorShape([None]))
        return loader