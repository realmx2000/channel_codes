import numpy as np

class InputDataloader(object):
    """Loads inputs"""
    def __init__(self, batch_size, block_length, num_examples):
        self.batch_size = batch_size
        self.block_length = block_length
        self.num_examples = num_examples

    def example_generator(self):
        for _ in range(self.num_examples):
            example = np.random.randint(0, 2, size=[self.batch_size, self.block_length, 1])
            yield example
