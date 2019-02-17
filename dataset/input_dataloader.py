import numpy as np

class InputDataloader(object):
    """Loads inputs"""
    def __init__(self, batch_size, bit_length, iters, is_training):
        self.batch_size = batch_size
        self.iters = iters
        self.is_training = is_training


    def __len__(self):
        return self.iters


    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter >= len(self):
            raise StopItertion
        else:
            self.counter += 1
            batch = np.random.randint(2, size=(batch_size, bit_length))
            return batch

