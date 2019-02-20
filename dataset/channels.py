import numpy as np
import tensorflow as tf

class BaseChannel(object):
    def __init__(self):
        self.name = "channel_"

    def apply_noise(self, x):
        pass

    def grad(self, op, grad):
        pass

    def apply_tensorflow(self, x):
        # Need to generate a unique name to avoid duplicates
        rnd_name = self.name + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(self.grad)
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            #TODO: x and float32 may need to be in lists. What does "stateful" control?
            return tf.py_func(self.apply_noise, x, tf.float32, stateful=True, name=self.name)

class AWGN(BaseChannel):
    """Generates noise"""
    def __init__(self, SNR):
        super().__init__()
        self.SNR = SNR
        self.name += "AWGN"

    def apply_noise(self, x):
        #Find way to compute variance
        var = 0.1
        noisy = x + np.random.standard_normal(x.shape)
        return noisy

    def grad(self, op, grad):
        return grad
