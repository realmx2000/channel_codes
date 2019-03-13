import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np

def get_optimizer(optimizer, lr, scheduler, decay, momentum=0):
    lr_decay = decay if scheduler == 'step' else 0.0
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(lr=lr, decay=lr_decay)
    elif optimizer == 'nesterov':
        opt = keras.optimizers.SGD(lr=lr, momentum=momentum, decay=lr_decay, nesterov=True)
    else:
        opt = keras.optimizers.SGD(lr=lr, decay=lr_decay, nesterov=False)

    return opt

def get_scheduler(scheduler, decay, patience):
    if scheduler == 'plateau':
        return keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=decay, patience=patience, cooldown=1)
    else:
        print("Unsupported scheduler used.")

def get_possible_inputs(L):
    if L == 0:
        return [np.zeros((0), dtype=float)]

    inputs = get_possible_inputs(L - 1)
    new_inputs = []
    for vec in inputs:
        new_inputs.append(np.append(vec, 0))
        new_inputs.append(np.append(vec, 1))
    return new_inputs

def loss_wrapper(loss, model, inputs, lamb, batch_size):
    def loss_fn(y_true, y_pred):
        loss_val = None
        if loss == 'binary_crossentropy':
            loss_val = keras.losses.binary_crossentropy(y_true, y_pred)
        elif loss == 'mean_squared_error':
            loss_val = keras.losses.mean_squared_error(y_true, y_pred)
        loss_val += lamb * md_regularization(model, inputs, batch_size)
        return loss_val
    return loss_fn

def md_regularization(model, inputs, batch_size):
    encodings = model.predict(inputs, batch_size=batch_size)
    min_dist = 0
    init = False
    for i in range(encodings.shape[0]):
        for j in range(i+1, encodings.shape[0]):
            out1 = encodings[i,:]
            out2 = encodings[j,:]
            dist = K.sum(K.square(out1 - out2))
            if not init :
                min_dist = dist
                init = True
            else:
                min_dist = K.minimum(min_dist, dist)
    return min_dist

def pi(sigma, inp):
    def add_noise(inp1):
        noise = sigma * np.random.standard_normal(inp1.shape.as_list())
        return inp1 + noise
    def log_gradient(op, grad):
        inp1 = op.inputs[0]
        out = op.outputs[0]
        cov = 1 / (sigma * sigma) * np.ones(inp1.shape.as_list())
        covinv = tf.convert_to_tensor(np.diag(cov))
        return grad * tf.matmul(covinv, out - inp1) # Grad should be a constant (the loss)

    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(log_gradient)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(add_noise, [inp], [tf.float32], stateful=True, name="Noise")

class Pi(Layer):
    def __init__(self, batch_size, sigma):
        super().__init__()
        self.std = sigma
        self.batch_size = batch_size

    def build(self, input_shape):
        self.shape = input_shape.as_list()
        self.shape[0] = self.batch_size

    def log_gradient(self, op, grad):
        inp1 = op.inputs[0]
        out = op.outputs[0]
        cov = 1 / (self.std * self.std) * np.ones(self.shape)
        covinv = tf.convert_to_tensor(cov, dtype=tf.float32)
        return tf.multiply(grad, tf.multiply(covinv, out - inp1)) # Grad should be a constant (the loss)

    def identity_grad(self, op, grad):
        return grad

    def add_noise(self, inp):
        noise = self.std * np.random.standard_normal(self.shape)
        return (inp + noise).astype(np.float32)

    def call(self, inp):
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(self.identity_grad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(self.add_noise, [inp], [tf.float32], stateful=True, name="Noise")
