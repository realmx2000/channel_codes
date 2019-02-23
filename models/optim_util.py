import keras

def get_optimizer(optimizer, lr, scheduler, decay, momentum=0, patience=10):
    lr_decay = decay if scheduler == 'step' else 0.0
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(lr=lr, decay=lr_decay)
    elif optimizer == 'nesterov':
        opt = keras.optimizers.SGD(lr=lr, momentum=momentum, decay=lr_decay, nesterov=True)
    else:
        opt = keras.optimizers.SGD(lr=lr, decay=lr_decay, nesterov=False)

    return opt

def bitwise_error(pred, true):
    # TODO: implement error

def blockwise_error(pred, true):
    # TODO: implement error