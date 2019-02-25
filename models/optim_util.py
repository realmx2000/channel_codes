import keras

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
