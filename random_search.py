import random
import subprocess
import numpy as np
import string

if __name__ == '__main__':
    
    number_samples = 10
    
    # space
    layers_space = [1,2]
    lr_scheduler_space = ['plateau', 'step']
    train_ratio_space = [1, 3, 5, 7]
    optimizer_space = ['adam', 'nesterov', 'sgd']
    loss_space = ['mse', 'bce']

    # fixed
    md_len = 7
    model_free = True
    channel = "BSC"
    redundancy = 3
    epsilon = .1

    for i in range(number_samples):
        print(f"Training Random Model {i}")

        # sample
        batch_size = round(2 ** (3 * np.random.random_sample() + 8))
        lr = np.exp(4 * np.random.random_sample() - 5)
        sigma = 0.45 * np.random.random_sample() + 0.1
        lr_scheduler = random.choice(lr_scheduler_space)
        train_ratio = random.choice(train_ratio_space)
        optimizer = random.choice(optimizer_space)
        loss = random.choice(loss_space)
        block_len = round(2 ** (3 * np.random.random_sample() + 5))
        layers = random.choice(layers_space)
        decay_step = 0.45 * np.random.random_sample() + .05
        encoder_size = round(375 * np.random.random_sample() + 25)
        decoder_size = round((400 - encoder_size) * np.random.random_sample() + encoder_size)

        name = "random_model_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

        command = ["python", "train.py",
                         "--name", name,
                         "--md_len", str(md_len),
                         "--modelfree",
                         "--channel", channel,
                         "--redundancy", str(redundancy),
                         "--epsilon", str(epsilon),
                         "--batch_size", str(batch_size),
                         "--lr", str(lr),
                         "--sigma", str(sigma),
                         "--lr_scheduler", lr_scheduler,
                         "--train_ratio", str(train_ratio),
                         "--optimizer", str(optimizer),
                         "--loss", loss,
                         "--block_len", str(block_len),
                         "--layers", str(layers),
                         "--enc_size", str(encoder_size),
                         "--dec_size", str(decoder_size),
                         "--decay_step", str(decay_step)]
        print(command)
        subprocess.call(command)
