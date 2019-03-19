import random
import subprocess

if __name__ == '__main__':
    
    number_samples = 10
    
    # space
    lr_space = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    encoder_size_space = [25, 50, 100, 200, 400]
    decoder_size_space = [25, 50, 100, 200, 400]
    layers_space = [1,2]
    lr_scheduler_space = ['plateau', 'step']
    train_ratio_space = [1, 3, 5, 7]
    optimizer_space = ['adam', 'nesterov', 'sgd']
    loss_space = ['mse', 'bce']
    block_len_space = [50, 100, 200]
    batch_size_space = [500, 1000, 2000]

    # fixed
    md_len = 7
    model_free = True
    channel = "BSC"
    redundancy = 3
    epsilon = .1

    for i in range(number_samples):
        print(f"Training Random Model {i}")

        # sample
        batch_size = random.choice(batch_size_space)
        lr = random.choice(lr_space)
        sigma = 0.6 * np.random_sample() + 0.1
        lr_scheduler = random.choice(lr_scheduler_space)
        train_ratio = random.choice(train_ratio_space)
        optimizer = random.choice(optimizer_space)
        loss = random.choice(loss_space)
        block_len = random.choice(block_len_space)
        layers = random.choice(layers_space)
        encoder_size = 375 * np.random_sample() + 25
        decoder_size = (400 - encoder_size) * np.random_sample()) + encoder_size

        name = f"random_model_{i}"

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
                         "--dec_size", str(decoder_size)]
        print(command)
        subprocess.call(command)
