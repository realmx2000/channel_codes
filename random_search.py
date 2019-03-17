import random
import subprocess

if __name__ == '__main__':
    
    number_samples = 10
    
    # space
    lr_space = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    sigma_space = [0.1, 0.3, 0.5, 0.7]
    encoder_size_space = [25, 50, 100, 200, 400]
    decoder_size_space = [25, 50, 100, 200, 400]
    enc_layers_space = [1,2]
    dec_layers_space = [1,2]
    lr_scheduler_space = ['plateau', 'step']
    train_ratio_space = [1, 3, 5, 7]
    optimizer_space = ['adam', 'nesterov', 'sgd']
    loss_space = ['mse', 'bce']
    block_len_space = [50, 100, 200]

    # fixed
    md_len = 2
    model_free = True
    channel = "BSC"
    redundancy = 4
    epsilon = .1

    for i in range(number_samples):
        print(f"Training Random Model {i}")

        # sample
        lr = random.choice(lr_space)
        sigma = random.choice(sigma_space)
        lr_scheduler = random.choice(lr_scheduler_space)
        train_ratio = random.choice(train_ratio_space)
        optimizer = random.choice(optimizer_space)
        loss = random.choice(loss_space)
        block_len = random.choice(block_len_space)
        enc_layers = random.choice(enc_layers_space)
        dec_layers = random.choice(dec_layers_space)
        encoder_size = random.choice(encoder_size_space)
        decoder_size = random.choice(decoder_size_space)
        while enc_layers > dec_layers:
            dec_layers = random.choice(dec_layers_space)
        while encoder_size > decoder_size:
            decoder_size = random.choice(decoder_size_space)

        name = f"random_model_{i}"

        subprocess.call(["python", "train.py",
                         "--name", name,
                         "--md_len", str(md_len),
                         "--modelfree",
                         "--channel", channel,
                         "--redundancy", str(redundancy),
                         "epsilon", str(epsilon)])
