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
                         "--epsilon", str(epsilon),
                         "--batch_size", str(batch_size),
                         "--lr", str(lr),
                         "--sigma", str(sigma),
                         "--lr_scheduler", lr_scheduler,
                         "--train_ratio", str(train_ratio),
                         "--optimizer", str(optimizer),
                         "--loss", loss,
                         "--block_len", str(block_len),
                         "--enc_layers", str(enc_layers),
                         "--dec_layers", str(dec_layers),
                         "--encoder_size", str(encoder_size),
                         "--decoder_size", str(decoder_size)])
