import argparse

class ArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser("Run channel codes project.")
        self.parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
        self.parser.add_argument("--block_length", default=100, type=int, help="Block length for inputs.")
        self.parser.add_argument("--SNR", required=True, type=float, help="Channel SNR.")
        self.parser.add_argument("--lr", default=0.001, type=float, help="Learning Rate.")
        self.parser.add_argument("--batches_per_epoch", default=100, type=int, help="Number of batches in an epoch.")
        self.parser.add_argument("--num_epochs", default=240, type=int, help="Number of epochs.")
        self.parser.add_argument("--loss", choices=["mse", "bce"], type=str, default="bce",
                                 help="Loss to use. Either 'bce' for binary cross entropy or 'mse' for mean squared error.")
        self.parser.add_argument("--optimizer", type=str, choices=["adam", "nesterov", "sgd"], default="adam",
                                 help="Optimizer to use. One of 'adam', 'nesterov', or 'sgd'.")
        self.parser.add_argument("--md_reg", type=float, default=0.001, help="Weight of minimum distance regularizer.")
        self.parser.add_argument("--train_ratio", type=int, default=5,
                                 help="Number of times to train transmitter per receiver train.")
        self.parser.add_argument("--enc_size", type=int, default=25, help="Number of units in encoder.")
        self.parser.add_argument("--enc_layers", type=int, default=2, help="Number of layers in encoder.")
        self.parser.add_argument("--dec_size", type=int, default=100, help="Number of units in decoder.")
        self.parser.add_argument("--dec_layers", type=int, default=2, help="Number of layers in decoder.")
        self.parser.add_argument("--lr_scheduler", type=str, choices=["plateau", "step"], default='plateau',
                                 help="Learning rate scheduler. 'plateau' for step on loss plateau, 'step' for fixed steps.")
        self.parser.add_argument("--decay_step", type=float, default=0.1, help="Learning rate decay step.")

    def parse_args(self):
        return self.parser.parse_args()