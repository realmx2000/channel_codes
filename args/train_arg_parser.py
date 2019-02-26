from .base_arg_parser import BaseArgParser


class TrainArgParser(BaseArgParser):


    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True

        # epoch args
        self.parser.add_argument("--batches_per_epoch", dest='data_args.batches_per_epoch',
                                 default=100, type=int, help="Number of batches in an epoch.")
        self.parser.add_argument("--num_epochs", dest='data_args.num_epochs',
                                 default=240, type=int, help="Number of epochs.")

        # logger args
        self.parser.add_argument("--iters_per_print", dest='logger_args.iters_per_print',
                                 default=10, type=int, help="Number of iters till print")

        # loss args
        self.parser.add_argument("--loss", choices=["mse", "bce"], dest='model_args.loss',
                                 type=str, default="bce",
                                 help="Loss to use. Either 'bce' for binary cross entropy or 'mse' for mean squared error.")

        # learning args
        self.parser.add_argument("--lr", dest='model_args.lr',
                                 default=0.001, type=float, help="Learning Rate.")
        self.parser.add_argument("--optimizer", dest='model_args.optimizer',
                                 type=str, choices=["adam", "nesterov", "sgd"], default="adam",
                                 help="Optimizer to use. One of 'adam', 'nesterov', or 'sgd'.")
        self.parser.add_argument("--md_reg", dest='model_args.md_reg',
                                 type=float, default=0.001, help="Weight of minimum distance regularizer.")
        self.parser.add_argument("--train_ratio", dest='model_args.train_ratio',
                                 type=int, default=5,
                                 help="Number of times to train transmitter per receiver train.")
        self.parser.add_argument("--lr_scheduler", dest='model_args.scheduler',
                                 type=str, choices=["plateau", "step"], default='plateau',
                                 help="Learning rate scheduler. 'plateau' for step on loss plateau, 'step' for fixed steps.")
        self.parser.add_argument("--decay_step", dest='model_args.decay',
                                 type=float, default=0.1, help="Learning rate decay step.")
        self.parser.add_argument("--momentum", default=0.1, dest='model_args.momentum', type=float,
                                 help="Momentum to use. Nesterov optimizer only.")
        self.parser.add_argument("--patience", dest='model_args.patience',
                                 type=int, default=10,
                                 help="Number of iterations to decay learning rate after if plateau.")

        # architecture args
        self.parser.add_argument("--enc_size", dest='model_args.enc_size',
                                 type=int, default=25, help="Number of units in encoder.")
        self.parser.add_argument("--layers", dest='model_args.num_layers',
                                 type=int, default=2, help="Number of layers.")
        self.parser.add_argument("--dec_size", dest='model_args.dec_size',
                                 type=int, default=100, help="Number of units in decoder.")

    def parse_args(self):
        args = super().parse_args()
        if args.model_args.loss == 'mse':
            args.model_args.loss = 'mean_squared_error'
        elif args.model_args.loss == 'bce':
            args.model_args.loss = 'binary_crossentropy'

        return args
