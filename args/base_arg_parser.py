import argparse
from pathlib import Path


class BaseArgParser(object):


    def __init__(self):
        self.parser = argparse.ArgumentParser("Run channel codes project to learn convolutional code.")

        self.parser.add_argument("--model", dest='model_args.model',
                                 type=str, default="AE",
                                 choices=["AE", "Conv", "Turbo"])
        self.parser.add_argument("--modelfree", dest='model_args.modelfree', action='store_true',
                                 help="Use model free training.")
        self.parser.add_argument("--sigma", dest='model_args.sigma', type=float, default=0.1,
                                 help="Standard deviation of the noise to use.")
        self.parser.add_argument("--sigma_decay", dest='model_args.sigma_decay', type=float, default=1.0,
                                 help="Amount to decay sigma for each epoch.")

        # directories
        self.parser.add_argument("--name", dest='logger_args.name',
                                 type=str, default='debugging')
        self.parser.add_argument("--save_dir", dest='logger_args.save_dir',
                                 type=str, default='experiments/')

        # data args
        self.parser.add_argument("--channel", dest='data_args.channel', type=str, choices=['AWGN', 'RBF', 'BSC', 'BEC'])
        self.parser.add_argument("--SNR", dest='data_args.SNR',
                                 default=5, type=float, help="Channel SNR (for continuous channels).")
        self.parser.add_argument("--scale", dest='data_args.scale', default=0.2, type=float, help="Scale for the"
                                 "rayleigh distribution (RBF channel only).")
        self.parser.add_argument("--epsilon", dest='data_args.epsilon', type=float, default=0.05,
                                 help="Channel error probability (for discrete channels).")
        self.parser.add_argument("--redundancy", dest='data_args.redundancy',
                                 default=2, type=int, help="Inverse of the rate.")
        self.parser.add_argument("--batch_size", dest='data_args.batch_size',
                                 default=1000, type=int, help="Batch size.")
        self.parser.add_argument("--block_length", dest='data_args.block_length',
                                 default=100, type=int, help="Block length for inputs.")

        # gpu
        self.parser.add_argument("--gpu", action='store_true', dest='model_args.gpu')

        self.is_training = False


    @staticmethod
    def fix_nested_namespaces(args):
        """Make sure that nested namespaces work.
        Args:
            args: a argsparse.namespace object containing all the
            arguments e.g args.logger_args.save_dir
        Obs: Only one level of nesting is supported.
        """
        group_name_keys = []

        for key in args.__dict__:
            if '.' in key:
                group, name = key.split('.')
                group_name_keys.append((group, name, key))

        for group, name, key in group_name_keys:
            if group not in args:
                args.__dict__[group] = argparse.Namespace()

            args.__dict__[group].__dict__[name] = args.__dict__[key]
            del args.__dict__[key]


    def parse_args(self):
        args = self.parser.parse_args()
        self.fix_nested_namespaces(args)

        # convert path strings to Path
        save_dir = (Path(args.logger_args.save_dir) / 
                    args.model_args.model / 
                    args.logger_args.name)
        save_dir.mkdir(parents=True, exist_ok=True)
        args.logger_args.save_dir = save_dir
        return args
