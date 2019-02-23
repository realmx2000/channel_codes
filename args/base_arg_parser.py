import argparse


class BaseArgParser(object):


    def __init__(self):
        self.parser = argparse.ArgumentParser("Run channel codes project to learn convolutional code.")

        # directorys
        self.parser.add_argument("--name", dest='logger_args.name',
                                 type=str, default='debugging')
        self.parser.add_argument("--save_dir", dest='logger_args.save_dir',
                                 type=str, default='experiments/')

        # data args
        self.parser.add_argument("--SNR", dest='data_args.SNR',
                                 default=0, type=float, help="Channel SNR.")
        self.parser.add_argument("--rate", dest='data_args.rate',
                                 default=0.5, type=float, help="Communication rate")
        self.parser.add_argument("--batch_size", dest='data_args.batch_size',
                                 default=1000, type=int, help="Batch size.")
        self.parser.add_argument("--block_length", dest='data_args.block_length',
                                 default=100, type=int, help="Block length for inputs.")

        # gpu
        self.parser.add_argument("--gpu", action='store_true')

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
        return args