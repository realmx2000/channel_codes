import argparse
import json
import pathlib
import tensorflow as tf
from dataset import InputDataloader, AWGN
from args import TrainArgParser
from models import AutoEncoder, get_scheduler


def write_args(args):
    save_dir = args.logger_args.save_dir
    """Save args to a JSON file."""
    with (save_dir / 'args.json').open('w') as fh:
        args_dict = vars(args)
        for k, v in args_dict.items():
            if type(v) is argparse.Namespace:
                args_dict[k] = vars(v)
                for k2, v2 in args_dict[k].items():
                    if type(v2) is pathlib.PosixPath:
                        args_dict[k][k2] = v2.as_posix()
        json.dump(args_dict, fh, indent=4, sort_keys=True)
        fh.write('\n')

"""
def train(args):
    write_args(args)

    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args

    num_examples = data_args.num_epochs * data_args.batches_per_epoch * data_args.batch_size
    loader = InputDataloader(data_args.batch_size, data_args.block_length, num_examples, True)

    generator = loader.example_generator()
    sess = tf.Session()
    encoder = Encoder(100, 25, 2, 1/2, False)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(100):
        val = sess.run(encoder.forward(next))
        print(val)
        input()
"""

def train(args):
    SNR = 1/5

    channel = AWGN(SNR, 1)
    model = AutoEncoder(args, channel)
    scheduler = get_scheduler(args.scheduler, args.decay, args.patience)
    loader = InputDataloader(args.batch_size, args.block_length, args.num_examples)
    loader = loader.example_generator()

    logs = {}
    logs['loss'] = []
    logs['accuracy'] = []

    scheduler.on_train_begin()
    while True: # Loop until StopIteration
        try:
            metrics = None
            for step in range(args.batches_per_epoch % (args.train_ratio + 1)):
                msg = next(loader)
                metrics = model.train_encoder(msg)
                for _ in range(args.train_ratio):
                    msg = next(loader)
                    metrics = model.train_decoder(msg)

            logs['loss'].append(metrics[0])
            logs['accuracy'].append(metrics[1])
            scheduler.on_epoch_end(logs=logs)
        except:
            break

if __name__ == '__main__':
    parser = TrainArgParser()
    train(parser.parse_args())


