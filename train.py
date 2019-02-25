import argparse
import copy
import json
import pathlib
import tensorflow as tf
from dataset import InputDataloader, AWGN
from args import TrainArgParser
from models import AutoEncoder, get_scheduler


def write_args(args):
    save_dir = args.logger_args.save_dir
    copy_args = copy.deepcopy(args)
    """Save args to a JSON file."""
    with (save_dir / 'args.json').open('w') as fh:
        args_dict = vars(copy_args)
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

    write_args(args)

    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args

    SNR = 1/5

    channel = AWGN(SNR, 1)
    model = AutoEncoder(model_args, data_args, channel)
    enc_scheduler = get_scheduler(model_args.scheduler, model_args.decay, model_args.patience)
    dec_scheduler = get_scheduler(model_args.scheduler, model_args.decay, model_args.patience)
    enc_scheduler.set_model(model.trainable_encoder)
    dec_scheduler.set_model(model.trainable_decoder)

    dataset_size = data_args.batch_size * data_args.batches_per_epoch * data_args.num_epochs
    loader = InputDataloader(data_args.batch_size, data_args.block_length, dataset_size)
    loader = loader.example_generator()

    logs = {}
    logs['loss'] = []
    logs['accuracy'] = []
    curr_log = {}

    enc_scheduler.on_train_begin()
    dec_scheduler.on_train_begin()
    epoch = 0
    while True: # Loop until StopIteration
        try:
            metrics = None
            for step in range(data_args.batches_per_epoch % (model_args.train_ratio + 1)):
                msg = next(loader)
                metrics = model.train_encoder(msg)
                for _ in range(model_args.train_ratio):
                    msg = next(loader)
                    metrics = model.train_decoder(msg)
            print(metrics)
            logs['loss'].append(metrics[0])
            logs['accuracy'].append(metrics[1])
            curr_log['loss'] = metrics[0]
            curr_log['accuracy'] = metrics[1]

            # Should these be joint or separate?
            enc_scheduler.on_epoch_end(epoch, logs=curr_log)
            dec_scheduler.on_epoch_end(epoch, logs=curr_log)
            epoch += 1
        except StopIteration:
            break

if __name__ == '__main__':
    parser = TrainArgParser()
    train(parser.parse_args())
