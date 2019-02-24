import argparse
import json
import pathlib
import tensorflow as tf
from dataset import InputDataloader
from args import TrainArgParser
from models import AutoEncoder


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


def train(args):
    """Train the model on the dataset."""
    write_args(args)

    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args

    num_examples = data_args.num_epochs * data_args.batches_per_epoch * data_args.batch_size
    loader = InputDataloader(data_args.batch_size, data_args.block_length, num_examples, True)

    it = loader.get_loader()
    next = it.get_next()
    sess = tf.Session()
    encoder = Encoder(100, 25, 2, 1/2, False)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(100):
        val = sess.run(encoder.forward(next))
        print(val)
        input()

if __name__ == '__main__':
    parser = TrainArgParser()
    train(parser.parse_args())

"""
SNR = 1/5

channel = AWGN(SNR)

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, args.block_length])
channel_in = encoder(x)
channel = channel.apply_input_power_constraint(channel_in)
channel_out = channel.apply_noise(channel)
y = decoder(channel_out)
loss = tf.nn.sigmoid_cross_entropy_with_logits(x, y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # train decoder
    for i, message in enumerate(loader):
        feed_dict = {x: message}
        loss_np = sess.run(loss, feed_dict=feed_dict)
"""
