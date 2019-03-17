from args import TestArgParser
from dataset import InputDataloader, PowerConstraint, get_channel
from models import AutoEncoder, get_scheduler
from models.optim_util import get_possible_inputs
from saver import ModelSaver

from pathlib import Path
import argparse
import json
import numpy as np


def load_args(path):
    args_dir = path / "args.json"
    with open(args_dir) as f:
        args = json.load(f)
    model_args = argparse.Namespace()
    data_args = argparse.Namespace()
    for k, v in args['model_args'].items():
        setattr(model_args, k, v)
    for k, v in args['data_args'].items():
        setattr(data_args, k, v)
    return model_args, data_args 


def get_md_set(md_len):
    possible_inputs = get_possible_inputs(md_len)
    possible_inputs = np.expand_dims(np.stack(possible_inputs, axis=0), axis=2)
    return possible_inputs


def test(args):
    print(args.logger_args.save_dir)

    model_args, data_args = load_args(args.logger_args.save_dir)

    saver = ModelSaver(Path(args.logger_args.save_dir), None)

    power_constraint = PowerConstraint()
    possible_inputs = get_md_set(model_args.md_len)

    data_args.batch_size = 1000
    data_args.batches_per_epoch = 1000
    dataset_size = data_args.batch_size * data_args.batches_per_epoch
    loader = InputDataloader(data_args.batch_size, data_args.block_length, dataset_size)
    loader = loader.example_generator()

    SNRs = [-1, 0, 1, 2, 3, 4]

    for SNR in SNRs:
        data_args.SNR = SNR
        print(data_args.channel)
        print(model_args.modelfree)
        channel = get_channel(data_args.channel, model_args.modelfree, data_args)
        model = AutoEncoder(model_args, data_args, power_constraint, channel, possible_inputs)
        saver.load(model)
        for step in range(data_args.batches_per_epoch):
            msg = next(loader)
            metrics = model.trainable_encoder.test_on_batch(msg, msg)
            print(metrics)

    saver.load(model)

if __name__ == '__main__':
    parser = TestArgParser()
    test(parser.parse_args())
