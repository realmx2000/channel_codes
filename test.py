from args import TestArgParser
from dataset import InputDataloader, PowerConstraint
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
    model = AutoEncoder(model_args, data_args, power_constraint, possible_inputs)

    saver.load(model)

if __name__ == '__main__':
    parser = TestArgParser()
    test(parser.parse_args())
