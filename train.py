import argparse
import copy
import json
import pathlib
import numpy as np
import tensorflow.keras.backend as K
from dataset import InputDataloader, PowerConstraint, get_channel
from logger import TrainLogger
from args import TrainArgParser
from models import AutoEncoder, get_scheduler
from models.optim_util import get_possible_inputs
from saver import ModelSaver


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

def get_md_set(md_len):
    possible_inputs = get_possible_inputs(md_len)
    possible_inputs = np.expand_dims(np.stack(possible_inputs, axis=0), axis=2)
    return possible_inputs

def train(args):
    write_args(args)

    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args

    print(f"Training {logger_args.name}")

    power_constraint = PowerConstraint()
    possible_inputs = get_md_set(model_args.md_len)
    channel = get_channel(data_args.channel, model_args.modelfree, data_args)

    model = AutoEncoder(model_args, data_args, power_constraint, channel, possible_inputs)
    enc_scheduler = get_scheduler(model_args.scheduler, model_args.decay, model_args.patience)
    dec_scheduler = get_scheduler(model_args.scheduler, model_args.decay, model_args.patience)
    
    enc_scheduler.set_model(model.trainable_encoder)
    dec_scheduler.set_model(model.trainable_decoder)
    dataset_size = data_args.batch_size * data_args.batches_per_epoch * data_args.num_epochs
    loader = InputDataloader(data_args.batch_size, data_args.block_length, dataset_size)
    loader = loader.example_generator()
    logger = TrainLogger(logger_args.save_dir, logger_args.name,
                         data_args.num_epochs, logger_args.iters_per_print)

    saver = ModelSaver(logger_args.save_dir, logger)
    
    enc_scheduler.on_train_begin()
    dec_scheduler.on_train_begin()

    while True: # Loop until StopIteration
        try:
            metrics = None
            logger.start_epoch()
            for step in range(data_args.batches_per_epoch // (model_args.train_ratio + 1)):
                # encoder train
                logger.start_iter()
                msg = next(loader)
                metrics = model.train_encoder(msg)
                logger.log_iter(metrics)
                logger.end_iter()
                
                # decoder train
                for _ in range(model_args.train_ratio):
                    logger.start_iter()
                    msg = next(loader)
                    metrics = model.train_decoder(msg)
                    logger.log_iter(metrics)
                    logger.end_iter()
            logger.end_epoch(None)

            model.Pi.std *= model_args.sigma_decay

            enc_scheduler.on_epoch_end(logger.epoch, logs=metrics)
            dec_scheduler.on_epoch_end(logger.epoch, logs=metrics)

            if logger.has_improved():
                saver.save(model)

            if logger.notImprovedCounter >= 7:
                break
        except StopIteration:
            break

if __name__ == '__main__':
    parser = TrainArgParser()
    train(parser.parse_args())
