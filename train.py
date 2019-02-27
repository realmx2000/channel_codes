import argparse
import copy
import json
import pathlib
import tensorflow as tf
from dataset import InputDataloader, AWGN
from logger import TrainLogger
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

    logger = TrainLogger(logger_args.save_dir, logger_args.name,
                         data_args.num_epochs, logger_args.iters_per_print)
  
    enc_scheduler.on_train_begin()
    dec_scheduler.on_train_begin()

    while True: # Loop until StopIteration
        try:
            metrics = None
            logger.start_epoch()
            for step in range(data_args.batches_per_epoch % (model_args.train_ratio + 1)):
                
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

            # Should these be joint or separate?
            # TODO: debug this part
            enc_scheduler.on_epoch_end(logger.epoch, logs=metrics)
            dec_scheduler.on_epoch_end(logger.epoch, logs=metrics)
        except StopIteration:
            break

if __name__ == '__main__':
    parser = TrainArgParser()
    train(parser.parse_args())
