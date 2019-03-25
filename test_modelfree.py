from dataset import InputDataloader, PowerConstraint, get_channel
from models import AutoEncoder, get_scheduler
from models.optim_util import get_possible_inputs
from saver import ModelSaver

from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dict_path",
        help="model dictionary path"
    )

    return parser.parse_args()


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


def load_model_dict(path):
    with open(path) as f:
        model_dict = json.load(f)
    return model_dict

def test(args):

    model_dict = load_model_dict(Path(args.model_dict_path))

    BER = []
    loss = []
    noises = []

    for noise, save_dir in model_dict.items():
        model_args, data_args = load_args(Path(save_dir))
        assert model_args.modelfree, "Code only evaluates on model free"
        
        saver = ModelSaver(Path(save_dir), None)
        power_constraint = PowerConstraint()
        possible_inputs = get_md_set(model_args.md_len)

        # TODO: change to batch size and batch per epoch to 1000
        data_args.batch_size = 100
        data_args.batches_per_epoch = 100
        dataset_size = data_args.batch_size * data_args.batches_per_epoch
        loader = InputDataloader(data_args.batch_size, data_args.block_length, dataset_size)
        loader = loader.example_generator()

        if data_args.channel == "AWGN":
            assert float(noise) == data_args.SNR
        else:
            assert float(noise) == data_args.epsilon

        print(f"Testing {noise} noise level")

        accuracy = []
        losses = []

        channel = get_channel(data_args.channel, model_args.modelfree, data_args)
        model = AutoEncoder(model_args, data_args, power_constraint, channel, possible_inputs)
        channel = get_channel(data_args.channel, model_args.modelfree, data_args)
        model = AutoEncoder(model_args, data_args, power_constraint, channel, possible_inputs)
        saver.load(model)
        for step in tqdm(range(data_args.batches_per_epoch)):
            msg = next(loader)
            metrics = model.trainable_encoder.test_on_batch(msg, msg)
            losses.append(metrics[0])
            accuracy.append(metrics[1])
        mean_loss = sum(losses) / len(losses)
        mean_BER = 1 - sum(accuracy) / len(accuracy)
        loss.append(mean_loss)
        BER.append(mean_BER)
        noises.append(noise)
        print(f"mean BER: {mean_BER}")
        print(f"mean loss: {mean_loss}")

    # create plots for results
    plt.plot(noises, BER, 'b--')
    plt.plot(noises, BER, 'bx')
    plt.ylabel("BER")
    plt.xlabel("noise")
    plt.yscale('log')
    plt.ylim([1e-6, 1.0])
    plt.savefig("figures/figure.png")

if __name__ == '__main__':
    test(get_args())
