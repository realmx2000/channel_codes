import tensorflow as tf
from dataset import InputDataloader
from args import ArgParser

parser = ArgParser()
args = parser.parse()
num_examples = args.num_epochs * args.batches_per_epoch * args.batch_size
loader = InputDataloader(args.batch_size, args.block_length, num_examples, True)