import tensorflow as tf
from dataset import InputDataloader
from args import ArgParser
from models import Encoder

parser = ArgParser()
args = parser.parse_args()
num_examples = args.num_epochs * args.batches_per_epoch * args.batch_size
loader = InputDataloader(args.batch_size, args.block_length, num_examples, True)

it = loader.get_loader()
next = it.get_next()
sess = tf.Session()
encoder = Encoder(100, 25, 2, 1/2, False)
for i in range(100):
    val = sess.run(encoder.forward(next))
    print(val)
    input()