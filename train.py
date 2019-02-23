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
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
for i in range(100):
    val = sess.run(encoder.forward(next))
    print(val)
    input()

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
