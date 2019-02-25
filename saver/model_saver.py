import tensorflow as tf
from tf.train import Saver


class ModelSaver(object):
    """Class to save tensorflow sessions"""

    def __init__(self, save_dir, logger):
        """Init saver/loader"""
        self.model_save_dir = save_dir / "model.ckpt"
        self.logger = logger
        self.saver = Saver()


    def save(self, sess):
        self.saver.save(sess, self.model_save_dir)


    def load(self, sess):
        self.saver.load(sess, self.model_save_dir)
